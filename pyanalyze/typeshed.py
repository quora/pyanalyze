"""

Code for getting annotations from typeshed (and from third-party stubs generally).

"""

import ast
import builtins
import collections.abc
import enum
import inspect
import sys
import types
from abc import abstractmethod
from collections.abc import Collection, MutableMapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from enum import EnumMeta
from types import GeneratorType, MethodDescriptorType, ModuleType
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import qcore
import typeshed_client
from typing_extensions import Protocol

from pyanalyze import node_visitor
from pyanalyze.functions import translate_vararg_type

from .analysis_lib import is_positional_only_arg_name
from .annotations import (
    Context,
    DecoratorValue,
    SyntheticEvaluator,
    TypeQualifierValue,
    make_type_var_value,
    type_from_value,
    value_from_ast,
)
from .error_code import Error, ErrorCode
from .extensions import deprecated as deprecated_decorator
from .extensions import evaluated, overload, real_overload
from .node_visitor import Failure
from .options import Options, PathSequenceOption
from .safe import hasattr_static, is_typing_name, safe_isinstance
from .signature import (
    ConcreteSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
    make_bound_method,
)
from .stacked_scopes import Composite, uniq_chain
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    DeprecatedExtension,
    Extension,
    GenericValue,
    KnownValue,
    SubclassValue,
    SyntheticModuleValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeVarValue,
    Value,
    annotate_value,
    extract_typevars,
    make_coro_type,
    unannotate_value,
)

PROPERTY_LIKE = {KnownValue(property), KnownValue(types.DynamicClassAttribute)}

if sys.version_info >= (3, 11):
    PROPERTY_LIKE.add(KnownValue(enum.property))


T_co = TypeVar("T_co", covariant=True)


@dataclass
class _AnnotationContext(Context):
    finder: "TypeshedFinder"
    module: str

    def show_error(
        self,
        message: str,
        error_code: Error = ErrorCode.invalid_annotation,
        node: Optional[ast.AST] = None,
    ) -> None:
        self.finder.log(message, ())

    def get_name(self, node: ast.Name) -> Value:
        return self.finder.resolve_name(self.module, node.id)

    def get_attribute(self, root_value: Value, node: ast.Attribute) -> Value:
        if isinstance(root_value, KnownValue):
            if isinstance(root_value.val, ModuleType):
                return self.finder.resolve_name(root_value.val.__name__, node.attr)
        elif isinstance(root_value, SyntheticModuleValue):
            return self.finder.resolve_name(".".join(root_value.module_path), node.attr)
        return super().get_attribute(root_value, node)


class _DummyErrorContext:
    all_failures: List[Failure] = []

    def show_error(
        self,
        node: ast.AST,
        e: str,
        error_code: node_visitor.ErrorCodeInstance,
        *,
        detail: Optional[str] = None,
        save: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Failure]:
        return None


class StubPath(PathSequenceOption):
    """Extra paths in which to look for stubs."""

    name = "stub_path"


# These are specified as just "List = _Alias()" in typing.pyi. Redirect
# them to the proper runtime equivalent.
_TYPING_ALIASES = {
    "typing.List": "builtins.list",
    "typing.Dict": "builtins.dict",
    "typing.DefaultDict": "collections.defaultdict",
    "typing.Set": "builtins.set",
    "typing.Frozenzet": "builtins.frozenset",
    "typing.Counter": "collections.Counter",
    "typing.Deque": "collections.deque",
    "typing.ChainMap": "collections.ChainMap",
    "typing.OrderedDict": "collections.OrderedDict",
    "typing.Tuple": "builtins.tuple",
}


@dataclass
class TypeshedFinder:
    ctx: CanAssignContext = field(repr=False)
    verbose: bool = True
    resolver: typeshed_client.Resolver = field(default_factory=typeshed_client.Resolver)
    _assignment_cache: Dict[Tuple[str, ast.AST], Value] = field(
        default_factory=dict, repr=False, init=False
    )
    _attribute_cache: Dict[Tuple[str, str, bool], Value] = field(
        default_factory=dict, repr=False, init=False
    )
    _active_infos: List[typeshed_client.resolver.ResolvedName] = field(
        default_factory=list, repr=False, init=False
    )

    @classmethod
    def make(
        cls,
        can_assign_ctx: CanAssignContext,
        options: Options,
        *,
        verbose: bool = False,
    ) -> "TypeshedFinder":
        extra_paths = options.get_value_for(StubPath)
        ctx = typeshed_client.get_search_context()
        ctx = typeshed_client.get_search_context(
            search_path=[*ctx.search_path, *extra_paths]
        )
        resolver = typeshed_client.Resolver(ctx)
        return TypeshedFinder(can_assign_ctx, verbose, resolver)

    def log(self, message: str, obj: object) -> None:
        if not self.verbose:
            return
        print(f"{message}: {obj!r}")

    def _get_sig_from_method_descriptor(
        self, obj: MethodDescriptorType, allow_call: bool
    ) -> Optional[ConcreteSignature]:
        objclass = obj.__objclass__
        fq_name = self._get_fq_name(objclass)
        if fq_name is None:
            return None
        info = self._get_info_for_name(fq_name)
        sig = self._get_method_signature_from_info(
            info, obj, fq_name, objclass.__module__, objclass, allow_call=allow_call
        )
        return sig

    def get_argspec(
        self,
        obj: object,
        *,
        allow_call: bool = False,
        type_params: Sequence[Value] = (),
    ) -> Optional[ConcreteSignature]:
        if isinstance(obj, str):
            # Synthetic type
            return self.get_argspec_for_fully_qualified_name(
                obj, obj, type_params=type_params
            )
        if inspect.ismethoddescriptor(obj) and hasattr_static(obj, "__objclass__"):
            return self._get_sig_from_method_descriptor(obj, allow_call)
        if inspect.isbuiltin(obj) and isinstance(obj.__self__, type):
            # This covers cases like dict.fromkeys and type.__subclasses__. We
            # want to make sure we get the underlying method descriptor object,
            # which we can apparently only get out of the __dict__.
            method = obj.__self__.__dict__.get(obj.__name__)
            if (
                method is not None
                and inspect.ismethoddescriptor(method)
                and hasattr_static(method, "__objclass__")
            ):
                sig = self._get_sig_from_method_descriptor(method, allow_call)
                if sig is None:
                    return None
                bound = make_bound_method(
                    sig, Composite(TypedValue(obj.__self__)), ctx=self.ctx
                )
                if bound is None:
                    return None
                return bound.get_signature(ctx=self.ctx)

        if inspect.ismethod(obj):
            self.log("Ignoring method", obj)
            return None
        if (
            hasattr_static(obj, "__qualname__")
            and hasattr_static(obj, "__name__")
            and hasattr_static(obj, "__module__")
            and isinstance(obj.__qualname__, str)
            and obj.__qualname__ != obj.__name__
            and "." in obj.__qualname__
        ):
            parent_name, own_name = obj.__qualname__.rsplit(".", maxsplit=1)
            # Work around the stub using the wrong name.
            # TODO we should be able to resolve this anyway.
            if parent_name == "EnumType" and obj.__module__ == "enum":
                parent_fqn = "enum.EnumMeta"
            else:
                parent_fqn = f"{obj.__module__}.{parent_name}"
            parent_info = self._get_info_for_name(parent_fqn)
            if parent_info is not None:
                maybe_info = self._get_child_info(parent_info, own_name, obj.__module__)
                if maybe_info is not None:
                    info, mod = maybe_info
                    fq_name = f"{parent_fqn}.{own_name}"
                    sig = self._get_signature_from_info(
                        info, obj, fq_name, mod, allow_call=allow_call
                    )
                    return sig

        fq_name = self._get_fq_name(obj)
        if fq_name is None:
            return None
        return self.get_argspec_for_fully_qualified_name(
            fq_name, obj, allow_call=allow_call, type_params=type_params
        )

    def get_argspec_for_fully_qualified_name(
        self,
        fq_name: str,
        obj: object,
        *,
        allow_call: bool = False,
        type_params: Sequence[Value] = (),
    ) -> Optional[ConcreteSignature]:
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        sig = self._get_signature_from_info(
            info, obj, fq_name, mod, allow_call=allow_call, type_params=type_params
        )
        return sig

    def get_bases(self, typ: type) -> Optional[List[Value]]:
        """Return the base classes for this type, including generic bases."""
        return self.get_bases_for_value(TypedValue(typ))

    def get_bases_for_value(self, val: Value) -> Optional[List[Value]]:
        if isinstance(val, TypedValue):
            if isinstance(val.typ, type):
                typ = val.typ
                # The way AbstractSet/Set is handled between collections and typing is
                # too confusing, just hardcode it.
                if typ is AbstractSet:
                    return [GenericValue(Collection, (TypeVarValue(T_co),))]
                if typ is collections.abc.Callable:
                    return None
                # In 3.11 it's named EnumType and EnumMeta is an alias, but the
                # stubs have it the other way around. We can't deal with that for now.
                if typ is EnumMeta:
                    return [TypedValue(type)]
                fq_name = self._get_fq_name(typ)
                if fq_name is None:
                    return None
            else:
                fq_name = val.typ
                if fq_name == "collections.abc.Set":
                    return [GenericValue(Collection, (TypeVarValue(T_co),))]
                elif fq_name == "contextlib.AbstractContextManager":
                    return [GenericValue(Protocol, (TypeVarValue(T_co),))]
                elif fq_name in ("typing.Callable", "collections.abc.Callable"):
                    return None
                elif is_typing_name(fq_name, "TypedDict"):
                    return [
                        GenericValue(
                            MutableMapping, [TypedValue(str), TypedValue(object)]
                        )
                    ]
            return self.get_bases_for_fq_name(fq_name)
        return None

    def is_protocol(self, typ: type) -> bool:
        """Return whether this type is marked as a Protocol in the stubs."""
        fq_name = self._get_fq_name(typ)
        if fq_name is None:
            return False
        bases = self.get_bases_for_value(TypedValue(fq_name))
        if bases is None:
            return False
        return any(
            isinstance(base, TypedValue) and is_typing_name(base.typ, "Protocol")
            for base in bases
        )

    def get_bases_recursively(self, typ: Union[type, str]) -> List[Value]:
        stack = [TypedValue(typ)]
        seen = set()
        bases = []
        # TODO return MRO order
        while stack:
            next_base = stack.pop()
            if next_base in seen:
                continue
            seen.add(next_base)
            bases.append(next_base)
            new_bases = self.get_bases_for_value(next_base)
            if new_bases is not None:
                bases += new_bases
        return bases

    def get_bases_for_fq_name(self, fq_name: str) -> Optional[List[Value]]:
        if fq_name in (
            "typing.Generic",
            "typing.Protocol",
            "typing_extensions.Protocol",
        ):
            return []
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._get_bases_from_info(info, mod, fq_name)

    def get_attribute(self, typ: type, attr: str, *, on_class: bool) -> Value:
        """Return the stub for this attribute.

        Does not look at parent classes. Returns UNINITIALIZED_VALUE if no
        stub can be found.

        """
        fq_name = self._get_fq_name(typ)
        if fq_name is None:
            return UNINITIALIZED_VALUE
        return self.get_attribute_for_fq_name(fq_name, attr, on_class=on_class)

    def get_attribute_for_fq_name(
        self, fq_name: str, attr: str, *, on_class: bool
    ) -> Value:
        key = (fq_name, attr, on_class)
        try:
            return self._attribute_cache[key]
        except KeyError:
            info = self._get_info_for_name(fq_name)
            mod, _ = fq_name.rsplit(".", maxsplit=1)
            val = self._get_attribute_from_info(info, mod, attr, on_class=on_class)
            self._attribute_cache[key] = val
            return val

    def get_attribute_recursively(
        self, fq_name: str, attr: str, *, on_class: bool
    ) -> Tuple[Value, Union[type, str, None]]:
        """Get an attribute from a fully qualified class.

        Returns a tuple (value, provider).

        """
        for base in self.get_bases_recursively(fq_name):
            if isinstance(base, TypedValue):
                if isinstance(base.typ, str):
                    possible_value = self.get_attribute_for_fq_name(
                        base.typ, attr, on_class=on_class
                    )
                else:
                    possible_value = self.get_attribute(
                        base.typ, attr, on_class=on_class
                    )
                if possible_value is not UNINITIALIZED_VALUE:
                    return possible_value, base.typ
        return UNINITIALIZED_VALUE, None

    def has_attribute(self, typ: Union[type, str], attr: str) -> bool:
        """Whether this type has this attribute in the stubs.

        Also looks at base classes.

        """
        if self._has_own_attribute(typ, attr):
            return True
        bases = self.get_bases_for_value(TypedValue(typ))
        if bases is not None:
            for base in bases:
                if not isinstance(base, TypedValue):
                    continue
                typ = base.typ
                if typ is Generic or is_typing_name(typ, "Protocol"):
                    continue
                if self.has_attribute(base.typ, attr):
                    return True
        return False

    def get_all_attributes(self, typ: Union[type, str]) -> Set[str]:
        if isinstance(typ, str):
            fq_name = typ
        else:
            fq_name = self._get_fq_name(typ)
            if fq_name is None:
                return set()
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._get_all_attributes_from_info(info, mod)

    def has_stubs(self, typ: type) -> bool:
        fq_name = self._get_fq_name(typ)
        if fq_name is None:
            return False
        info = self._get_info_for_name(fq_name)
        return info is not None

    def resolve_name(self, module: str, name: str) -> Value:
        info = self._get_info_for_name(f"{module}.{name}")
        if info is not None:
            return self._value_from_info(info, module)
        elif hasattr(builtins, name):
            val = getattr(builtins, name)
            if val is None or isinstance(val, type):
                return KnownValue(val)
        # TODO change to UNINITIALIZED_VALUE
        return AnyValue(AnySource.inference)

    def _get_attribute_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        mod: str,
        attr: str,
        *,
        on_class: bool,
        is_typeddict: bool = False,
    ) -> Value:
        if info is None:
            return UNINITIALIZED_VALUE
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_attribute_from_info(
                info.info, ".".join(info.source_module), attr, on_class=on_class
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    child_info = info.child_nodes[attr]
                    if isinstance(child_info, typeshed_client.NameInfo):
                        return self._get_value_from_child_info(
                            child_info.ast,
                            mod,
                            is_typeddict=is_typeddict,
                            on_class=on_class,
                            parent_name=info.ast.name,
                        )
                    assert False, repr(child_info)
                return UNINITIALIZED_VALUE
            elif isinstance(info.ast, ast.Assign):
                val = self._parse_type(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.get_attribute(val.val, attr, on_class=on_class)
                else:
                    return UNINITIALIZED_VALUE
            else:
                return UNINITIALIZED_VALUE
        return UNINITIALIZED_VALUE

    def _get_value_from_child_info(
        self,
        node: Union[
            ast.AST, typeshed_client.OverloadedName, typeshed_client.ImportedName
        ],
        mod: str,
        *,
        is_typeddict: bool,
        on_class: bool,
        parent_name: str,
    ) -> Value:
        if isinstance(node, ast.AnnAssign):
            return self._parse_type(node.annotation, mod, is_typeddict=is_typeddict)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_property = False
            for decorator_node in node.decorator_list:
                decorator_value = self._parse_expr(decorator_node, mod)
                if decorator_value in PROPERTY_LIKE:
                    is_property = True
            if is_property:
                if node.returns:
                    return self._parse_type(node.returns, mod)
                else:
                    return AnyValue(AnySource.unannotated)
            else:
                # TODO: apply decorators to the return value
                sig = self._get_signature_from_func_def(
                    node, None, mod, autobind=not on_class
                )
                if sig is None:
                    return AnyValue(AnySource.inference)
                else:
                    return CallableValue(sig)
        elif isinstance(node, ast.ClassDef):
            # should be
            # SubclassValue(TypedValue(f"{mod}.{parent_name}.{node.name}"), exactly=True)
            # but that doesn't currently work well
            return AnyValue(AnySource.inference)
        elif isinstance(node, ast.Assign):
            return UNINITIALIZED_VALUE
        elif isinstance(node, typeshed_client.OverloadedName):
            sigs = []
            for subnode in node.definitions:
                val = self._get_value_from_child_info(
                    subnode,
                    mod,
                    is_typeddict=is_typeddict,
                    on_class=on_class,
                    parent_name=parent_name,
                )
                sig = self._sig_from_value(val)
                if not isinstance(sig, Signature):
                    return AnyValue(AnySource.inference)
                sigs.append(sig)
            return CallableValue(OverloadedSignature(sigs))
        assert False, repr(node)

    def _get_child_info(
        self, info: typeshed_client.resolver.ResolvedName, attr: str, mod: str
    ) -> Optional[Tuple[typeshed_client.resolver.ResolvedName, str]]:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_child_info(info.info, attr, ".".join(info.source_module))
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    return info.child_nodes[attr], mod
                return None
            elif isinstance(info.ast, ast.Assign):
                return None  # TODO maybe we need this for aliased methods
            else:
                return None
        return None

    def _has_own_attribute(self, typ: Union[type, str], attr: str) -> bool:
        # Special case since otherwise we think every object has every attribute
        if typ is object and attr == "__getattribute__":
            return False
        if isinstance(typ, str):
            fq_name = typ
        else:
            fq_name = self._get_fq_name(typ)
            if fq_name is None:
                return False
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._has_attribute_from_info(info, mod, attr)

    def _get_all_attributes_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str
    ) -> Set[str]:
        if info is None:
            return set()
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_all_attributes_from_info(
                info.info, ".".join(info.source_module)
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                if info.child_nodes is not None:
                    return set(info.child_nodes)
            elif isinstance(info.ast, ast.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.get_all_attributes(val.val)
                else:
                    return set()
            else:
                return set()
        return set()

    def _has_attribute_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str, attr: str
    ) -> bool:
        if info is None:
            return False
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._has_attribute_from_info(
                info.info, ".".join(info.source_module), attr
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    return True
                return False
            elif isinstance(info.ast, ast.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.has_attribute(val.val, attr)
                else:
                    return False
            else:
                return False
        return False

    def _get_bases_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str, fq_name: str
    ) -> Optional[List[Value]]:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_bases_from_info(
                info.info, ".".join(info.source_module), fq_name
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast.ClassDef):
                bases = info.ast.bases
                return [self._parse_type(base, mod) for base in bases]
            elif isinstance(info.ast, ast.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    new_fq_name = self._get_fq_name(val.val)
                    if fq_name == new_fq_name:
                        # prevent infinite recursion
                        return [AnyValue(AnySource.inference)]
                    return self.get_bases(val.val)
                else:
                    return [AnyValue(AnySource.inference)]
            elif isinstance(
                info.ast,
                (
                    # overloads are not supported yet
                    typeshed_client.OverloadedName,
                    typeshed_client.ImportedName,
                    # typeshed pretends the class is a function
                    ast.FunctionDef,
                ),
            ):
                return None
            else:
                raise NotImplementedError(ast.dump(info.ast))
        return None

    def _get_method_signature_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
        objclass: type,
        *,
        allow_call: bool = False,
    ) -> Optional[ConcreteSignature]:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_method_signature_from_info(
                info.info,
                obj,
                fq_name,
                ".".join(info.source_module),
                objclass,
                allow_call=allow_call,
            )
        elif isinstance(info, typeshed_client.NameInfo):
            # Note that this doesn't handle names inherited from base classes
            if info.child_nodes and obj.__name__ in info.child_nodes:
                child_info = info.child_nodes[obj.__name__]
                return self._get_signature_from_info(
                    child_info, obj, fq_name, mod, objclass, allow_call=allow_call
                )
            else:
                return None
        else:
            self.log("Ignoring unrecognized info", (fq_name, info))
            return None

    def _get_fq_name(self, obj: Any) -> Optional[str]:
        if obj is GeneratorType:
            return "typing.Generator"
        # It claims to be io.open, but typeshed puts it in builtins
        if obj is open:
            return "builtins.open"
        try:
            module_name = obj.__module__
            if module_name is None:
                module_name = "builtins"
            # Objects like io.BytesIO are technically in the _io module,
            # but typeshed puts them in io, which at runtime just re-exports
            # them.
            if module_name == "_io":
                module_name = "io"
            fq_name = ".".join([module_name, obj.__qualname__])
            # Avoid looking for stubs we won't find anyway.
            if not _obj_from_qualname_is(module_name, obj.__qualname__, obj):
                self.log("Ignoring invalid name", fq_name)
                return None
            return _TYPING_ALIASES.get(fq_name, fq_name)
        except (AttributeError, TypeError):
            self.log("Ignoring object without module or qualname", obj)
            return None

    def _sig_from_value(self, val: Value) -> Optional[ConcreteSignature]:
        val, extensions = unannotate_value(val, DeprecatedExtension)
        if isinstance(val, AnnotatedValue):
            val = val.value
        if not isinstance(val, CallableValue):
            return None
        sig = val.signature
        if isinstance(sig, Signature):
            for extension in extensions:
                sig = replace(sig, deprecated=extension.deprecation_message)
        return sig

    def _get_signature_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
        objclass: Optional[type] = None,
        *,
        allow_call: bool = False,
        type_params: Sequence[Value] = (),
    ) -> Optional[ConcreteSignature]:
        if isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return self._get_signature_from_func_def(
                    info.ast, obj, mod, objclass, allow_call=allow_call
                )
            elif isinstance(info.ast, typeshed_client.OverloadedName):
                sigs = []
                for defn in info.ast.definitions:
                    if not isinstance(defn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.log(
                            "Ignoring unrecognized AST in overload", (fq_name, info)
                        )
                        return None
                    sig = self._get_signature_from_func_def(
                        defn, obj, mod, objclass, allow_call=allow_call
                    )
                    if sig is None:
                        self.log("Could not get sig for overload member", (defn,))
                        return None
                    sigs.append(sig)
                return OverloadedSignature(sigs)
            elif isinstance(info.ast, ast.ClassDef):
                new_value, provider = self.get_attribute_recursively(
                    fq_name, "__new__", on_class=True
                )
                from_init = False
                if new_value is UNINITIALIZED_VALUE or provider is object:
                    init_value, provider = self.get_attribute_recursively(
                        fq_name, "__init__", on_class=True
                    )
                    if (sig := self._sig_from_value(init_value)) is not None:
                        from_init = True
                else:
                    sig = self._sig_from_value(new_value)
                if sig is not None:
                    if safe_isinstance(obj, type):
                        if allow_call:
                            if isinstance(sig, Signature):
                                sig = replace(sig, allow_call=True, callable=obj)
                            else:
                                sig = OverloadedSignature(
                                    [
                                        replace(sig, allow_call=True, callable=obj)
                                        for sig in sig.signatures
                                    ]
                                )
                        typ = obj
                    else:
                        typ = fq_name
                    if type_params:
                        self_val = GenericValue(typ, type_params)
                    else:
                        self_val = TypedValue(typ)
                    if from_init:
                        sig = sig.replace_return_value(self_val)
                        self_annotation_value = self_val
                    else:
                        self_annotation_value = SubclassValue(self_val)
                    bound_sig = make_bound_method(
                        sig, Composite(self_val), ctx=self.ctx
                    )
                    if bound_sig is None:
                        return None
                    sig = bound_sig.get_signature(
                        ctx=self.ctx, self_annotation_value=self_annotation_value
                    )
                    return sig

                return None
            else:
                self.log("Ignoring unrecognized AST", (fq_name, info))
                return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_signature_from_info(
                info.info,
                obj,
                fq_name,
                ".".join(info.source_module),
                objclass,
                allow_call=allow_call,
            )
        elif info is None:
            return None
        else:
            self.log("Ignoring unrecognized info", (fq_name, info))
            return None

    @qcore.caching.cached_per_instance()
    def _get_info_for_name(self, fq_name: str) -> typeshed_client.resolver.ResolvedName:
        return self.resolver.get_fully_qualified_name(fq_name)

    def _get_signature_from_func_def(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        obj: object,
        mod: str,
        objclass: Optional[type] = None,
        *,
        autobind: bool = False,
        allow_call: bool = False,
    ) -> Optional[Signature]:
        is_classmethod = is_staticmethod = is_evaluated = False
        deprecated = None
        for decorator_ast in node.decorator_list:
            decorator = self._parse_expr(decorator_ast, mod)
            if (
                decorator == KnownValue(abstractmethod)
                or decorator == KnownValue(overload)
                or decorator == KnownValue(real_overload)
            ):
                continue
            elif decorator == KnownValue(classmethod):
                is_classmethod = True
                if autobind:  # TODO support classmethods otherwise
                    continue
            elif decorator == KnownValue(staticmethod):
                is_staticmethod = True
                if autobind:  # TODO support staticmethods otherwise
                    continue
            elif decorator == KnownValue(evaluated):
                is_evaluated = True
                continue
            elif (
                isinstance(decorator, DecoratorValue)
                and decorator.decorator is deprecated_decorator
            ):
                arg = decorator.args[0]
                if isinstance(arg, KnownValue) and isinstance(arg.val, str):
                    deprecated = arg.val
            # something we don't recognize; ignore it
        if node.returns is None:
            return_value = AnyValue(AnySource.unannotated)
        else:
            return_value = self._parse_type(node.returns, mod)
        # ignore self type for class and static methods
        if node.decorator_list:
            objclass = None
        args = node.args
        arguments: List[SigParameter] = []
        num_pos_only_args = len(args.posonlyargs)
        defaults = args.defaults
        num_pos_only_defaults = len(defaults) - len(args.args)
        if num_pos_only_defaults > 0:
            num_without_default = num_pos_only_args - num_pos_only_defaults
            pos_only_defaults = [None] * num_without_default + defaults[
                :num_pos_only_defaults
            ]
            defaults = defaults[num_pos_only_defaults:]
        else:
            pos_only_defaults = [None for _ in args.posonlyargs]
        arguments += self._parse_param_list(
            args.posonlyargs,
            pos_only_defaults,
            mod,
            ParameterKind.POSITIONAL_ONLY,
            objclass,
        )

        num_without_defaults = len(args.args) - len(defaults)
        defaults = [None] * num_without_defaults + defaults
        arguments += self._parse_param_list(
            args.args, defaults, mod, ParameterKind.POSITIONAL_OR_KEYWORD, objclass
        )
        if autobind:
            if is_classmethod or not is_staticmethod:
                arguments = arguments[1:]

        if args.vararg is not None:
            arguments.append(
                self._parse_param(args.vararg, None, mod, ParameterKind.VAR_POSITIONAL)
            )
        arguments += self._parse_param_list(
            args.kwonlyargs, args.kw_defaults, mod, ParameterKind.KEYWORD_ONLY
        )
        if args.kwarg is not None:
            arguments.append(
                self._parse_param(args.kwarg, None, mod, ParameterKind.VAR_KEYWORD)
            )
        # some typeshed types have a positional-only after a normal argument,
        # and Signature doesn't like that
        seen_non_positional = False
        cleaned_arguments = []
        for arg in arguments:
            if arg.kind is ParameterKind.POSITIONAL_ONLY and seen_non_positional:
                cleaned_arguments = [
                    replace(arg, kind=ParameterKind.POSITIONAL_ONLY)
                    for arg in cleaned_arguments
                ]
                seen_non_positional = False
            else:
                seen_non_positional = True
            cleaned_arguments.append(arg)
        if is_evaluated:
            ctx = _AnnotationContext(self, mod)
            evaluator = SyntheticEvaluator(
                node, return_value, _DummyErrorContext(), ctx
            )
        else:
            evaluator = None
        return Signature.make(
            cleaned_arguments,
            callable=obj,
            return_annotation=(
                make_coro_type(return_value)
                if isinstance(node, ast.AsyncFunctionDef)
                else return_value
            ),
            allow_call=allow_call,
            evaluator=evaluator,
            deprecated=deprecated,
        )

    def _parse_param_list(
        self,
        args: Iterable[ast.arg],
        defaults: Iterable[Optional[ast.AST]],
        module: str,
        kind: ParameterKind,
        objclass: Optional[type] = None,
    ) -> Iterable[SigParameter]:
        for i, (arg, default) in enumerate(zip(args, defaults)):
            yield self._parse_param(
                arg, default, module, kind, objclass=objclass if i == 0 else None
            )

    def _parse_param(
        self,
        arg: ast.arg,
        default: Optional[ast.AST],
        module: str,
        kind: ParameterKind,
        *,
        objclass: Optional[type] = None,
    ) -> SigParameter:
        typ = AnyValue(AnySource.unannotated)
        if arg.annotation is not None:
            typ = self._parse_type(
                arg.annotation, module, allow_unpack=kind.allow_unpack()
            )
        elif objclass is not None:
            bases = self.get_bases(objclass)
            if bases is None:
                typ = TypedValue(objclass)
            else:
                typevars = uniq_chain(extract_typevars(base) for base in bases)
                if typevars:
                    typ = GenericValue(
                        objclass,
                        [
                            make_type_var_value(
                                tv,
                                _AnnotationContext(finder=self, module=tv.__module__),
                            )
                            for tv in typevars
                        ],
                    )
                else:
                    typ = TypedValue(objclass)

        name = arg.arg
        if kind is ParameterKind.POSITIONAL_OR_KEYWORD and is_positional_only_arg_name(
            name
        ):
            kind = ParameterKind.POSITIONAL_ONLY
            name = name[2:]
        typ = translate_vararg_type(kind, typ, self.ctx)
        # Mark self as positional-only. objclass should be given only if we believe
        # it's the "self" parameter.
        if objclass is not None:
            kind = ParameterKind.POSITIONAL_ONLY
        if default is None:
            return SigParameter(name, kind, annotation=typ)
        else:
            default_value = self._parse_expr(default, module)
            if default_value == KnownValue(...):
                default_value = AnyValue(AnySource.unannotated)
            return SigParameter(name, kind, annotation=typ, default=default_value)

    def _parse_expr(self, node: ast.AST, module: str) -> Value:
        ctx = _AnnotationContext(finder=self, module=module)
        return value_from_ast(node, ctx=ctx)

    def _parse_type(
        self,
        node: ast.AST,
        module: str,
        *,
        is_typeddict: bool = False,
        allow_unpack: bool = False,
    ) -> Value:
        val = self._parse_expr(node, module)
        ctx = _AnnotationContext(finder=self, module=module)
        typ = type_from_value(
            val, ctx=ctx, is_typeddict=is_typeddict, allow_unpack=allow_unpack
        )
        if self.verbose and isinstance(typ, AnyValue):
            self.log("Got Any", (ast.dump(node), module))
        return typ

    def _parse_call_assignment(
        self, info: typeshed_client.NameInfo, module: str
    ) -> Value:
        try:
            __import__(module)
            mod = sys.modules[module]
            return KnownValue(getattr(mod, info.name))
        except Exception:
            pass

        if not isinstance(info.ast, ast.Assign) or not isinstance(
            info.ast.value, ast.Call
        ):
            return AnyValue(AnySource.inference)
        ctx = _AnnotationContext(finder=self, module=module)
        return value_from_ast(info.ast.value, ctx=ctx)

    def _extract_metadata(self, module: str, node: ast.ClassDef) -> Sequence[Extension]:
        metadata = []
        for decorator in node.decorator_list:
            decorator_val = self._parse_expr(decorator, module)
            extension = self._extract_extension_from_decorator(decorator_val)
            if extension is not None:
                metadata.append(extension)
        return metadata

    def _extract_extension_from_decorator(
        self, decorator_val: Value
    ) -> Optional[Extension]:
        if (
            isinstance(decorator_val, DecoratorValue)
            and decorator_val.decorator is deprecated_decorator
        ):
            arg = decorator_val.args[0]
            if isinstance(arg, KnownValue) and isinstance(arg.val, str):
                return DeprecatedExtension(arg.val)
        return None

    def make_synthetic_type(self, module: str, info: typeshed_client.NameInfo) -> Value:
        fq_name = f"{module}.{info.name}"
        bases = self._get_bases_from_info(info, module, fq_name)
        typ = TypedValue(fq_name)
        if isinstance(info.ast, ast.ClassDef):
            metadata = self._extract_metadata(module, info.ast)
        else:
            metadata = []
        if bases is not None:
            if any(
                (isinstance(base, KnownValue) and is_typing_name(base.val, "TypedDict"))
                or isinstance(base, TypedDictValue)
                for base in bases
            ):
                typ = self._make_typeddict(module, info, bases)
        val = SubclassValue(typ, exactly=True)
        if metadata:
            return annotate_value(val, metadata)
        return val

    def _make_typeddict(
        self, module: str, info: typeshed_client.NameInfo, bases: Sequence[Value]
    ) -> TypedDictValue:
        total = True
        if isinstance(info.ast, ast.ClassDef):
            for keyword in info.ast.keywords:
                # TODO support PEP 728 here
                if keyword.arg == "total":
                    val = self._parse_expr(keyword.value, module)
                    if isinstance(val, KnownValue) and isinstance(val.val, bool):
                        total = val.val
        attrs = self._get_all_attributes_from_info(info, module)
        fields = [
            self._get_attribute_from_info(
                info, module, attr, on_class=True, is_typeddict=True
            )
            for attr in attrs
        ]
        items = {}
        for base in bases:
            if isinstance(base, TypedDictValue):
                items.update(base.items)
        items.update(
            {
                attr: self._make_td_value(field, total)
                for attr, field in zip(attrs, fields)
            }
        )
        return TypedDictValue(items)

    def _make_td_value(self, field: Value, total: bool) -> TypedDictEntry:
        readonly = False
        required = total
        while isinstance(field, TypeQualifierValue):
            if field.qualifier == "ReadOnly":
                readonly = True
            elif field.qualifier == "Required":
                required = True
            elif field.qualifier == "NotRequired":
                required = False
            field = field.value
        return TypedDictEntry(readonly=readonly, required=required, typ=field)

    def _value_from_info(
        self, info: typeshed_client.resolver.ResolvedName, module: str
    ) -> Value:
        # This guard against infinite recursion if a type refers to itself
        # (real-world example: os._ScandirIterator). Needs to change in
        # order to support recursive types.
        if info in self._active_infos:
            return AnyValue(AnySource.inference)
        self._active_infos.append(info)
        try:
            return self._value_from_info_inner(info, module)
        finally:
            self._active_infos.pop()

    def _value_from_info_inner(
        self, info: typeshed_client.resolver.ResolvedName, module: str
    ) -> Value:
        if isinstance(info, typeshed_client.ImportedInfo):
            return self._value_from_info(info.info, ".".join(info.source_module))
        elif isinstance(info, typeshed_client.NameInfo):
            fq_name = f"{module}.{info.name}"
            if fq_name in _TYPING_ALIASES:
                new_fq_name = _TYPING_ALIASES[fq_name]
                info = self._get_info_for_name(new_fq_name)
                return self._value_from_info(
                    info, new_fq_name.rsplit(".", maxsplit=1)[0]
                )
            if isinstance(info.ast, ast.Assign):
                key = (module, info.ast)
                if key in self._assignment_cache:
                    return self._assignment_cache[key]
                if isinstance(info.ast.value, ast.Call):
                    value = self._parse_call_assignment(info, module)
                else:
                    value = self._parse_expr(info.ast.value, module)
                self._assignment_cache[key] = value
                return value
            try:
                __import__(module)
                mod = sys.modules[module]
                return KnownValue(getattr(mod, info.name))
            except Exception:
                if isinstance(info.ast, ast.ClassDef):
                    return self.make_synthetic_type(module, info)
                elif isinstance(info.ast, ast.AnnAssign):
                    val = self._parse_type(info.ast.annotation, module)
                    if val != AnyValue(AnySource.incomplete_annotation):
                        return val
                    if info.ast.value:
                        return self._parse_expr(info.ast.value, module)
                elif isinstance(
                    info.ast,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        typeshed_client.OverloadedName,
                    ),
                ):
                    sig = self._get_signature_from_info(info, None, fq_name, module)
                    if sig is not None:
                        return CallableValue(sig)
                self.log("Unable to import", (module, info))
                return AnyValue(AnySource.inference)
        elif isinstance(info, tuple):
            module_path = ".".join(info)
            try:
                __import__(module_path)
                return KnownValue(sys.modules[module_path])
            except Exception:
                return SyntheticModuleValue(info)
        else:
            self.log("Ignoring info", info)
            return AnyValue(AnySource.inference)


def _obj_from_qualname_is(module_name: str, qualname: str, obj: object) -> bool:
    try:
        if module_name not in sys.modules:
            __import__(module_name)
        mod = sys.modules[module_name]
        actual = mod
        for piece in qualname.split("."):
            actual = getattr(actual, piece)
        return obj is actual
    except Exception:
        return False
