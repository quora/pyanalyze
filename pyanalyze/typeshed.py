"""

Code for getting annotations from typeshed (and from third-party stubs generally).

"""

from .annotations import Context, type_from_value, value_from_ast
from .error_code import ErrorCode
from .safe import is_typing_name
from .stacked_scopes import uniq_chain
from .signature import ConcreteSignature, OverloadedSignature, SigParameter, Signature
from .value import (
    AnySource,
    AnyValue,
    CallableValue,
    TypedValue,
    GenericValue,
    KnownValue,
    UNINITIALIZED_VALUE,
    Value,
    TypeVarValue,
    extract_typevars,
)

from abc import abstractmethod
import ast
import builtins
from collections.abc import (
    Awaitable,
    Collection,
    MutableMapping,
    Set as AbstractSet,
    Sized,
)
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import collections.abc
import qcore
import inspect
import sys
from types import GeneratorType
from typing import (
    Dict,
    Set,
    Tuple,
    cast,
    Any,
    Generic,
    Iterable,
    Optional,
    Union,
    Callable,
    List,
    TypeVar,
    overload,
)
from typing_extensions import Protocol, TypedDict
import typeshed_client
from typed_ast import ast3


T_co = TypeVar("T_co", covariant=True)


IS_PRE_38: bool = sys.version_info < (3, 8)


@dataclass
class _AnnotationContext(Context):
    finder: "TypeshedFinder"
    module: str

    def show_error(
        self, message: str, error_code: ErrorCode = ErrorCode.invalid_annotation
    ) -> None:
        self.finder.log(message, ())

    def get_name(self, node: ast.Name) -> Value:
        return self.finder.resolve_name(self.module, node.id)


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
    verbose: bool = True
    resolver: typeshed_client.Resolver = field(default_factory=typeshed_client.Resolver)
    _assignment_cache: Dict[Tuple[str, ast3.AST], Value] = field(
        default_factory=dict, repr=False, init=False
    )
    _attribute_cache: Dict[Tuple[str, str, bool], Value] = field(
        default_factory=dict, repr=False, init=False
    )

    def log(self, message: str, obj: object) -> None:
        if not self.verbose:
            return
        print("%s: %r" % (message, obj))

    def get_argspec(
        self, obj: object, *, allow_call: bool = False
    ) -> Optional[ConcreteSignature]:
        if inspect.ismethoddescriptor(obj) and hasattr(obj, "__objclass__"):
            objclass = obj.__objclass__
            fq_name = self._get_fq_name(objclass)
            if fq_name is None:
                return None
            info = self._get_info_for_name(fq_name)
            sig = self._get_method_signature_from_info(
                info, obj, fq_name, objclass.__module__, objclass, allow_call=allow_call
            )
            return sig

        if inspect.ismethod(obj):
            self.log("Ignoring method", obj)
            return None
        if (
            hasattr(obj, "__qualname__")
            and hasattr(obj, "__name__")
            and hasattr(obj, "__module__")
            and obj.__qualname__ != obj.__name__
            and "." in obj.__qualname__
        ):
            parent_name, own_name = obj.__qualname__.rsplit(".", maxsplit=1)
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
            fq_name, obj, allow_call=allow_call
        )

    def get_argspec_for_fully_qualified_name(
        self, fq_name: str, obj: object, *, allow_call: bool = False
    ) -> Optional[ConcreteSignature]:
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        sig = self._get_signature_from_info(
            info, obj, fq_name, mod, allow_call=allow_call
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
                # too confusing, just hardcode it. Same for (Abstract)ContextManager.
                if typ is AbstractSet:
                    return [GenericValue(Collection, (TypeVarValue(T_co),))]
                if typ is AbstractContextManager:
                    return [GenericValue(Generic, (TypeVarValue(T_co),))]
                if typ is Callable or typ is collections.abc.Callable:
                    return None
                if typ is TypedDict:
                    return [
                        GenericValue(
                            MutableMapping, [TypedValue(str), TypedValue(object)]
                        )
                    ]
                fq_name = self._get_fq_name(typ)
                if fq_name is None:
                    return None
            else:
                fq_name = val.typ
                if fq_name == "collections.abc.Set":
                    return [GenericValue(Collection, (TypeVarValue(T_co),))]
                elif fq_name == "contextlib.AbstractContextManager":
                    return [GenericValue(Generic, (TypeVarValue(T_co),))]
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
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._get_bases_from_info(info, mod)

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
        return AnyValue(AnySource.inference)

    def _get_attribute_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        mod: str,
        attr: str,
        *,
        on_class: bool,
    ) -> Value:
        if info is None:
            return UNINITIALIZED_VALUE
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_attribute_from_info(
                info.info, ".".join(info.source_module), attr, on_class=on_class
            )
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast3.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    child_info = info.child_nodes[attr]
                    if isinstance(child_info, typeshed_client.NameInfo):
                        if isinstance(child_info.ast, ast3.AnnAssign):
                            return self._parse_type(child_info.ast.annotation, mod)
                        elif isinstance(child_info.ast, ast3.FunctionDef):
                            decorators = [
                                self._parse_expr(decorator, mod)
                                for decorator in child_info.ast.decorator_list
                            ]
                            if child_info.ast.returns and decorators == [
                                KnownValue(property)
                            ]:
                                return self._parse_type(child_info.ast.returns, mod)
                            sig = self._get_signature_from_func_def(
                                child_info.ast, None, mod, autobind=not on_class
                            )
                            if sig is None:
                                return AnyValue(AnySource.inference)
                            else:
                                return CallableValue(sig)
                        elif isinstance(child_info.ast, ast3.AsyncFunctionDef):
                            return UNINITIALIZED_VALUE
                        elif isinstance(child_info.ast, ast3.Assign):
                            return UNINITIALIZED_VALUE
                    assert False, repr(child_info)
                return UNINITIALIZED_VALUE
            elif isinstance(info.ast, ast3.Assign):
                val = self._parse_type(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.get_attribute(val.val, attr, on_class=on_class)
                else:
                    return UNINITIALIZED_VALUE
            else:
                return UNINITIALIZED_VALUE
        return UNINITIALIZED_VALUE

    def _get_child_info(
        self, info: typeshed_client.resolver.ResolvedName, attr: str, mod: str
    ) -> Optional[Tuple[typeshed_client.resolver.ResolvedName, str]]:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_child_info(info.info, attr, ".".join(info.source_module))
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast3.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    return info.child_nodes[attr], mod
                return None
            elif isinstance(info.ast, ast3.Assign):
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
            if isinstance(info.ast, ast3.ClassDef):
                if info.child_nodes is not None:
                    return set(info.child_nodes)
            elif isinstance(info.ast, ast3.Assign):
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
            if isinstance(info.ast, ast3.ClassDef):
                if info.child_nodes and attr in info.child_nodes:
                    return True
                return False
            elif isinstance(info.ast, ast3.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.has_attribute(val.val, attr)
                else:
                    return False
            else:
                return False
        return False

    def _get_bases_from_info(
        self, info: typeshed_client.resolver.ResolvedName, mod: str
    ) -> Optional[List[Value]]:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_bases_from_info(info.info, ".".join(info.source_module))
        elif isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast3.ClassDef):
                bases = info.ast.bases
                return [self._parse_type(base, mod) for base in bases]
            elif isinstance(info.ast, ast3.Assign):
                val = self._parse_type(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.get_bases(val.val)
                else:
                    return [val]
            elif isinstance(
                info.ast,
                (
                    # overloads are not supported yet
                    typeshed_client.OverloadedName,
                    typeshed_client.ImportedName,
                    # typeshed pretends the class is a function
                    ast3.FunctionDef,
                ),
            ):
                return None
            else:
                raise NotImplementedError(ast3.dump(info.ast))
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
        if IS_PRE_38:
            if obj is Sized:
                return "typing.Sized"
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

    def _get_signature_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
        objclass: Optional[type] = None,
        *,
        allow_call: bool = False,
    ) -> Optional[ConcreteSignature]:
        if isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, (ast3.FunctionDef, ast3.AsyncFunctionDef)):
                return self._get_signature_from_func_def(
                    info.ast, obj, mod, objclass, allow_call=allow_call
                )
            elif isinstance(info.ast, typeshed_client.OverloadedName):
                sigs = []
                for defn in info.ast.definitions:
                    if not isinstance(defn, (ast3.FunctionDef, ast3.AsyncFunctionDef)):
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
        node: Union[ast3.FunctionDef, ast3.AsyncFunctionDef],
        obj: object,
        mod: str,
        objclass: Optional[type] = None,
        *,
        autobind: bool = False,
        allow_call: bool = False,
    ) -> Optional[Signature]:
        is_classmethod = is_staticmethod = False
        for decorator_ast in node.decorator_list:
            decorator = self._parse_expr(decorator_ast, mod)
            if decorator == KnownValue(abstractmethod) or decorator == KnownValue(
                overload
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
            # might be @overload or something else we don't recognize
            return None
        if node.returns is None:
            return_value = AnyValue(AnySource.unannotated)
        else:
            return_value = self._parse_type(node.returns, mod)
        # ignore self type for class and static methods
        if node.decorator_list:
            objclass = None
        args = node.args

        num_without_defaults = len(args.args) - len(args.defaults)
        defaults = [None] * num_without_defaults + args.defaults
        arguments = list(
            self._parse_param_list(
                args.args, defaults, mod, SigParameter.POSITIONAL_OR_KEYWORD, objclass
            )
        )
        if autobind:
            if is_classmethod or not is_staticmethod:
                arguments = arguments[1:]

        if args.vararg is not None:
            vararg_param = self._parse_param(
                args.vararg, None, mod, SigParameter.VAR_POSITIONAL
            )
            annotation = GenericValue(tuple, [vararg_param.annotation])
            arguments.append(vararg_param.replace(annotation=annotation))
        arguments += self._parse_param_list(
            args.kwonlyargs, args.kw_defaults, mod, SigParameter.KEYWORD_ONLY
        )
        if args.kwarg is not None:
            kwarg_param = self._parse_param(
                args.kwarg, None, mod, SigParameter.VAR_KEYWORD
            )
            annotation = GenericValue(dict, [TypedValue(str), kwarg_param.annotation])
            arguments.append(kwarg_param.replace(annotation=annotation))
        # some typeshed types have a positional-only after a normal argument,
        # and Signature doesn't like that
        seen_non_positional = False
        cleaned_arguments = []
        for arg in arguments:
            if arg.kind is SigParameter.POSITIONAL_ONLY and seen_non_positional:
                cleaned_arguments = [
                    arg.replace(kind=SigParameter.POSITIONAL_ONLY)
                    for arg in cleaned_arguments
                ]
                seen_non_positional = False
            else:
                seen_non_positional = True
            cleaned_arguments.append(arg)
        return Signature.make(
            cleaned_arguments,
            callable=obj,
            return_annotation=GenericValue(Awaitable, [return_value])
            if isinstance(node, ast3.AsyncFunctionDef)
            else return_value,
            allow_call=allow_call,
        )

    def _parse_param_list(
        self,
        args: Iterable[ast3.arg],
        defaults: Iterable[Optional[ast3.AST]],
        module: str,
        kind: inspect._ParameterKind,
        objclass: Optional[type] = None,
    ) -> Iterable[SigParameter]:
        for i, (arg, default) in enumerate(zip(args, defaults)):
            yield self._parse_param(
                arg, default, module, kind, objclass if i == 0 else None
            )

    def _parse_param(
        self,
        arg: ast3.arg,
        default: Optional[ast3.arg],
        module: str,
        kind: inspect._ParameterKind,
        objclass: Optional[type] = None,
    ) -> SigParameter:
        typ = AnyValue(AnySource.unannotated)
        if arg.annotation is not None:
            typ = self._parse_type(arg.annotation, module)
        elif objclass is not None:
            bases = self.get_bases(objclass)
            if bases is None:
                typ = TypedValue(objclass)
            else:
                typevars = uniq_chain(extract_typevars(base) for base in bases)
                if typevars:
                    typ = GenericValue(objclass, [TypeVarValue(tv) for tv in typevars])
                else:
                    typ = TypedValue(objclass)

        name = arg.arg
        # Arguments that start with __ are positional-only in typeshed
        if kind is SigParameter.POSITIONAL_OR_KEYWORD and name.startswith("__"):
            kind = SigParameter.POSITIONAL_ONLY
            name = name[2:]
        # Mark self as positional-only. objclass should be given only if we believe
        # it's the "self" parameter.
        if objclass is not None:
            kind = SigParameter.POSITIONAL_ONLY
        if default is None:
            return SigParameter(name, kind, annotation=typ)
        else:
            default_value = self._parse_expr(default, module)
            if default_value == KnownValue(...):
                default_value = AnyValue(AnySource.unannotated)
            return SigParameter(name, kind, annotation=typ, default=default_value)

    def _parse_expr(self, node: ast3.AST, module: str) -> Value:
        ctx = _AnnotationContext(finder=self, module=module)
        return value_from_ast(cast(ast.AST, node), ctx=ctx)

    def _parse_type(self, node: ast3.AST, module: str) -> Value:
        val = self._parse_expr(node, module)
        ctx = _AnnotationContext(finder=self, module=module)
        typ = type_from_value(val, ctx=ctx)
        if self.verbose and isinstance(typ, AnyValue):
            self.log("Got Any", (ast3.dump(node), module))
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

        if not isinstance(info.ast, ast3.Assign) or not isinstance(
            info.ast.value, ast3.Call
        ):
            return AnyValue(AnySource.inference)
        ctx = _AnnotationContext(finder=self, module=module)
        return value_from_ast(cast(ast.AST, info.ast.value), ctx=ctx)

    def _value_from_info(
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
            elif IS_PRE_38:
                if fq_name in ("typing.Protocol", "typing_extensions.Protocol"):
                    return KnownValue(Protocol)
            if isinstance(info.ast, ast3.Assign):
                key = (module, info.ast)
                if key in self._assignment_cache:
                    return self._assignment_cache[key]
                if isinstance(info.ast.value, ast3.Call):
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
                if isinstance(info.ast, ast3.ClassDef):
                    return TypedValue(f"{module}.{info.name}")
                elif isinstance(info.ast, ast3.AnnAssign):
                    return self._parse_type(info.ast.annotation, module)
                self.log("Unable to import", (module, info))
                return AnyValue(AnySource.inference)
        elif isinstance(info, tuple):
            module_path = ".".join(info)
            try:
                __import__(module_path)
                return KnownValue(sys.modules[module_path])
            except Exception:
                self.log("Unable to import", module_path)
                return AnyValue(AnySource.inference)
        else:
            self.log("Ignoring info", info)
            return AnyValue(AnySource.inference)


def _obj_from_qualname_is(module_name: str, qualname: str, obj: object) -> bool:
    try:
        __import__(module_name)
        mod = sys.modules[module_name]
        actual = mod
        for piece in qualname.split("."):
            actual = getattr(actual, piece)
        return obj is actual
    except Exception:
        return False
