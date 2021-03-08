"""

Implementation of extended argument specifications used by test_scope.

"""

from .annotations import (
    type_from_runtime,
    Context,
    type_from_ast,
    is_typing_name,
    type_from_maybe_generic,
)
from .config import Config
from .error_code import ErrorCode
from .find_unused import used
from . import implementation
from .safe import safe_hasattr
from .stacked_scopes import uniq_chain
from .signature import (
    ExtendedArgSpec,
    Parameter,
    Logger,
    ImplementationFn,
    BoundMethodArgSpecWrapper,
    MaybeArgspec,
    PropertyArgSpec,
)
from .value import (
    TypedValue,
    GenericValue,
    KnownValue,
    UNRESOLVED_VALUE,
    Value,
    VariableNameValue,
    TypeVarValue,
    extract_typevars,
    substitute_typevars,
)

import asyncio
import ast
import asynq
import builtins
from collections.abc import Awaitable, Collection, Set as AbstractSet, Sized
from contextlib import AbstractContextManager
from dataclasses import dataclass
import collections.abc
import contextlib
import qcore
import inspect
import sys
from types import GeneratorType
from typing import (
    cast,
    Any,
    Sequence,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Union,
    Callable,
    Dict,
    List,
    TypeVar,
)
from typing_extensions import Protocol
import typing_inspect
import typeshed_client
from typed_ast import ast3


T_co = TypeVar("T_co", covariant=True)


IS_PRE_38 = sys.version_info < (3, 8)


@used  # exposed as an API
@contextlib.contextmanager
def with_implementation(
    fn: object, implementation_fn: ImplementationFn
) -> Iterable[None]:
    """Temporarily sets the implementation of fn to be implementation_fn.

    This is useful for invoking test_scope to aggregate all calls to a particular function. For
    example, the following can be used to find the names of all scribe categories we log to:

        categories = set()
        def _scribe_log_impl(variables, visitor, node):
            if isinstance(variables['category'], pyanalyze.value.KnownValue):
                categories.add(variables['category'].val)

        with pyanalyze.arg_spec.with_implementation(qclient.scribe.log, _scribe_log_impl):
            test_scope.test_all()

        print(categories)

    """
    if fn in ArgSpecCache.DEFAULT_ARGSPECS:
        with qcore.override(
            ArgSpecCache.DEFAULT_ARGSPECS[fn], "implementation", implementation_fn
        ):
            yield
    else:
        argspec = ArgSpecCache(Config()).get_argspec(
            fn, implementation=implementation_fn
        )
        if argspec is None:
            # builtin or something, just use a generic argspec
            argspec = ExtendedArgSpec(
                [], starargs="args", kwargs="kwargs", implementation=implementation_fn
            )
        known_argspecs = dict(ArgSpecCache.DEFAULT_ARGSPECS)
        known_argspecs[fn] = argspec
        with qcore.override(ArgSpecCache, "DEFAULT_ARGSPECS", known_argspecs):
            yield


def is_dot_asynq_function(obj: Any) -> bool:
    """Returns whether obj is the .asynq member on an async function."""
    try:
        self_obj = obj.__self__
    except AttributeError:
        # the attribute doesn't exist
        return False
    except Exception:
        # The object has a buggy __getattr__ that threw an error. Just ignore it.
        return False
    if qcore.inspection.is_classmethod(obj):
        return False
    if obj is self_obj:
        return False
    try:
        is_async_fn = asynq.is_async_fn(self_obj)
    except Exception:
        # The object may have a buggy __getattr__. Ignore it. This happens with
        # pylons request objects.
        return False
    if not is_async_fn:
        return False

    return getattr(obj, "__name__", None) in ("async", "asynq")


class ArgSpecCache:
    DEFAULT_ARGSPECS = implementation.get_default_argspecs()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ts_finder = TypeshedFinder(verbose=False)
        self.known_argspecs = {}
        self.generic_bases_cache = {}
        default_argspecs = dict(self.DEFAULT_ARGSPECS)
        default_argspecs.update(self.config.get_known_argspecs(self))

        for obj, argspec in default_argspecs.items():
            self.known_argspecs[obj] = argspec

    def __reduce_ex__(self, proto: object) -> object:
        # Don't pickle the actual argspecs, which are frequently unpicklable.
        return self.__class__, (self.config,)

    def from_argspec(
        self,
        argspec: inspect.Signature,
        *,
        kwonly_args: Any = None,
        name: Optional[str] = None,
        logger: Optional[Logger] = None,
        implementation: Optional[ImplementationFn] = None,
        function_object: Optional[object] = None,
    ) -> ExtendedArgSpec:
        """Constructs an ExtendedArgSpec from a standard argspec.

        argspec is an inspect.Signature.

        kwonly_args may be a list of custom keyword-only arguments added to the argspec or None.

        name is the name of the function. This is used for better error messages.

        logger is the log function to be used.

        implementation is an implementation function for this object.

        TODO: do we need support for non-Signature argspecs and for the separate kwonly_args
        argument?

        """
        if kwonly_args is None:
            kwonly_args = []
        else:
            kwonly_args = list(kwonly_args)
        func_globals = getattr(function_object, "__globals__", None)

        # inspect.Signature object
        starargs = None
        kwargs = None
        args = []
        if argspec.return_annotation is argspec.empty:
            return_value = UNRESOLVED_VALUE
        else:
            return_value = type_from_runtime(
                argspec.return_annotation, globals=func_globals
            )
        for i, parameter in enumerate(argspec.parameters.values()):
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                starargs = parameter.name
            elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                kwargs = parameter.name
            elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                kwonly_args.append(
                    self._parameter_from_signature(
                        parameter, func_globals, function_object, i
                    )
                )
            else:
                # positional or positional-or-keyword
                args.append(
                    self._parameter_from_signature(
                        parameter, func_globals, function_object, i
                    )
                )
        return ExtendedArgSpec(
            args,
            starargs=starargs,
            kwargs=kwargs,
            kwonly_args=kwonly_args,
            return_value=return_value,
            name=name,
            logger=logger,
            implementation=implementation,
        )

    def _parameter_from_signature(
        self,
        parameter: inspect.Parameter,
        func_globals: Mapping[str, object],
        function_object: Optional[object],
        index: int,
    ) -> Parameter:
        """Given an inspect.Parameter, returns a Parameter object."""
        typ = self._get_type_for_parameter(
            parameter, func_globals, function_object, index
        )
        if parameter.default is inspect.Parameter.empty:
            default_value = Parameter.no_default_value
        elif parameter.default is Parameter.no_default_value:
            # hack to prevent errors in code that use Parameter
            default_value = object()
        else:
            default_value = parameter.default
        return Parameter(parameter.name, default_value=default_value, typ=typ)

    def _get_type_for_parameter(
        self,
        parameter: inspect.Parameter,
        func_globals: Mapping[str, object],
        function_object: Optional[object],
        index: int,
    ) -> Optional[Value]:
        if parameter.annotation is not inspect.Parameter.empty:
            return type_from_runtime(parameter.annotation, globals=func_globals)
        # If this is the self argument of a method, try to infer the self type.
        elif index == 0 and parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            module_name = getattr(function_object, "__module__", None)
            qualname = getattr(function_object, "__qualname__", None)
            name = getattr(function_object, "__name__", None)
            if (
                qualname != name
                and module_name is not None
                and module_name in sys.modules
            ):
                module = sys.modules[module_name]
                *class_names, function_name = qualname.split(".")
                class_obj = module
                for class_name in class_names:
                    class_obj = getattr(class_obj, class_name, None)
                    if class_obj is None:
                        break
                if (
                    class_obj is not None
                    and inspect.getattr_static(class_obj, function_name, None)
                    is function_object
                ):
                    return type_from_maybe_generic(class_obj)
        return VariableNameValue.from_varname(
            parameter.name, self.config.varname_value_map()
        )

    def get_argspec(
        self,
        obj: object,
        name: Optional[str] = None,
        logger: Optional[Logger] = None,
        implementation: Optional[ImplementationFn] = None,
    ) -> MaybeArgspec:
        """Constructs the ExtendedArgSpec for a Python object."""
        kwargs = {"name": name, "logger": logger, "implementation": implementation}
        argspec = self._cached_get_argspec(obj, kwargs)
        return argspec

    def _cached_get_argspec(
        self, obj: object, kwargs: Mapping[str, Any]
    ) -> MaybeArgspec:
        try:
            if obj in self.known_argspecs:
                return self.known_argspecs[obj]
        except Exception:
            hashable = False  # unhashable, or __eq__ failed
        else:
            hashable = True

        extended = self._uncached_get_argspec(obj, kwargs)
        if extended is None:
            return None

        if hashable:
            self.known_argspecs[obj] = extended
        return extended

    def _uncached_get_argspec(
        self, obj: Any, kwargs: Mapping[str, Any]
    ) -> MaybeArgspec:
        if isinstance(obj, tuple) or hasattr(obj, "__getattr__"):
            return None  # lost cause

        # Cythonized methods, e.g. fn.asynq
        if is_dot_asynq_function(obj):
            try:
                return self._cached_get_argspec(obj.__self__, kwargs)
            except TypeError:
                # some cythonized methods have __self__ but it is not a function
                pass

        # for bound methods, see if we have an argspec for the unbound method
        if inspect.ismethod(obj) and obj.__self__ is not None:
            argspec = self._cached_get_argspec(obj.__func__, kwargs)
            if argspec is None:
                return None
            return BoundMethodArgSpecWrapper(argspec, KnownValue(obj.__self__))

        if hasattr(obj, "fn") or hasattr(obj, "original_fn"):
            # many decorators put the original function in the .fn attribute
            try:
                original_fn = qcore.get_original_fn(obj)
            except (TypeError, AttributeError):
                # fails when executed on an object that doesn't allow setting attributes,
                # e.g. certain extension classes
                pass
            else:
                return self._cached_get_argspec(original_fn, kwargs)

        argspec = self.ts_finder.get_argspec(obj)
        if argspec is not None:
            if (
                asyncio.iscoroutinefunction(obj)
                and argspec.return_value is UNRESOLVED_VALUE
            ):
                argspec.return_value = GenericValue(Awaitable, [UNRESOLVED_VALUE])
            return argspec

        if inspect.isfunction(obj):
            if hasattr(obj, "inner"):
                # @qclient.task_queue.exec_after_request() puts the original function in .inner
                return self._cached_get_argspec(obj.inner, kwargs)

            argspec = self.from_argspec(
                self._safe_get_argspec(obj), function_object=obj, **kwargs
            )
            if asyncio.iscoroutinefunction(obj):
                argspec.return_value = GenericValue(Awaitable, [argspec.return_value])
            return argspec

        # decorator binders
        if _is_qcore_decorator(obj):
            argspec = self._cached_get_argspec(obj.decorator, kwargs)
            # wrap if it's a bound method
            if obj.instance is not None and argspec is not None:
                return BoundMethodArgSpecWrapper(argspec, KnownValue(obj.instance))
            return argspec

        if inspect.isclass(obj):
            obj = self.config.unwrap_cls(obj)
            if issubclass(obj, self.config.CLASSES_USING_INIT):
                constructor = obj.init
            elif hasattr(obj, "__init__"):
                constructor = obj.__init__
            else:
                # old-style class
                return None
            argspec = self._safe_get_argspec(constructor)
            if argspec is None:
                return None

            kwonly_args = []
            for cls_, args in self.config.CLASS_TO_KEYWORD_ONLY_ARGUMENTS.items():
                if issubclass(obj, cls_):
                    kwonly_args += [
                        Parameter(param_name, default_value=None) for param_name in args
                    ]
            argspec = self.from_argspec(
                argspec, function_object=constructor, kwonly_args=kwonly_args, **kwargs
            )
            if argspec is None:
                return None
            return BoundMethodArgSpecWrapper(argspec, TypedValue(obj))

        if inspect.isbuiltin(obj):
            if hasattr(obj, "__self__"):
                cls = type(obj.__self__)
                try:
                    method = getattr(cls, obj.__name__)
                except AttributeError:
                    return None
                if method == obj:
                    return None
                argspec = self._cached_get_argspec(method, kwargs)
                if argspec is None:
                    return None
                return BoundMethodArgSpecWrapper(argspec, KnownValue(obj.__self__))
            return None

        if hasattr(obj, "__call__"):
            # we could get an argspec here in some cases, but it's impossible to figure out
            # the argspec for some builtin methods (e.g., dict.__init__), and no way to detect
            # these with inspect, so just give up.
            return None

        if isinstance(obj, property):
            # If we know the getter, inherit its return value.
            if obj.fget:
                fget_argspec = self._cached_get_argspec(obj.fget, kwargs)
                if fget_argspec is not None and fget_argspec.has_return_value():
                    return PropertyArgSpec(obj, return_value=fget_argspec.return_value)
            return PropertyArgSpec(obj)

        raise TypeError("%r object is not callable" % (obj,))

    def _safe_get_argspec(self, obj: Any) -> Optional[inspect.Signature]:
        """Wrapper around inspect.getargspec that catches TypeErrors."""
        try:
            # follow_wrapped=True leads to problems with decorators that
            # mess with the arguments, such as mock.patch.
            sig = inspect.signature(obj, follow_wrapped=False)
        except (TypeError, ValueError, AttributeError):
            # TypeError if signature() does not support the object, ValueError
            # if it cannot provide a signature, and AttributeError if we're on
            # Python 2.
            return None
        else:
            # Signature preserves the return annotation for wrapped functions,
            # because @functools.wraps copies the __annotations__ of the wrapped function. We
            # don't want that, because the wrapper may have changed the return type.
            # This caused problems with @contextlib.contextmanager.
            if safe_hasattr(obj, "__wrapped__"):
                return sig.replace(return_annotation=inspect.Signature.empty)
            else:
                return sig

    def get_generic_bases(
        self, typ: type, generic_args: Sequence[Value] = ()
    ) -> Dict[type, Sequence[Value]]:
        if typ is Generic or is_typing_name(typ, "Protocol"):
            return {}
        generic_bases = self._get_generic_bases_cached(typ)
        if typ not in generic_bases:
            return generic_bases
        my_typevars = generic_bases[typ]
        if not my_typevars:
            return generic_bases
        tv_map = {}
        for i, tv_value in enumerate(my_typevars):
            if not isinstance(tv_value, TypeVarValue):
                continue
            try:
                value = generic_args[i]
            except IndexError:
                value = UNRESOLVED_VALUE
            tv_map[tv_value.typevar] = value
        return {
            base: substitute_typevars(args, tv_map)
            for base, args in generic_bases.items()
        }

    def _get_generic_bases_cached(self, typ: type) -> Dict[type, Sequence[Value]]:
        try:
            return self.generic_bases_cache[typ]
        except KeyError:
            pass
        except Exception:
            return {}  # We don't support unhashable types.
        bases = self.ts_finder.get_bases(typ)
        generic_bases = self._extract_bases(typ, bases)
        if generic_bases is None:
            bases = [type_from_runtime(base) for base in self.get_runtime_bases(typ)]
            generic_bases = self._extract_bases(typ, bases)
            assert (
                generic_bases is not None
            ), f"failed to extract runtime bases from {typ}"
        return generic_bases

    def _extract_bases(
        self, typ: type, bases: Optional[Sequence[Value]]
    ) -> Optional[Dict[type, Sequence[Value]]]:
        if bases is None:
            return None
        my_typevars = uniq_chain(extract_typevars(base) for base in bases)
        generic_bases = {}
        generic_bases[typ] = [TypeVarValue(tv) for tv in my_typevars]
        for base in bases:
            if isinstance(base, TypedValue):
                assert base.typ is not typ, base
                if isinstance(base, GenericValue):
                    args = base.args
                else:
                    args = ()
                generic_bases.update(self.get_generic_bases(base.typ, args))
            else:
                return None
        return generic_bases

    def get_runtime_bases(self, typ: type) -> Sequence[Value]:
        if typing_inspect.is_generic_type(typ):
            return typing_inspect.get_generic_bases(typ)
        return typ.__bases__


@dataclass
class _AnnotationContext(Context):
    finder: "TypeshedFinder"
    module: str

    def show_error(
        self, message: str, error_code: ErrorCode = ErrorCode.invalid_annotation
    ) -> None:
        self.finder.log(message, ())

    def get_name(self, node: ast.AST) -> Value:
        info = self.finder._get_info_for_name("%s.%s" % (self.module, node.id))
        if info is not None:
            return self.finder._value_from_info(info, self.module)
        elif hasattr(builtins, node.id):
            val = getattr(builtins, node.id)
            if val is None or isinstance(val, type):
                return KnownValue(val)
        return UNRESOLVED_VALUE


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


class TypeshedFinder(object):
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.resolver = typeshed_client.Resolver(version=sys.version_info[:2])
        self._assignment_cache = {}

    def log(self, message: str, obj: object) -> None:
        if not self.verbose:
            return
        print("%s: %r" % (message, obj))

    def get_argspec(self, obj: object) -> Optional[ExtendedArgSpec]:
        if inspect.ismethoddescriptor(obj) and hasattr(obj, "__objclass__"):
            objclass = obj.__objclass__
            fq_name = self._get_fq_name(objclass)
            if fq_name is None:
                return None
            info = self._get_info_for_name(fq_name)
            argspec = self._get_method_argspec_from_info(
                info, obj, fq_name, objclass.__module__
            )
            if argspec is not None:
                self.log("Found argspec", (obj, argspec))
            return argspec

        if inspect.ismethod(obj):
            self.log("Ignoring method", obj)
            return None
        fq_name = self._get_fq_name(obj)
        if fq_name is None:
            return None
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        argspec = self._get_argspec_from_info(info, obj, fq_name, mod)
        if argspec is not None:
            self.log("Found argspec", (fq_name, argspec))
        return argspec

    def get_bases(self, typ: type) -> Optional[List[Value]]:
        # The way AbstractSet/Set is handled between collections and typing is
        # too confusing, just hardcode it.
        if typ is AbstractSet:
            return [GenericValue(Collection, (TypeVarValue(T_co),))]
        if typ is AbstractContextManager:
            return [GenericValue(Generic, (TypeVarValue(T_co),))]
        if typ is Callable or typ is collections.abc.Callable:
            return None
        fq_name = self._get_fq_name(typ)
        if fq_name is None:
            return None
        info = self._get_info_for_name(fq_name)
        mod, _ = fq_name.rsplit(".", maxsplit=1)
        return self._get_bases_from_info(info, mod)

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
                return [self._parse_expr(base, mod) for base in bases]
            elif isinstance(info.ast, ast3.Assign):
                val = self._parse_expr(info.ast.value, mod)
                if isinstance(val, KnownValue) and isinstance(val.val, type):
                    return self.get_bases(val.val)
                else:
                    return [val]
            elif isinstance(
                info.ast, (typeshed_client.OverloadedName, typeshed_client.ImportedName)
            ):
                return None  # overloads are not supported yet
            else:
                raise NotImplementedError(ast3.dump(info.ast))
        return None

    def _get_method_argspec_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
    ) -> Optional[ExtendedArgSpec]:
        if info is None:
            return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_method_argspec_from_info(
                info.info, obj, fq_name, ".".join(info.source_module)
            )
        elif isinstance(info, typeshed_client.NameInfo):
            # Note that this doesn't handle names inherited from base classes
            if obj.__name__ in info.child_nodes:
                child_info = info.child_nodes[obj.__name__]
                return self._get_argspec_from_info(child_info, obj, fq_name, mod)
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
            module = obj.__module__
            if module is None:
                module = "builtins"
            fq_name = ".".join([module, obj.__qualname__])
            return _TYPING_ALIASES.get(fq_name, fq_name)
        except (AttributeError, TypeError):
            self.log("Ignoring object without module or qualname", obj)
            return None

    def _get_argspec_from_info(
        self,
        info: typeshed_client.resolver.ResolvedName,
        obj: object,
        fq_name: str,
        mod: str,
    ) -> Optional[ExtendedArgSpec]:
        if isinstance(info, typeshed_client.NameInfo):
            if isinstance(info.ast, ast3.FunctionDef):
                return self._get_argspec_from_func_def(
                    info.ast, obj, mod, is_async_fn=False
                )
            elif isinstance(info.ast, ast3.AsyncFunctionDef):
                return self._get_argspec_from_func_def(
                    info.ast, obj, mod, is_async_fn=True
                )
            else:
                self.log("Ignoring unrecognized AST", (fq_name, info))
                return None
        elif isinstance(info, typeshed_client.ImportedInfo):
            return self._get_argspec_from_info(
                info.info, obj, fq_name, ".".join(info.source_module)
            )
        elif info is None:
            return None
        else:
            self.log("Ignoring unrecognized info", (fq_name, info))
            return None

    @qcore.caching.cached_per_instance()
    def _get_info_for_name(self, fq_name: str) -> typeshed_client.resolver.ResolvedName:
        return self.resolver.get_fully_qualified_name(fq_name)

    def _get_argspec_from_func_def(
        self,
        node: Union[ast3.FunctionDef, ast3.AsyncFunctionDef],
        obj: object,
        mod: str,
        is_async_fn: bool,
    ) -> Optional[ExtendedArgSpec]:
        if node.decorator_list:
            # might be @overload or something else we don't recognize
            return None
        if node.returns is None:
            return_value = UNRESOLVED_VALUE
        else:
            return_value = self._parse_expr(node.returns, mod)
        args = node.args
        if args.vararg is None:
            starargs = None
        else:
            starargs = args.vararg.arg
        if args.kwarg is None:
            kwargs = None
        else:
            kwargs = args.kwarg.arg
        num_without_defaults = len(args.args) - len(args.defaults)
        defaults = [None] * num_without_defaults + args.defaults
        arguments = list(self._parse_param_list(args.args, defaults, mod))
        kwonly = list(self._parse_param_list(args.kwonlyargs, args.kw_defaults, mod))
        return ExtendedArgSpec(
            arguments,
            starargs=starargs,
            kwargs=kwargs,
            kwonly_args=kwonly,
            return_value=GenericValue(Awaitable, [return_value])
            if is_async_fn
            else return_value,
            name=obj.__name__,
        )

    def _parse_param_list(
        self,
        args: Iterable[ast3.arg],
        defaults: Iterable[Optional[ast3.AST]],
        module: str,
    ) -> Iterable[Parameter]:
        for arg, default in zip(args, defaults):
            typ = None
            if arg.annotation is not None:
                typ = self._parse_expr(arg.annotation, module)

            if default is None:
                yield Parameter(arg.arg, typ=typ)
            else:
                # doesn't matter what the default is
                yield Parameter(arg.arg, typ=typ, default_value=None)

    def _parse_expr(self, node: ast3.AST, module: str) -> Value:
        ctx = _AnnotationContext(finder=self, module=module)
        typ = type_from_ast(cast(ast.AST, node), ctx=ctx)
        if self.verbose and typ is UNRESOLVED_VALUE:
            self.log("Got UNRESOLVED_VALUE", (ast3.dump(node), module))
        return typ

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
                value = self._parse_expr(info.ast.value, module)
                self._assignment_cache[key] = value
                return value
            try:
                __import__(module)
                mod = sys.modules[module]
                return KnownValue(getattr(mod, info.name))
            except Exception:
                self.log("Unable to import", (module, info))
                return UNRESOLVED_VALUE
        elif isinstance(info, tuple):
            module_path = ".".join(info)
            try:
                __import__(module_path)
                return KnownValue(sys.modules[module_path])
            except Exception:
                self.log("Unable to import", module_path)
                return UNRESOLVED_VALUE
        else:
            self.log("Ignoring info", info)
            return UNRESOLVED_VALUE


def _is_qcore_decorator(obj: object) -> bool:
    try:
        return (
            hasattr(obj, "is_decorator")
            and obj.is_decorator()
            and hasattr(obj, "decorator")
        )
    except Exception:
        # black.Line has an is_decorator attribute but it is not a method
        return False
