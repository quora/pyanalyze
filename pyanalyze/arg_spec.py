"""

Implementation of extended argument specifications used by test_scope.

"""

from .options import Options, PyObjectSequenceOption
from .analysis_lib import is_positional_only_arg_name
from .extensions import CustomCheck, get_overloads, get_type_evaluations
from .annotations import Context, RuntimeEvaluator, type_from_runtime
from .config import Config
from .find_unused import used
from . import implementation
from .safe import (
    all_of_type,
    is_newtype,
    safe_hasattr,
    safe_issubclass,
    is_typing_name,
    safe_isinstance,
    get_fully_qualified_name,
)
from .stacked_scopes import Composite, uniq_chain
from .signature import (
    ANY_SIGNATURE,
    ELLIPSIS_PARAM,
    ConcreteSignature,
    Impl,
    MaybeSignature,
    OverloadedSignature,
    PropertyArgSpec,
    make_bound_method,
    SigParameter,
    Signature,
    ParameterKind,
)
from .typeshed import TypeshedFinder
from .value import (
    AnySource,
    AnyValue,
    Extension,
    GenericBases,
    KVPair,
    TypedValue,
    GenericValue,
    NewTypeValue,
    KnownValue,
    Value,
    TypeVarValue,
    extract_typevars,
    make_weak,
)

import ast
import asyncio
import asynq
from collections.abc import Awaitable
import contextlib
from dataclasses import dataclass, replace
import qcore
import inspect
import sys
import textwrap
from types import FunctionType, ModuleType
from typing import (
    Any,
    Callable,
    Iterator,
    Sequence,
    Generic,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import typing_inspect
from unittest import mock

# types.MethodWrapperType in 3.7+
MethodWrapperType = type(object().__str__)


@used  # exposed as an API
@contextlib.contextmanager
def with_implementation(fn: object, implementation_fn: Impl) -> Iterator[None]:
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
            ArgSpecCache.DEFAULT_ARGSPECS[fn], "impl", implementation_fn
        ):
            yield
    else:
        options = Options.from_option_list([], Config())
        tsf = TypeshedFinder.make(options)
        argspec = ArgSpecCache(options, tsf).get_argspec(fn, impl=implementation_fn)
        if argspec is None:
            # builtin or something, just use a generic argspec
            argspec = Signature.make(
                [
                    SigParameter("args", ParameterKind.VAR_POSITIONAL),
                    SigParameter("kwargs", ParameterKind.VAR_KEYWORD),
                ],
                callable=fn,
                impl=implementation_fn,
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


@dataclass
class AnnotationsContext(Context):
    arg_spec_cache: "ArgSpecCache"
    globals: Optional[Mapping[str, object]] = None

    def __post_init__(self) -> None:
        super().__init__()

    def get_name(self, node: ast.Name) -> Value:
        if self.globals is not None:
            return self.get_name_from_globals(node.id, self.globals)
        return self.handle_undefined_name(node.id)


class IgnoredCallees(PyObjectSequenceOption[object]):
    """Calls to these aren't checked for argument validity."""

    default_value = [
        # getargspec gets confused about this subclass of tuple that overrides __new__ and __call__
        mock.call,
        mock.MagicMock,
        mock.Mock,
    ]
    name = "ignored_callees"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[object]:
        return fallback.IGNORED_CALLEES


class ClassesSafeToInstantiate(PyObjectSequenceOption[type]):
    """We will instantiate instances of these classes if we can infer the value of all of
    their arguments. This is useful mostly for classes that are commonly instantiated with static
    arguments."""

    name = "classes_safe_to_instantiate"
    default_value = [
        CustomCheck,
        Value,
        Extension,
        KVPair,
        asynq.ConstFuture,
        range,
        tuple,
    ]

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[type]:
        return fallback.CLASSES_SAFE_TO_INSTANTIATE


class FunctionsSafeToCall(PyObjectSequenceOption[object]):
    """We will instantiate instances of these classes if we can infer the value of all of
    their arguments. This is useful mostly for classes that are commonly instantiated with static
    arguments."""

    name = "functions_safe_to_call"
    default_value = [sorted, asynq.asynq, make_weak]

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[object]:
        return fallback.FUNCTIONS_SAFE_TO_CALL


_HookReturn = Union[None, ConcreteSignature, inspect.Signature, Callable[..., Any]]
_ConstructorHook = Callable[[type], _HookReturn]


class ConstructorHooks(PyObjectSequenceOption[_ConstructorHook]):
    """Customize the constructor signature for a class.

    These hooks may return either a function that pyanalyze will use the signature of, an inspect
    Signature object, or a pyanalyze Signature object. The function or signature
    should take a self parameter.

    """

    name = "constructor_hooks"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[_ConstructorHook]:
        return [fallback.get_constructor]

    @classmethod
    def get_constructor(cls, typ: type, options: Options) -> _HookReturn:
        for hook in options.get_value_for(cls):
            result = hook(typ)
            if result is not None:
                return result
        return None


_SigProvider = Callable[["ArgSpecCache"], Mapping[object, ConcreteSignature]]


class KnownSignatures(PyObjectSequenceOption[_SigProvider]):
    """Provide hardcoded signatures (and potentially :term:`impl` functions) for
    particular objects.

    Each entry in the list must be a function that takes an :class:`ArgSpecCache`
    instance and returns a mapping from Python object to
    :class:`pyanalyze.signature.Signature`.

    """

    name = "known_signatures"
    default_value = []

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[_SigProvider]:
        return [fallback.get_known_argspecs]


_Unwrapper = Callable[[type], type]


class UnwrapClass(PyObjectSequenceOption[_Unwrapper]):
    """Provides functions that can unwrap decorated classes.

    For example, if your codebase commonly uses a decorator that
    wraps classes in a `Wrapper` subclass with a `.wrapped` attribute,
    you may define an unwrapper like this:

        def unwrap_class(typ: type) -> type:
            if issubclass(typ, Wrapper) and typ is not Wrapper:
                return typ.wrapped
            return typ

    """

    name = "unwrap_class"

    @classmethod
    def get_value_from_fallback(cls, fallback: Config) -> Sequence[_Unwrapper]:
        return [fallback.unwrap_cls]

    @classmethod
    def unwrap(cls, typ: type, options: Options) -> type:
        for unwrapper in options.get_value_for(cls):
            typ = unwrapper(typ)
        return typ


class ArgSpecCache:
    DEFAULT_ARGSPECS = implementation.get_default_argspecs()

    def __init__(
        self,
        options: Options,
        ts_finder: TypeshedFinder,
        *,
        vnv_provider: Callable[[str], Optional[Value]] = lambda _: None,
    ) -> None:
        self.vnv_provider = vnv_provider
        self.options = options
        self.config = options.fallback
        self.ts_finder = ts_finder
        self.known_argspecs = {}
        self.generic_bases_cache = {}
        self.default_context = AnnotationsContext(self)
        default_argspecs = dict(self.DEFAULT_ARGSPECS)
        for provider in options.get_value_for(KnownSignatures):
            default_argspecs.update(provider(self))

        for obj, argspec in default_argspecs.items():
            self.known_argspecs[obj] = argspec

    def __reduce_ex__(self, proto: object) -> object:
        # Don't pickle the actual argspecs, which are frequently unpicklable.
        return self.__class__, (self.config,)

    def from_signature(
        self,
        sig: inspect.Signature,
        *,
        impl: Optional[Impl] = None,
        function_object: object,
        is_async: bool = False,
        is_asynq: bool = False,
        returns: Optional[Value] = None,
        allow_call: bool = False,
    ) -> Signature:
        """Constructs a pyanalyze Signature from an inspect.Signature.

        kwonly_args may be a list of custom keyword-only arguments added to the argspec or None.

        impl is an implementation function for this object.

        function_object is the underlying callable.

        """
        func_globals = getattr(function_object, "__globals__", None)
        # Signature preserves the return annotation for wrapped functions,
        # because @functools.wraps copies the __annotations__ of the wrapped function. We
        # don't want that, because the wrapper may have changed the return type.
        # This caused problems with @contextlib.contextmanager.
        is_wrapped = safe_hasattr(function_object, "__wrapped__")

        if returns is not None:
            has_return_annotation = True
        else:
            if is_wrapped or sig.return_annotation is inspect.Signature.empty:
                returns = AnyValue(AnySource.unannotated)
                has_return_annotation = False
            else:
                returns = type_from_runtime(
                    sig.return_annotation, ctx=AnnotationsContext(self, func_globals)
                )
                has_return_annotation = True
            if is_async:
                returns = GenericValue(Awaitable, [returns])

        parameters = []
        for i, parameter in enumerate(sig.parameters.values()):
            param, make_everything_pos_only = self._make_sig_parameter(
                parameter, func_globals, function_object, is_wrapped, i
            )
            if make_everything_pos_only:
                parameters = [
                    replace(param, kind=ParameterKind.POSITIONAL_ONLY)
                    for param in parameters
                ]
            parameters.append(param)

        return Signature.make(
            parameters,
            returns,
            impl=impl,
            callable=function_object,
            has_return_annotation=has_return_annotation,
            is_asynq=is_asynq,
            allow_call=allow_call
            or FunctionsSafeToCall.contains(function_object, self.options),
        )

    def _make_sig_parameter(
        self,
        parameter: inspect.Parameter,
        func_globals: Optional[Mapping[str, object]],
        function_object: Optional[object],
        is_wrapped: bool,
        index: int,
    ) -> Tuple[SigParameter, bool]:
        """Given an inspect.Parameter, returns a Parameter object."""
        if is_wrapped:
            typ = AnyValue(AnySource.inference)
        else:
            typ = self._get_type_for_parameter(
                parameter, func_globals, function_object, index
            )
        if parameter.default is inspect.Parameter.empty:
            default = None
        else:
            default = KnownValue(parameter.default)
        if (
            parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and is_positional_only_arg_name(
                parameter.name, _get_class_name(function_object)
            )
        ):
            kind = ParameterKind.POSITIONAL_ONLY
            make_everything_pos_only = True
        else:
            kind = ParameterKind(parameter.kind)
            make_everything_pos_only = False
        return (
            SigParameter(parameter.name, kind, default=default, annotation=typ),
            make_everything_pos_only,
        )

    def _get_type_for_parameter(
        self,
        parameter: inspect.Parameter,
        func_globals: Optional[Mapping[str, object]],
        function_object: Optional[object],
        index: int,
    ) -> Value:
        if parameter.annotation is not inspect.Parameter.empty:
            typ = type_from_runtime(
                parameter.annotation, ctx=AnnotationsContext(self, func_globals)
            )
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                return GenericValue(tuple, [typ])
            elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
                return GenericValue(dict, [TypedValue(str), typ])
            return typ
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
                    isinstance(class_obj, type)
                    and inspect.getattr_static(class_obj, function_name, None)
                    is function_object
                ):
                    generic_bases = self._get_generic_bases_cached(class_obj)
                    if generic_bases and generic_bases.get(class_obj):
                        return GenericValue(
                            class_obj, generic_bases[class_obj].values()
                        )
                    return TypedValue(class_obj)
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            vnv = self.vnv_provider(parameter.name)
            if vnv is not None:
                return vnv
        return AnyValue(AnySource.unannotated)

    def get_argspec(
        self, obj: object, impl: Optional[Impl] = None, is_asynq: bool = False
    ) -> MaybeSignature:
        """Constructs the Signature for a Python object."""
        return self._cached_get_argspec(
            obj, impl, is_asynq, in_overload_resolution=False
        )

    def _cached_get_argspec(
        self,
        obj: object,
        impl: Optional[Impl],
        is_asynq: bool,
        in_overload_resolution: bool,
    ) -> MaybeSignature:
        try:
            if obj in self.known_argspecs:
                return self.known_argspecs[obj]
        except Exception:
            hashable = False  # unhashable, or __eq__ failed
        else:
            hashable = True

        extended = self._uncached_get_argspec(
            obj, impl, is_asynq, in_overload_resolution
        )
        if extended is None:
            return None

        if hashable:
            self.known_argspecs[obj] = extended
        return extended

    def _maybe_make_evaluator_sig(
        self, func: Callable[..., Any], impl: Optional[Impl], is_asynq: bool
    ) -> MaybeSignature:
        try:
            key = f"{func.__module__}.{func.__qualname__}"
        except AttributeError:
            return None
        evaluation_funcs = get_type_evaluations(key)
        if not evaluation_funcs:
            return None
        sigs = []
        for evaluation_func in evaluation_funcs:
            if evaluation_func is None or not hasattr(evaluation_func, "__globals__"):
                return None
            sig = self._cached_get_argspec(
                evaluation_func, impl, is_asynq, in_overload_resolution=True
            )
            if not isinstance(sig, Signature):
                return None
            lines, _ = inspect.getsourcelines(evaluation_func)
            code = textwrap.dedent("".join(lines))
            body = ast.parse(code)
            if not body.body:
                return None
            evaluator_node = body.body[0]
            if not isinstance(evaluator_node, ast.FunctionDef):
                return None
            evaluator = RuntimeEvaluator(
                evaluator_node,
                sig.return_value,
                evaluation_func.__globals__,
                evaluation_func,
            )
            sigs.append(replace(sig, evaluator=evaluator))
        if len(sigs) == 1:
            return sigs[0]
        return OverloadedSignature(sigs)

    def _uncached_get_argspec(
        self,
        obj: Any,
        impl: Optional[Impl],
        is_asynq: bool,
        in_overload_resolution: bool,
    ) -> MaybeSignature:
        if not in_overload_resolution:
            fq_name = get_fully_qualified_name(obj)
            if fq_name is not None:
                overloads = get_overloads(fq_name)
                if overloads:
                    sigs = [
                        self._cached_get_argspec(
                            overload, impl, is_asynq, in_overload_resolution=True
                        )
                        for overload in overloads
                    ]
                    if all_of_type(sigs, Signature):
                        return OverloadedSignature(sigs)
                evaluator_sig = self._maybe_make_evaluator_sig(obj, impl, is_asynq)
                if evaluator_sig is not None:
                    return evaluator_sig

        if isinstance(obj, tuple) or hasattr(obj, "__getattr__"):
            return None  # lost cause

        # Cythonized methods, e.g. fn.asynq
        if is_dot_asynq_function(obj):
            try:
                return self._cached_get_argspec(
                    obj.__self__, impl, is_asynq, in_overload_resolution
                )
            except TypeError:
                # some cythonized methods have __self__ but it is not a function
                pass

        if safe_isinstance(obj, MethodWrapperType):
            try:
                unbound = getattr(obj.__objclass__, obj.__name__)
            except Exception:
                pass
            else:
                sig = self._cached_get_argspec(
                    unbound, impl, is_asynq, in_overload_resolution
                )
                if sig is not None:
                    return make_bound_method(sig, Composite(KnownValue(obj.__self__)))

        # for bound methods, see if we have an argspec for the unbound method
        if inspect.ismethod(obj) and obj.__self__ is not None:
            argspec = self._cached_get_argspec(
                obj.__func__, impl, is_asynq, in_overload_resolution
            )
            return make_bound_method(argspec, Composite(KnownValue(obj.__self__)))

        if hasattr(obj, "fn") or hasattr(obj, "original_fn"):
            is_asynq = is_asynq or hasattr(obj, "asynq")
            # many decorators put the original function in the .fn attribute
            try:
                original_fn = qcore.get_original_fn(obj)
            except (TypeError, AttributeError):
                # fails when executed on an object that doesn't allow setting attributes,
                # e.g. certain extension classes
                pass
            else:
                return self._cached_get_argspec(
                    original_fn, impl, is_asynq, in_overload_resolution
                )

        allow_call = FunctionsSafeToCall.contains(obj, self.options)
        argspec = self.ts_finder.get_argspec(obj, allow_call=allow_call)
        if argspec is not None:
            return argspec

        if is_newtype(obj):
            return Signature.make(
                [
                    SigParameter(
                        "x",
                        ParameterKind.POSITIONAL_ONLY,
                        annotation=type_from_runtime(
                            obj.__supertype__, ctx=self.default_context
                        ),
                    )
                ],
                NewTypeValue(obj),
                callable=obj,
            )

        if inspect.isfunction(obj):
            if hasattr(obj, "inner"):
                # @qclient.task_queue.exec_after_request() puts the original function in .inner
                return self._cached_get_argspec(
                    obj.inner, impl, is_asynq, in_overload_resolution
                )

            inspect_sig = self._safe_get_signature(obj)
            if inspect_sig is None:
                return self._make_any_sig(obj)

            return self.from_signature(
                inspect_sig,
                function_object=obj,
                is_async=asyncio.iscoroutinefunction(obj),
                impl=impl,
                is_asynq=is_asynq,
            )

        # decorator binders
        if _is_qcore_decorator(obj):
            argspec = self._cached_get_argspec(
                obj.decorator, impl, is_asynq, in_overload_resolution
            )
            # wrap if it's a bound method
            if obj.instance is not None and argspec is not None:
                return make_bound_method(argspec, Composite(KnownValue(obj.instance)))
            return argspec

        if inspect.isclass(obj):
            obj = UnwrapClass.unwrap(obj, self.options)
            override = ConstructorHooks.get_constructor(obj, self.options)
            if isinstance(override, Signature):
                signature = override
            else:
                should_ignore = IgnoredCallees.contains(obj, self.options)
                return_type = (
                    AnyValue(AnySource.error) if should_ignore else TypedValue(obj)
                )
                safe = tuple(self.options.get_value_for(ClassesSafeToInstantiate))
                allow_call = safe_issubclass(obj, safe)
                if isinstance(override, inspect.Signature):
                    inspect_sig = override
                else:
                    if override is not None:
                        constructor = override
                    # We pick __new__ if it is implemented as a Python function only;
                    # if we picked it whenever it was overridden we'd get too many C
                    # types that have a meaningless __new__ signature. Typeshed
                    # usually doesn't have a __new__ signature. Alternatively, we
                    # could try __new__ first and fall back to __init__ if __new__
                    # doesn't have a useful signature.
                    # In practice, we saw this make a difference with NamedTuples.
                    elif isinstance(obj.__new__, FunctionType):
                        constructor = obj.__new__
                    else:
                        constructor = obj.__init__
                    inspect_sig = self._safe_get_signature(constructor)
                if inspect_sig is None:
                    return Signature.make(
                        [ELLIPSIS_PARAM],
                        return_type,
                        callable=obj,
                        allow_call=allow_call,
                    )

                signature = self.from_signature(
                    inspect_sig,
                    function_object=obj,
                    impl=impl,
                    returns=return_type,
                    allow_call=allow_call,
                )
            bound_sig = make_bound_method(signature, Composite(TypedValue(obj)))
            if bound_sig is None:
                return None
            sig = bound_sig.get_signature(preserve_impl=True)
            if sig is not None:
                return sig
            return bound_sig

        if inspect.isbuiltin(obj):
            if not isinstance(obj.__self__, ModuleType):
                cls = type(obj.__self__)
                try:
                    method = getattr(cls, obj.__name__)
                except AttributeError:
                    return self._make_any_sig(obj)
                if method == obj:
                    return self._make_any_sig(obj)
                argspec = self._cached_get_argspec(
                    method, impl, is_asynq, in_overload_resolution
                )
                return make_bound_method(argspec, Composite(KnownValue(obj.__self__)))
            inspect_sig = self._safe_get_signature(obj)
            if inspect_sig is not None:
                return self.from_signature(inspect_sig, function_object=obj)
            return self._make_any_sig(obj)

        if hasattr(obj, "__call__"):
            # we could get an argspec here in some cases, but it's impossible to figure out
            # the argspec for some builtin methods (e.g., dict.__init__), and no way to detect
            # these with inspect, so just give up.
            return self._make_any_sig(obj)

        if isinstance(obj, property):
            # If we know the getter, inherit its return value.
            if obj.fget:
                fget_argspec = self._cached_get_argspec(
                    obj.fget, impl, is_asynq, in_overload_resolution
                )
                if fget_argspec is not None and fget_argspec.has_return_value():
                    return PropertyArgSpec(obj, return_value=fget_argspec.return_value)
            return PropertyArgSpec(obj)

        return None

    def _make_any_sig(self, obj: object) -> Signature:
        if FunctionsSafeToCall.contains(obj, self.options):
            return Signature.make(
                [ELLIPSIS_PARAM],
                AnyValue(AnySource.inference),
                is_asynq=True,
                allow_call=True,
                callable=obj,
            )
        else:
            return ANY_SIGNATURE

    def _safe_get_signature(self, obj: Any) -> Optional[inspect.Signature]:
        """Wrapper around inspect.getargspec that catches TypeErrors."""
        try:
            # follow_wrapped=True leads to problems with decorators that
            # mess with the arguments, such as mock.patch.
            return inspect.signature(obj, follow_wrapped=False)
        except (TypeError, ValueError, AttributeError):
            # TypeError if signature() does not support the object, ValueError
            # if it cannot provide a signature, and AttributeError if we're on
            # Python 2.
            return None

    def get_generic_bases(
        self, typ: Union[type, str], generic_args: Sequence[Value] = ()
    ) -> GenericBases:
        if (
            typ is Generic
            or is_typing_name(typ, "Protocol")
            or typ is object
            or typ in ("typing.Generic", "builtins.object")
        ):
            return {}
        generic_bases = self._get_generic_bases_cached(typ)
        if typ not in generic_bases:
            return generic_bases
        my_typevars = generic_bases[typ]
        if not my_typevars:
            return generic_bases
        tv_map = {}
        for i, tv_value in enumerate(my_typevars.values()):
            if not isinstance(tv_value, TypeVarValue):
                continue
            try:
                value = generic_args[i]
            except IndexError:
                value = AnyValue(AnySource.generic_argument)
            tv_map[tv_value.typevar] = value
        return {
            base: {tv: value.substitute_typevars(tv_map) for tv, value in args.items()}
            for base, args in generic_bases.items()
        }

    def _get_generic_bases_cached(self, typ: Union[type, str]) -> GenericBases:
        try:
            return self.generic_bases_cache[typ]
        except KeyError:
            pass
        except Exception:
            return {}  # We don't support unhashable types.
        if isinstance(typ, str):
            bases = self.ts_finder.get_bases_for_fq_name(typ)
        else:
            bases = self.ts_finder.get_bases(typ)
        generic_bases = self._extract_bases(typ, bases)
        if generic_bases is None:
            assert isinstance(
                typ, type
            ), f"failed to extract typeshed bases for {typ!r}"
            bases = [
                type_from_runtime(base, ctx=self.default_context)
                for base in self.get_runtime_bases(typ)
            ]
            generic_bases = self._extract_bases(typ, bases)
            assert (
                generic_bases is not None
            ), f"failed to extract runtime bases from {typ}"
        self.generic_bases_cache[typ] = generic_bases
        return generic_bases

    def _extract_bases(
        self, typ: Union[type, str], bases: Optional[Sequence[Value]]
    ) -> Optional[GenericBases]:
        if bases is None:
            return None
        # Put Generic first since it determines the order of the typevars. This matters
        # for typing.Coroutine.
        bases = sorted(
            bases,
            key=lambda base: not isinstance(base, TypedValue)
            or base.typ is not Generic,
        )
        my_typevars = uniq_chain(extract_typevars(base) for base in bases)
        generic_bases = {}
        generic_bases[typ] = {tv: TypeVarValue(tv) for tv in my_typevars}
        for base in bases:
            if isinstance(base, TypedValue):
                if isinstance(base.typ, str):
                    assert base.typ != typ, base
                else:
                    assert base.typ is not typ, base
                if isinstance(base, GenericValue):
                    args = base.args
                else:
                    args = ()
                generic_bases.update(self.get_generic_bases(base.typ, args))
            else:
                return None
        return generic_bases

    def get_runtime_bases(self, typ: type) -> Sequence[object]:
        if typing_inspect.is_generic_type(typ):
            return typing_inspect.get_generic_bases(typ)
        return typ.__bases__


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


def _get_class_name(obj: object) -> Optional[str]:
    if hasattr(obj, "__qualname__"):
        pieces = obj.__qualname__.split(".")
        if len(pieces) >= 2:
            return pieces[-2]
    return None
