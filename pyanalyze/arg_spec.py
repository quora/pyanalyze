"""

Implementation of extended argument specifications used by test_scope.

"""

import ast
import asyncio
import collections
import contextlib
import enum
import inspect
import sys
import textwrap
import typing
from dataclasses import dataclass, replace
from types import FunctionType, MethodType, ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unittest import mock

import asynq
import qcore
import typing_extensions
from typing_extensions import is_typeddict

import pyanalyze

from . import implementation
from .analysis_lib import is_positional_only_arg_name
from .annotations import Context, RuntimeEvaluator, type_from_runtime
from .extensions import CustomCheck, TypeGuard, get_type_evaluations
from .extensions import get_overloads as pyanalyze_get_overloads
from .find_unused import used
from .functions import translate_vararg_type
from .options import Options, PyObjectSequenceOption
from .safe import (
    all_of_type,
    get_fully_qualified_name,
    hasattr_static,
    is_newtype,
    is_typing_name,
    safe_equals,
    safe_getattr,
    safe_hasattr,
    safe_isinstance,
    safe_issubclass,
)
from .signature import (
    ANY_SIGNATURE,
    ELLIPSIS_PARAM,
    ConcreteSignature,
    Impl,
    MaybeSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
    make_bound_method,
)
from .stacked_scopes import Composite, uniq_chain
from .typeshed import TypeshedFinder
from .value import (
    AnySource,
    AnyValue,
    CanAssignContext,
    Extension,
    GenericBases,
    GenericValue,
    KnownValue,
    KVPair,
    NewTypeValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    SubclassValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeVarValue,
    Value,
    extract_typevars,
    make_coro_type,
)

_GET_OVERLOADS = []

try:
    from typing_extensions import get_overloads
except ImportError:
    pass
else:
    _GET_OVERLOADS.append(get_overloads)
if sys.version_info >= (3, 11):
    # TODO: support version checks
    # static analysis: ignore[undefined_attribute]
    _GET_OVERLOADS.append(typing.get_overloads)

# types.MethodWrapperType in 3.7+
MethodWrapperType = type(object().__str__)

_ENUM_CALL = enum.Enum.__call__.__func__

_SELF_PARAM = inspect.Parameter("__self", inspect.Parameter.POSITIONAL_ONLY)


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
        checker = pyanalyze.checker.Checker()
        argspec = checker.arg_spec_cache.get_argspec(fn, impl=implementation_fn)
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


TYPING_OBJECTS_SAFE_TO_CALL = [
    getattr(mod, name)
    for mod in (typing, typing_extensions)
    for name in (
        "TypeVar",
        "TypeVarTuple",
        "ParamSpec",
        "NewType",
        "TypeAliasType",
        "NamedTuple",
        "TypedDict",
        "deprecated",
        "dataclass_transform",
    )
    if hasattr(mod, name)
]


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
        TypedDictEntry,
        asynq.ConstFuture,
        range,
        tuple,
        *[obj for obj in TYPING_OBJECTS_SAFE_TO_CALL if safe_isinstance(obj, type)],
    ]


class FunctionsSafeToCall(PyObjectSequenceOption[object]):
    """We will instantiate instances of these classes if we can infer the value of all of
    their arguments. This is useful mostly for classes that are commonly instantiated with static
    arguments."""

    name = "functions_safe_to_call"
    default_value = [
        sorted,
        asynq.asynq,
        collections.namedtuple,
        *[obj for obj in TYPING_OBJECTS_SAFE_TO_CALL if not safe_isinstance(obj, type)],
    ]


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
    def unwrap(cls, typ: type, options: Options) -> type:
        for unwrapper in options.get_value_for(cls):
            typ = unwrapper(typ)
        return typ


_BUILTIN_KNOWN_SIGNATURES = []

try:
    import _pytest
    import pytest
except ImportError:
    # if pytest is not installed in this environment, don't use it
    pass
else:
    _E = TypeVar("_E", bound=BaseException)

    # pytest.raises gets imported before our @overload
    # monkeypatch takes effect, so we need to manually
    # specify the overloads. This will be unnecessary in 3.11+
    # where we get to use typing.get_overloads().
    def _raises_overload1(
        expected_exception: Union[Type[_E], Tuple[Type[_E], ...]],
        *,
        match: Optional[Union[str, Pattern[str]]] = ...,
    ) -> _pytest.python_api.RaisesContext[_E]:
        raise NotImplementedError

    # TODO use ParamSpec here
    def _raises_overload2(
        expected_exception: Union[Type[_E], Tuple[Type[_E], ...]],
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> _pytest._code.ExceptionInfo[_E]:
        raise NotImplementedError

    def _get_pytest_signatures(
        arg_spec_cache: "ArgSpecCache",
    ) -> Dict[object, ConcreteSignature]:
        """Return hardcoded Signatures for specific pytest functions."""
        sigs = [
            arg_spec_cache.get_concrete_signature(_raises_overload1),
            arg_spec_cache.get_concrete_signature(_raises_overload2),
        ]
        return {pytest.raises: OverloadedSignature(sigs)}

    _BUILTIN_KNOWN_SIGNATURES.append(_get_pytest_signatures)


class ArgSpecCache:
    DEFAULT_ARGSPECS = implementation.get_default_argspecs()

    def __init__(
        self,
        options: Options,
        ts_finder: TypeshedFinder,
        ctx: CanAssignContext,
        *,
        vnv_provider: Callable[[str], Optional[Value]] = lambda _: None,
    ) -> None:
        self.vnv_provider = vnv_provider
        self.options = options
        self.ts_finder = ts_finder
        self.ctx = ctx
        self.known_argspecs = {}
        self.generic_bases_cache = {}
        self.default_context = AnnotationsContext(self)
        self.safe_bases = tuple(self.options.get_value_for(ClassesSafeToInstantiate))

        default_argspecs = dict(self.DEFAULT_ARGSPECS)
        for provider in _BUILTIN_KNOWN_SIGNATURES:
            default_argspecs.update(provider(self))
        for provider in options.get_value_for(KnownSignatures):
            default_argspecs.update(provider(self))

        for obj, argspec in default_argspecs.items():
            self.known_argspecs[obj] = argspec

    def from_signature(
        self,
        sig: inspect.Signature,
        *,
        impl: Optional[Impl] = None,
        callable_object: object,
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
        is_wrapped = hasattr_static(function_object, "__wrapped__")

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
                returns = make_coro_type(returns)

        parameters = []
        seen_paramspec_args: Optional[ParamSpecArgsValue] = None
        for i, parameter in enumerate(sig.parameters.values()):
            param, make_everything_pos_only, new_ps_args = self._make_sig_parameter(
                parameter,
                func_globals,
                function_object,
                is_wrapped,
                i,
                seen_paramspec_args,
            )
            if make_everything_pos_only:
                parameters = [
                    replace(param, kind=ParameterKind.POSITIONAL_ONLY)
                    for param in parameters
                ]
            if new_ps_args is not None:
                seen_paramspec_args = new_ps_args
            if param is None:
                continue
            parameters.append(param)

        return Signature.make(
            parameters,
            returns,
            impl=impl,
            callable=callable_object,
            has_return_annotation=has_return_annotation,
            is_asynq=is_asynq,
            allow_call=allow_call
            or FunctionsSafeToCall.contains(callable_object, self.options),
        )

    def _make_sig_parameter(
        self,
        parameter: inspect.Parameter,
        func_globals: Optional[Mapping[str, object]],
        function_object: Optional[object],
        is_wrapped: bool,
        index: int,
        seen_paramspec_args: Optional[ParamSpecArgsValue],
    ) -> Tuple[Optional[SigParameter], bool, Optional[ParamSpecArgsValue]]:
        """Given an inspect.Parameter, returns a Parameter object."""
        if is_wrapped:
            typ = AnyValue(AnySource.inference)
        else:
            typ = self._get_type_for_parameter(
                parameter, func_globals, function_object, index
            )
        if (
            isinstance(typ, ParamSpecArgsValue)
            and parameter.kind is inspect.Parameter.VAR_POSITIONAL
        ):
            return (None, False, typ)
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
        if (
            seen_paramspec_args is not None
            and kind is ParameterKind.VAR_KEYWORD
            and isinstance(typ, ParamSpecKwargsValue)
            and seen_paramspec_args.param_spec is typ.param_spec
        ):
            kind = ParameterKind.PARAM_SPEC
            typ = TypeVarValue(typ.param_spec)
        return (
            SigParameter(parameter.name, kind, default=default, annotation=typ),
            make_everything_pos_only,
            None,
        )

    def _get_type_for_parameter(
        self,
        parameter: inspect.Parameter,
        func_globals: Optional[Mapping[str, object]],
        function_object: Optional[object],
        index: int,
    ) -> Value:
        if parameter.annotation is not inspect.Parameter.empty:
            kind = ParameterKind(parameter.kind)
            ctx = AnnotationsContext(self, func_globals)
            typ = type_from_runtime(
                parameter.annotation, ctx=ctx, allow_unpack=kind.allow_unpack()
            )
            return translate_vararg_type(kind, typ, self.ctx)
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
        self,
        obj: object,
        impl: Optional[Impl] = None,
        is_asynq: bool = False,
        allow_synthetic_type: bool = False,
    ) -> MaybeSignature:
        """Constructs the Signature for a Python object."""
        if safe_isinstance(obj, str) and not allow_synthetic_type:
            return None
        return self._cached_get_argspec(
            obj, impl, is_asynq, in_overload_resolution=False
        )

    def get_concrete_signature(
        self, obj: object, impl: Optional[Impl] = None, *, allow_call: bool = False
    ) -> Signature:
        """Return a concrete signature for an object."""
        sig = self.get_argspec(obj, impl=impl)
        if not isinstance(sig, Signature):
            raise TypeError(f"failed to find a concrete signature or {obj}")
        return replace(sig, allow_call=allow_call)

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
        if isinstance(obj, tuple):
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
                    return make_bound_method(
                        sig, Composite(KnownValue(obj.__self__)), ctx=self.ctx
                    )

        # for bound methods, see if we have an argspec for the unbound method
        if inspect.ismethod(obj) and obj.__self__ is not None:
            argspec = self._cached_get_argspec(
                obj.__func__, impl, is_asynq, in_overload_resolution
            )
            return make_bound_method(
                argspec, Composite(KnownValue(obj.__self__)), ctx=self.ctx
            )

        # Must be after the check for bound methods, because otherwise we
        # won't bind self correctly.
        if not in_overload_resolution:
            for get_overloads_func in _GET_OVERLOADS:
                inner_obj = safe_getattr(obj, "__func__", obj)
                if safe_hasattr(inner_obj, "__module__") and safe_hasattr(
                    inner_obj, "__qualname__"
                ):
                    sig = self._maybe_make_overloaded_signature(
                        get_overloads_func(inner_obj), impl, is_asynq
                    )
                    if sig is not None:
                        return sig
            fq_name = get_fully_qualified_name(obj)
            if fq_name is not None:
                sig = self._maybe_make_overloaded_signature(
                    pyanalyze_get_overloads(fq_name), impl, is_asynq
                )
                if sig is not None:
                    return sig
                evaluator_sig = self._maybe_make_evaluator_sig(obj, impl, is_asynq)
                if evaluator_sig is not None:
                    return evaluator_sig

        if hasattr_static(obj, "fn") or hasattr_static(obj, "original_fn"):
            is_asynq = is_asynq or hasattr_static(obj, "asynq")
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

        # Special case for EnumMeta.__call__. Ideally this should be generalized.
        if (
            safe_isinstance(obj, type)
            and safe_issubclass(obj, enum.Enum)
            and safe_isinstance(obj.__call__, MethodType)
            and safe_equals(obj.__call__.__func__, _ENUM_CALL)
        ):
            signature = self._cached_get_argspec(
                _ENUM_CALL, impl, is_asynq, in_overload_resolution
            )
            self_value = SubclassValue(TypedValue(obj))
            bound_sig = make_bound_method(
                signature, Composite(self_value), ctx=self.ctx
            )
            if bound_sig is None:
                return None
            sig = bound_sig.get_signature(
                preserve_impl=True, ctx=self.ctx, self_annotation_value=self_value
            )
            if sig is not None:
                return sig
            return bound_sig

        allow_call = FunctionsSafeToCall.contains(obj, self.options) or (
            safe_isinstance(obj, type) and safe_issubclass(obj, self.safe_bases)
        )
        if safe_isinstance(obj, (type, str)):
            type_params = self.get_type_parameters(obj)
        else:
            type_params = []
        argspec = self.ts_finder.get_argspec(
            obj, allow_call=allow_call, type_params=type_params
        )
        if argspec is not None:
            return argspec

        if is_typeddict(obj) and not is_typing_name(obj, "TypedDict"):
            td_type = type_from_runtime(obj)
            if isinstance(td_type, TypedDictValue):
                params = [
                    SigParameter(
                        key,
                        ParameterKind.KEYWORD_ONLY,
                        default=None if entry.required else KnownValue(...),
                        annotation=entry.typ,
                    )
                    for key, entry in td_type.items.items()
                ]
                if td_type.extra_keys is not None:
                    annotation = GenericValue(
                        dict, [TypedValue(str), td_type.extra_keys]
                    )
                    params.append(
                        SigParameter(
                            "%kwargs", ParameterKind.VAR_KEYWORD, annotation=annotation
                        )
                    )
                return Signature.make(params, td_type, callable=obj)

        if is_newtype(obj):
            assert hasattr(obj, "__supertype__")
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
            if hasattr_static(obj, "inner"):
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
                callable_object=obj,
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
                return make_bound_method(
                    argspec, Composite(KnownValue(obj.instance)), ctx=self.ctx
                )
            return argspec

        if inspect.isclass(obj):
            obj = UnwrapClass.unwrap(obj, self.options)
            override = ConstructorHooks.get_constructor(obj, self.options)
            is_dunder_new = False
            if isinstance(override, Signature):
                signature = override
            else:
                should_ignore = IgnoredCallees.contains(obj, self.options)
                if should_ignore:
                    return_type = AnyValue(AnySource.error)
                elif type_params:
                    return_type = GenericValue(obj, type_params)
                else:
                    return_type = TypedValue(obj)
                if isinstance(override, inspect.Signature):
                    inspect_sig = override
                    constructor = None
                elif (
                    override is None
                    and hasattr_static(obj, "__signature__")
                    and safe_isinstance(obj.__signature__, inspect.Signature)
                ):
                    # Pydantic classes set a static __signature__ field
                    inspect_sig = obj.__signature__.replace(
                        parameters=[_SELF_PARAM, *obj.__signature__.parameters.values()]
                    )
                    constructor = obj
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
                        is_dunder_new = True
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
                    function_object=constructor,
                    callable_object=obj,
                    impl=impl,
                    returns=return_type,
                    allow_call=allow_call,
                )
            bound_sig = make_bound_method(
                signature, Composite(TypedValue(obj)), ctx=self.ctx
            )
            if bound_sig is None:
                return None
            if is_dunder_new:
                self_annotation_value = KnownValue(obj)
            else:
                if type_params:
                    self_annotation_value = GenericValue(obj, type_params)
                else:
                    self_annotation_value = TypedValue(obj)
            sig = bound_sig.get_signature(
                preserve_impl=True,
                ctx=self.ctx,
                self_annotation_value=self_annotation_value,
            )
            if sig is not None:
                return sig
            return bound_sig

        if inspect.isbuiltin(obj):
            if isinstance(obj.__self__, ModuleType):
                inspect_sig = self._safe_get_signature(obj)
                if inspect_sig is not None:
                    return self.from_signature(
                        inspect_sig, function_object=obj, callable_object=obj
                    )
                return self._make_any_sig(obj)
            else:
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
                return make_bound_method(
                    argspec, Composite(KnownValue(obj.__self__)), ctx=self.ctx
                )

        if hasattr_static(obj, "__call__"):
            # we could get an argspec here in some cases, but it's impossible to figure out
            # the argspec for some builtin methods (e.g., dict.__init__), and no way to detect
            # these with inspect, so just give up.
            return self._make_any_sig(obj)

        return None

    def _maybe_make_overloaded_signature(
        self,
        overloads: Sequence[Callable[..., Any]],
        impl: Optional[Impl],
        is_asynq: bool,
    ) -> Optional[OverloadedSignature]:
        if not overloads:
            return None
        sigs = [
            self._cached_get_argspec(
                overload, impl, is_asynq, in_overload_resolution=True
            )
            for overload in overloads
        ]
        if not all_of_type(sigs, Signature):
            return None
        return OverloadedSignature(sigs)

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

    def get_type_parameters(self, typ: Union[type, str]) -> List[Value]:
        bases = self.get_generic_bases(typ, substitute_typevars=False)
        tv_map = bases.get(typ, {})
        return [tv for tv in tv_map.values()]

    def get_generic_bases(
        self,
        typ: Union[type, str],
        generic_args: Sequence[Value] = (),
        *,
        substitute_typevars: bool = True,
    ) -> GenericBases:
        if (
            typ is Generic
            or is_typing_name(typ, "Protocol")
            or typ is object
            or typ in ("typing.Generic", "typing_extensions.Generic", "builtins.object")
        ):
            return {}
        generic_bases = self._get_generic_bases_cached(typ)
        if typ not in generic_bases:
            return generic_bases
        my_typevars = generic_bases[typ]
        if not my_typevars:
            return generic_bases
        tv_map = {}
        if substitute_typevars:
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
        # TODO: use typing_extensions.get_orig_bases()
        if is_typeddict(typ):
            return (dict,)
        try:
            return typ.__orig_bases__
        except AttributeError:
            return typ.__bases__


def _is_qcore_decorator(obj: object) -> TypeGuard[Any]:
    try:
        return (
            hasattr_static(obj, "is_decorator")
            and obj.is_decorator()
            and hasattr_static(obj, "decorator")
        )
    except Exception:
        # black.Line has an is_decorator attribute but it is not a method
        return False


def _get_class_name(obj: object) -> Optional[str]:
    if hasattr_static(obj, "__qualname__"):
        pieces = obj.__qualname__.split(".")
        if len(pieces) >= 2:
            return pieces[-2]
    return None
