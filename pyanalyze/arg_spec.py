"""

Implementation of extended argument specifications used by test_scope.

"""

from .extensions import get_overloads
from .annotations import Context, type_from_runtime
from .config import Config
from .find_unused import used
from . import implementation
from .safe import (
    all_of_type,
    is_newtype,
    safe_hasattr,
    safe_in,
    safe_issubclass,
    is_typing_name,
)
from .stacked_scopes import Composite, uniq_chain
from .signature import (
    ANY_SIGNATURE,
    Impl,
    MaybeSignature,
    OverloadedSignature,
    PropertyArgSpec,
    make_bound_method,
    SigParameter,
    Signature,
)
from .typeshed import TypeshedFinder
from .value import (
    AnySource,
    AnyValue,
    GenericBases,
    TypedValue,
    GenericValue,
    NewTypeValue,
    KnownValue,
    Value,
    VariableNameValue,
    TypeVarValue,
    extract_typevars,
)

import ast
import asyncio
import asynq
from collections.abc import Awaitable
import contextlib
from dataclasses import dataclass
import qcore
import inspect
import sys
from types import FunctionType, ModuleType
from typing import Any, Sequence, Generic, Iterable, Mapping, Optional, Union
import typing_inspect


@used  # exposed as an API
@contextlib.contextmanager
def with_implementation(fn: object, implementation_fn: Impl) -> Iterable[None]:
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
        argspec = ArgSpecCache(Config()).get_argspec(fn, impl=implementation_fn)
        if argspec is None:
            # builtin or something, just use a generic argspec
            argspec = Signature.make(
                [
                    SigParameter("args", SigParameter.VAR_POSITIONAL),
                    SigParameter("kwargs", SigParameter.VAR_KEYWORD),
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


class ArgSpecCache:
    DEFAULT_ARGSPECS = implementation.get_default_argspecs()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.ts_finder = TypeshedFinder(verbose=False)
        self.known_argspecs = {}
        self.generic_bases_cache = {}
        self.default_context = AnnotationsContext(self)
        default_argspecs = dict(self.DEFAULT_ARGSPECS)
        default_argspecs.update(self.config.get_known_argspecs(self))

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
            parameters.append(
                self._make_sig_parameter(
                    parameter, func_globals, function_object, is_wrapped, i
                )
            )

        return Signature.make(
            parameters,
            returns,
            impl=impl,
            callable=function_object,
            has_return_annotation=has_return_annotation,
            is_asynq=is_asynq,
            allow_call=allow_call
            or safe_in(function_object, self.config.FUNCTIONS_SAFE_TO_CALL),
        )

    def _make_sig_parameter(
        self,
        parameter: inspect.Parameter,
        func_globals: Optional[Mapping[str, object]],
        function_object: Optional[object],
        is_wrapped: bool,
        index: int,
    ) -> SigParameter:
        """Given an inspect.Parameter, returns a Parameter object."""
        if is_wrapped:
            typ = AnyValue(AnySource.inference)
        else:
            typ = self._get_type_for_parameter(
                parameter, func_globals, function_object, index
            )
        if parameter.default is SigParameter.empty:
            default = None
        else:
            default = KnownValue(parameter.default)
        return SigParameter(
            parameter.name, parameter.kind, default=default, annotation=typ
        )

    def _get_type_for_parameter(
        self,
        parameter: inspect.Parameter,
        func_globals: Optional[Mapping[str, object]],
        function_object: Optional[object],
        index: int,
    ) -> Optional[Value]:
        if parameter.annotation is not SigParameter.empty:
            typ = type_from_runtime(
                parameter.annotation, ctx=AnnotationsContext(self, func_globals)
            )
            if parameter.kind is SigParameter.VAR_POSITIONAL:
                return GenericValue(tuple, [typ])
            elif parameter.kind is SigParameter.VAR_KEYWORD:
                return GenericValue(dict, [TypedValue(str), typ])
            return typ
        # If this is the self argument of a method, try to infer the self type.
        elif index == 0 and parameter.kind in (
            SigParameter.POSITIONAL_ONLY,
            SigParameter.POSITIONAL_OR_KEYWORD,
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
            SigParameter.POSITIONAL_ONLY,
            SigParameter.POSITIONAL_OR_KEYWORD,
            SigParameter.KEYWORD_ONLY,
        ):
            return VariableNameValue.from_varname(
                parameter.name, self.config.varname_value_map()
            )
        return None

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

    def _uncached_get_argspec(
        self,
        obj: Any,
        impl: Optional[Impl],
        is_asynq: bool,
        in_overload_resolution: bool,
    ) -> MaybeSignature:
        if not in_overload_resolution:
            fq_name = _get_fully_qualified_name(obj)
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

        allow_call = safe_in(obj, self.config.FUNCTIONS_SAFE_TO_CALL)
        argspec = self.ts_finder.get_argspec(obj, allow_call=allow_call)
        if argspec is not None:
            return argspec

        if is_newtype(obj):
            return Signature.make(
                [
                    SigParameter(
                        "x",
                        SigParameter.POSITIONAL_ONLY,
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
            obj = self.config.unwrap_cls(obj)
            override = self.config.get_constructor(obj)
            if isinstance(override, Signature):
                signature = override
            else:
                should_ignore = safe_in(obj, self.config.IGNORED_CALLEES)
                return_type = (
                    AnyValue(AnySource.error) if should_ignore else TypedValue(obj)
                )
                allow_call = safe_issubclass(
                    obj, self.config.CLASSES_SAFE_TO_INSTANTIATE
                )
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
                        [],
                        return_type,
                        is_ellipsis_args=True,
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
        if safe_in(obj, self.config.FUNCTIONS_SAFE_TO_CALL):
            return Signature.make(
                [],
                AnyValue(AnySource.inference),
                is_ellipsis_args=True,
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


def _get_fully_qualified_name(obj: object) -> Optional[str]:
    if hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
        return f"{obj.__module__}.{obj.__qualname__}"
    return None
