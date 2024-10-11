"""

The checker maintains global state that is preserved across different modules.

"""

import collections.abc
import itertools
import sys
import types
from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import InitVar, dataclass, field
from typing import Callable, Optional, Union

import qcore

from .arg_spec import ArgSpecCache, GenericBases
from .attributes import AttrContext, get_attribute
from .node_visitor import Failure
from .options import Options, PyObjectSequenceOption
from .reexport import ImplicitReexportTracker
from .safe import is_instance_of_typing_name, is_typing_name, safe_getattr
from .shared_options import VariableNameValues
from .signature import (
    ANY_SIGNATURE,
    BoundMethodSignature,
    ConcreteSignature,
    MaybeSignature,
    OverloadedSignature,
    Signature,
    make_bound_method,
)
from .stacked_scopes import Composite
from .suggested_type import CallableTracker
from .type_object import TypeObject, get_mro
from .typeshed import TypeshedFinder
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnyValue,
    CallableValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    SubclassValue,
    TypeAlias,
    TypedValue,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    VariableNameValue,
    flatten_values,
    is_union,
    unite_values,
)

_BaseProvider = Callable[[Union[type, super]], set[type]]


class AdditionalBaseProviders(PyObjectSequenceOption[_BaseProvider]):
    """Sets functions that provide additional (virtual) base classes for a class.
    These are used for the purpose of type checking.

    For example, if the following is configured to be used as a base provider:

        def provider(typ: type) -> Set[type]:
            if typ is B:
                return {A}
            return set()

    Then to the type checker `B` is a subclass of `A`.

    """

    name = "additional_base_providers"


@dataclass
class Checker:
    raw_options: InitVar[Optional[Options]] = None
    options: Options = field(init=False)
    arg_spec_cache: ArgSpecCache = field(init=False)
    ts_finder: TypeshedFinder = field(init=False)
    reexport_tracker: ImplicitReexportTracker = field(init=False)
    callable_tracker: CallableTracker = field(init=False)
    type_object_cache: dict[Union[type, super, str], TypeObject] = field(
        default_factory=dict, init=False, repr=False
    )
    assumed_compatibilities: list[tuple[TypeObject, TypeObject]] = field(
        default_factory=list
    )
    vnv_map: dict[str, VariableNameValue] = field(default_factory=dict)
    type_alias_cache: dict[object, TypeAlias] = field(default_factory=dict)
    _should_exclude_any: bool = False
    _has_used_any_match: bool = False

    def __post_init__(self, raw_options: Optional[Options]) -> None:
        if raw_options is None:
            self.options = Options.from_option_list()
        else:
            self.options = raw_options
        self.ts_finder = TypeshedFinder.make(self, self.options)
        self.arg_spec_cache = ArgSpecCache(
            self.options,
            self.ts_finder,
            self,
            vnv_provider=self.maybe_get_variable_name_value,
        )
        self.reexport_tracker = ImplicitReexportTracker(self.options)
        self.callable_tracker = CallableTracker()

        for vnv in self.options.get_value_for(VariableNameValues):
            for variable in vnv.varnames:
                self.vnv_map[variable] = vnv

    def maybe_get_variable_name_value(
        self, varname: str
    ) -> Optional[VariableNameValue]:
        return VariableNameValue.from_varname(varname, self.vnv_map)

    def perform_final_checks(self) -> list[Failure]:
        return self.callable_tracker.check()

    def get_additional_bases(self, typ: Union[type, super]) -> set[type]:
        bases = set()
        for provider in self.options.get_value_for(AdditionalBaseProviders):
            bases |= provider(typ)
        return bases

    def make_type_object(self, typ: Union[type, super, str]) -> TypeObject:
        try:
            in_cache = typ in self.type_object_cache
        except Exception:
            return self._build_type_object(typ)
        if in_cache:
            return self.type_object_cache[typ]
        type_object = self._build_type_object(typ)
        self.type_object_cache[typ] = type_object
        return type_object

    def _build_type_object(self, typ: Union[type, super, str]) -> TypeObject:
        if isinstance(typ, str):
            # Synthetic type
            bases = self._get_typeshed_bases(typ)
            is_protocol = any(is_typing_name(base, "Protocol") for base in bases)
            if is_protocol:
                protocol_members = self._get_protocol_members(bases)
            else:
                protocol_members = set()
            return TypeObject(
                typ, bases, is_protocol=is_protocol, protocol_members=protocol_members
            )
        elif isinstance(typ, super):
            return TypeObject(typ, self.get_additional_bases(typ))
        else:
            plugin_bases = self.get_additional_bases(typ)
            typeshed_bases = self._get_recursive_typeshed_bases(typ)
            additional_bases = plugin_bases | typeshed_bases
            # Is it marked as a protocol in stubs? If so, use the stub definition.
            if self.ts_finder.is_protocol(typ):
                return TypeObject(
                    typ,
                    additional_bases,
                    is_protocol=True,
                    protocol_members=self._get_protocol_members(typeshed_bases),
                )
            # Is it a protocol at runtime?
            if is_instance_of_typing_name(typ, "_ProtocolMeta") and safe_getattr(
                typ, "_is_protocol", False
            ):
                bases = get_mro(typ)
                members = set(
                    itertools.chain.from_iterable(
                        _extract_protocol_members(base) for base in bases
                    )
                )
                return TypeObject(
                    typ, additional_bases, is_protocol=True, protocol_members=members
                )

            return TypeObject(typ, additional_bases)

    def _get_recursive_typeshed_bases(
        self, typ: Union[type, str]
    ) -> set[Union[type, str]]:
        seen = set()
        to_do = {typ}
        result = set()
        while to_do:
            typ = to_do.pop()
            if typ in seen:
                continue
            bases = self._get_typeshed_bases(typ)
            result |= bases
            to_do |= bases
            seen.add(typ)
        return result

    def _get_typeshed_bases(self, typ: Union[type, str]) -> set[Union[type, str]]:
        base_values = self.ts_finder.get_bases_recursively(typ)
        return {base.typ for base in base_values if isinstance(base, TypedValue)}

    def _get_protocol_members(self, bases: Iterable[Union[type, str]]) -> set[str]:
        return set(
            itertools.chain.from_iterable(
                self.ts_finder.get_all_attributes(base) for base in bases
            )
        )

    def get_generic_bases(
        self, typ: Union[type, str], generic_args: Sequence[Value] = ()
    ) -> GenericBases:
        return self.arg_spec_cache.get_generic_bases(typ, generic_args)

    def get_signature(
        self, obj: object, is_asynq: bool = False
    ) -> Optional[ConcreteSignature]:
        sig = self.arg_spec_cache.get_argspec(obj, is_asynq=is_asynq)
        if isinstance(sig, Signature):
            return sig
        elif isinstance(sig, BoundMethodSignature):
            return sig.get_signature(ctx=self)
        elif isinstance(sig, OverloadedSignature):
            return sig
        return None

    def can_assume_compatibility(self, left: TypeObject, right: TypeObject) -> bool:
        return (left, right) in self.assumed_compatibilities

    @contextmanager
    def assume_compatibility(
        self, left: TypeObject, right: TypeObject
    ) -> Iterator[None]:
        """Context manager that notes that left and right can be assumed to be compatible."""
        pair = (left, right)
        self.assumed_compatibilities.append(pair)
        try:
            yield
        finally:
            new_pair = self.assumed_compatibilities.pop()
            assert pair == new_pair

    def display_value(self, value: Value) -> str:
        message = f"'{value!s}'"
        if isinstance(value, KnownValue):
            sig = self.arg_spec_cache.get_argspec(value.val)
        elif isinstance(value, UnboundMethodValue):
            sig = value.get_signature(self)
        elif isinstance(value, SubclassValue) and value.exactly:
            sig = self.signature_from_value(value)
        else:
            sig = None
        if sig is not None:
            message += f", signature is {sig!s}"
        return message

    def has_used_any_match(self) -> bool:
        """Whether Any was used to secure a match."""
        return self._has_used_any_match

    def record_any_used(self) -> None:
        """Record that Any was used to secure a match."""
        self._has_used_any_match = True

    def reset_any_used(self) -> AbstractContextManager[None]:
        """Context that resets the value used by :meth:`has_used_any_match` and
        :meth:`record_any_match`."""
        return qcore.override(self, "_has_used_any_match", False)

    def set_exclude_any(self) -> AbstractContextManager[None]:
        """Within this context, `Any` is compatible only with itself."""
        return qcore.override(self, "_should_exclude_any", True)

    def should_exclude_any(self) -> bool:
        """Whether Any should be compatible only with itself."""
        return self._should_exclude_any

    def signature_from_value(
        self,
        value: Value,
        *,
        get_return_override: Callable[
            [MaybeSignature], Optional[Value]
        ] = lambda _: None,
        get_call_attribute: Optional[Callable[[Value], Value]] = None,
    ) -> MaybeSignature:
        if isinstance(value, AnnotatedValue):
            value = value.value
        if isinstance(value, TypeVarValue):
            value = value.get_fallback_value()
        if isinstance(value, KnownValue):
            argspec = self.arg_spec_cache.get_argspec(value.val)
            if argspec is None:
                if get_call_attribute is not None:
                    method_object = get_call_attribute(value)
                else:
                    method_object = self.get_attribute_from_value(value, "__call__")
                if method_object is UNINITIALIZED_VALUE:
                    return None
                else:
                    return ANY_SIGNATURE
            if isinstance(value, KnownValueWithTypeVars):
                return argspec.substitute_typevars(value.typevars)
            return argspec
        elif isinstance(value, UnboundMethodValue):
            method = value.get_method()
            if method is not None:
                sig = self.arg_spec_cache.get_argspec(method)
                if sig is None:
                    # TODO return None here and figure out when the signature is missing
                    # Probably because of cythonized methods
                    return ANY_SIGNATURE
                return_override = get_return_override(sig)
                bound = make_bound_method(
                    sig, value.composite, return_override, ctx=self
                )
                if bound is not None and value.typevars is not None:
                    bound = bound.substitute_typevars(value.typevars)
                return bound
            return None
        elif isinstance(value, CallableValue):
            return value.signature
        elif isinstance(value, TypedValue):
            typ = value.typ
            if typ is collections.abc.Callable or typ is types.FunctionType:
                return ANY_SIGNATURE
            if isinstance(typ, str):
                if get_call_attribute is not None:
                    call_method = get_call_attribute(value)
                else:
                    call_method = self.get_attribute_from_value(value, "__call__")
                if call_method is UNINITIALIZED_VALUE:
                    return None
                return self.signature_from_value(
                    call_method,
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
            if getattr(typ.__call__, "__objclass__", None) is type and not issubclass(
                typ, type
            ):
                return None
            call_fn = typ.__call__
            sig = self.arg_spec_cache.get_argspec(call_fn)
            return_override = get_return_override(sig)
            bound_method = make_bound_method(
                sig, Composite(value), return_override, ctx=self
            )
            if bound_method is None:
                return None
            return bound_method.get_signature(ctx=self)
        elif isinstance(value, SubclassValue):
            if isinstance(value.typ, TypedValue):
                if value.typ.typ is tuple:
                    # Probably an unknown namedtuple
                    return ANY_SIGNATURE
                argspec = self.arg_spec_cache.get_argspec(
                    value.typ.typ, allow_synthetic_type=True
                )
                if argspec is None:
                    return ANY_SIGNATURE
                return argspec
            else:
                # TODO generic SubclassValue
                return ANY_SIGNATURE
        elif isinstance(value, AnyValue):
            return ANY_SIGNATURE
        elif isinstance(value, MultiValuedValue):
            sigs = [
                self.signature_from_value(
                    subval,
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
                for subval in value.vals
            ]
            if all(sig is not None for sig in sigs):
                # TODO we can't return a Union if we get here
                return ANY_SIGNATURE
            else:
                return None
        else:
            return None

    def get_attribute_from_value(
        self, root_value: Value, attribute: str, *, prefer_typeshed: bool = False
    ) -> Value:
        if isinstance(root_value, TypeVarValue):
            root_value = root_value.get_fallback_value()
        if is_union(root_value):
            results = [
                self.get_attribute_from_value(
                    subval, attribute, prefer_typeshed=prefer_typeshed
                )
                for subval in flatten_values(root_value)
            ]
            return unite_values(*results)
        ctx = CheckerAttrContext(
            Composite(root_value),
            attribute,
            self.options,
            skip_mro=False,
            skip_unwrap=False,
            prefer_typeshed=prefer_typeshed,
            checker=self,
        )
        return get_attribute(ctx)


EXCLUDED_PROTOCOL_MEMBERS = {
    "__abstractmethods__",
    "__annotate__",
    "__annotations__",
    "__dict__",
    "__doc__",
    "__init__",
    "__new__",
    "__module__",
    "__parameters__",
    "__subclasshook__",
    "__weakref__",
    "_abc_impl",
    "_abc_cache",
    "_is_protocol",
    "__next_in_mro__",
    "_abc_generic_negative_cache_version",
    "__orig_bases__",
    "__args__",
    "_abc_registry",
    "__extra__",
    "_abc_generic_negative_cache",
    "__origin__",
    "__tree_hash__",
    "_gorg",
    "_is_runtime_protocol",
    "__protocol_attrs__",
    "__callable_proto_members_only__",
    "__non_callable_proto_members__",
    "__static_attributes__",
    "__firstlineno__",
}


def _extract_protocol_members(typ: type) -> set[str]:
    if (
        typ is object
        or is_typing_name(typ, "Generic")
        or is_typing_name(typ, "Protocol")
    ):
        return set()
    members = set(typ.__dict__) - EXCLUDED_PROTOCOL_MEMBERS
    # Starting in 3.10 __annotations__ always exists on types
    if sys.version_info >= (3, 10) or hasattr(typ, "__annotations__"):
        members |= set(typ.__annotations__)
    return members


@dataclass
class CheckerAttrContext(AttrContext):
    checker: Checker

    def resolve_name_from_typeshed(self, module: str, name: str) -> Value:
        return self.checker.ts_finder.resolve_name(module, name)

    def get_attribute_from_typeshed(self, typ: type, *, on_class: bool) -> Value:
        return self.checker.ts_finder.get_attribute(typ, self.attr, on_class=on_class)

    def get_attribute_from_typeshed_recursively(
        self, fq_name: str, *, on_class: bool
    ) -> tuple[Value, object]:
        return self.checker.ts_finder.get_attribute_recursively(
            fq_name, self.attr, on_class=on_class
        )

    def get_signature(self, obj: object) -> MaybeSignature:
        return self.checker.signature_from_value(KnownValue(obj))

    def get_generic_bases(
        self, typ: Union[type, str], generic_args: Sequence[Value]
    ) -> GenericBases:
        return self.checker.get_generic_bases(typ, generic_args)
