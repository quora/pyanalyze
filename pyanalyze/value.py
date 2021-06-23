"""

Value classes represent the value of an expression in a Python program. Values
are the key data type used in pyanalyze's internals.

Values are instances of a subclass of :class:`Value`. This module defines
these subclasses and some related utilities.

:func:`dump_value` can be used to show inferred values during type checking. Examples::

    from typing import Any
    from pyanalyze import dump_value

    def function(x: int, y: list[int], z: Any):
        dump_value(1)  # KnownValue(1)
        dump_value(x)  # TypedValue(int)
        dump_value(y)  # GenericValue(list, [TypedValue(int)])
        dump_value(z)  # UnresolvedValue()

"""

import collections.abc
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field, InitVar
import inspect
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    Type,
    TypeVar,
)
from typing_extensions import Literal

import pyanalyze

from .safe import safe_isinstance, safe_issubclass
from .type_object import TypeObject

T = TypeVar("T")
# __builtin__ in Python 2 and builtins in Python 3
BUILTIN_MODULE = str.__module__
KNOWN_MUTABLE_TYPES = (list, set, dict, deque)

TypeVarMap = Mapping["TypeVar", "Value"]

_type_object_cache: Dict[Union[type, super], TypeObject] = {}


class Value:
    """Base class for all values."""

    __slots__ = ()

    def can_assign(self, other: "Value", ctx: "CanAssignContext") -> "CanAssign":
        """Whether other can be assigned to self.

        If yes, return a (possibly empty) map with the TypeVar values dictated by the
        assignment. If not, return a :class:`CanAssignError` explaining why the types
        are not compatible.

        For example, calling ``a.can_assign(b, ctx)`` where `a` is ``Iterable[T]``
        and `b` is ``List[int]`` will return ``{T: TypedValue(int)}``.

        This is the primary mechanism used for checking type compatibility.

        """
        if other is UNRESOLVED_VALUE or isinstance(other, VariableNameValue):
            return {}
        elif isinstance(other, MultiValuedValue):
            tv_maps = []
            for val in other.vals:
                tv_map = self.can_assign(val, ctx)
                if isinstance(tv_map, CanAssignError):
                    # Adding an additional layer here isn't helpful
                    return tv_map
                tv_maps.append(tv_map)
            if not tv_maps:
                return CanAssignError(f"Cannot assign {other} to {self}")
            return unify_typevar_maps(tv_maps)
        elif isinstance(other, AnnotatedValue):
            return self.can_assign(other.value, ctx)
        elif isinstance(other, TypeVarValue):
            return self.can_assign(other.get_fallback_value(), ctx)
        elif (
            isinstance(other, UnboundMethodValue)
            and other.secondary_attr_name is not None
        ):
            # Allow any UnboundMethodValue with a secondary attr; it might not be
            # a method.
            return {}
        elif self == other:
            return {}
        return CanAssignError(f"Cannot assign {other} to {self}")

    def is_assignable(self, other: "Value", ctx: "CanAssignContext") -> bool:
        """Similar to :meth:`can_assign` but returns a bool for simplicity."""
        return isinstance(self.can_assign(other, ctx), dict)

    def walk_values(self) -> Iterable["Value"]:
        """Iterator that yields all sub-values contained in this value."""
        yield self

    def substitute_typevars(self, typevars: TypeVarMap) -> "Value":
        """Substitute the typevars in the map to produce a new Value.

        This is used to specialize a generic. For example, substituting
        ``{T: int}`` on ``List[T]`` will produce ``List[int]``.

        """
        return self

    def is_type(self, typ: type) -> bool:
        """Returns whether this value is an instance of the given type.

        This method should be avoided. Use :meth:`can_assign` instead for
        checking compatibility.

        """
        return False

    def get_type(self) -> Optional[type]:
        """Returns the type of this value, or None if it is not known.

        This method should be avoided.

        """
        return None

    def get_type_value(self) -> "Value":
        """Return the type of this object as used for dunder lookups."""
        return self

    def __or__(self, other: "Value") -> "Value":
        """Shortcut for defining a MultiValuedValue."""
        return unite_values(self, other)

    def __ror__(self, other: "Value") -> "Value":
        return unite_values(other, self)


class CanAssignContext:
    """A context passed to the :meth:`Value.can_assign` method.

    Provides access to various functionality used for type checking.

    """

    def get_additional_bases(self, typ: Union[type, super]) -> Set[type]:
        """Return classes that should be considered base classes of `typ`.

        This can be used to make the type checker treat a type as a subclass
        of another when it is not actually a subtype at runtime.

        """
        return set()

    def make_type_object(self, typ: Union[type, super]) -> TypeObject:
        # Undocumented because TypeObject isn't a very useful concept;
        # we may be able to deprecate it.
        try:
            in_cache = typ in _type_object_cache
        except Exception:
            return TypeObject(typ, self.get_additional_bases(typ))
        if in_cache:
            return _type_object_cache[typ]
        type_object = TypeObject(typ, self.get_additional_bases(typ))
        _type_object_cache[typ] = type_object
        return type_object

    def get_generic_bases(
        self, typ: type, generic_args: Sequence["Value"] = ()
    ) -> Dict[type, TypeVarMap]:
        """Return the base classes for `typ` with their generic arguments.

        For example, calling
        ``ctx.get_generic_bases(dict, [TypedValue(int), TypedValue(str)])``
        may produce a map containing the following::

            {
                dict: [TypedValue(int), TypedValue(str)],
                Mapping: [TypedValue(int), TypedValue(str)],
                Iterable: [TypedValue(int)],
                Sized: [],
            }

        """
        return {}

    def get_signature(self, obj: object) -> Optional["pyanalyze.signature.Signature"]:
        """Return a :class:`pyanalyze.signature.Signature` for this object.

        Return None if the object is not callable.

        """
        return None

    def signature_from_value(
        self, value: "Value"
    ) -> "pyanalyze.signature.MaybeSignature":
        """Return a :class:`pyanalyze.signature.Signature` for a :class:`Value`.

        Return None if the object is not callable.

        """
        return None


@dataclass(frozen=True)
class CanAssignError:
    """A type checking error message with nested details.

    This exists in order to produce more useful error messages
    when there is a mismatch between complex types.

    """

    message: str
    children: List["CanAssignError"] = field(default_factory=list)

    def display(self, depth: int = 2) -> str:
        """Display all errors in a human-readable format."""
        result = f"{' ' * depth}{self.message}\n"
        result += "".join(child.display(depth=depth + 2) for child in self.children)
        return result

    def __str__(self) -> str:
        return self.display()


# Return value of CanAssign
CanAssign = Union[TypeVarMap, CanAssignError]


def assert_is_value(obj: object, value: Value) -> None:
    """Used to test pyanalyze's value inference.

    Takes two arguments: a Python object and a :class:`Value` object. At runtime
    this does nothing, but pyanalyze throws an error if the object is not
    inferred to be the same as the :class:`Value`.

    Example usage::

        assert_is_value(1, KnownValue(1))  # passes
        assert_is_value(1, TypedValue(int))  # shows an error

    """
    pass


def dump_value(value: object) -> None:
    """Print out the :class:`Value` representation of its argument.

    Calling it will make pyanalyze print out an internal
    representation of the argument's inferred value. Does nothing
    at runtime. Use :func:`pyanalyze.extensions.reveal_type` for a
    more user-friendly representation.

    """
    pass


@dataclass(frozen=True)
class UnresolvedValue(Value):
    """An unknown value, equivalent to ``typing.Any``."""

    def __str__(self) -> str:
        return "Any"

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        return {}  # Always allowed


UNRESOLVED_VALUE = UnresolvedValue()
"""The only instance of :class:`UnresolvedValue`."""


@dataclass(frozen=True)
class UninitializedValue(Value):
    """Value for variables that have not been initialized.

    Usage of variables with this value should be an error.

    """

    def __str__(self) -> str:
        return "<uninitialized>"


UNINITIALIZED_VALUE = UninitializedValue()
"""The only instance of :class:`UninitializedValue`."""


@dataclass(frozen=True)
class KnownValue(Value):
    """Equivalent to ``typing.Literal``. Represents a specific value.

    This is inferred for constants and for references to objects
    like modules, classes, and functions.

    """

    val: Any
    """The Python object that this ``KnownValue`` represents."""

    def is_type(self, typ: type) -> bool:
        return self.get_type_object().is_assignable_to_type(typ)

    def get_type(self) -> type:
        return type(self.val)

    def get_type_object(self) -> TypeObject:
        return TypeObject.make(type(self.val))

    def get_type_value(self) -> Value:
        return KnownValue(type(self.val))

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        # Make Literal[function] equivalent to a Callable type
        signature = ctx.get_signature(self.val)
        if signature is not None:
            return CallableValue(signature).can_assign(other, ctx)
        if isinstance(other, KnownValue):
            if self.val == other.val and type(self.val) is type(other.val):
                return {}
        return super().can_assign(other, ctx)

    def __eq__(self, other: Value) -> bool:
        return (
            isinstance(other, KnownValue)
            and type(self.val) is type(other.val)
            and self.val == other.val
        )

    def __ne__(self, other: Value) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        # Make sure e.g. 1 and True are handled differently.
        return hash((type(self.val), self.val))

    def __str__(self) -> str:
        if self.val is None:
            return "None"
        else:
            return "Literal[%r]" % (self.val,)


@dataclass(frozen=True)
class UnboundMethodValue(Value):
    """Value that represents a method on an underlying :class:`Value`.

    Despite the name this really represents a method bound to a value. For
    example, given ``s: str``, ``s.strip`` will be inferred as
    ``UnboundMethodValue("strip", TypedValue(str))``.

    """

    attr_name: str
    """Name of the method."""
    typ: Value
    """Value the method is bound to."""
    secondary_attr_name: Optional[str] = None
    """Used when an attribute is accessed on an existing ``UnboundMethodValue``.

    This is mostly useful in conjunction with asynq, where we might use
    ``object.method.asynq``. In that case, we would infer an ``UnboundMethodValue``
    with `secondary_attr_name` set to ``"asynq"``.

    """

    def get_method(self) -> Optional[Any]:
        """Return the runtime callable for this ``UnboundMethodValue``, or
        None if it cannot be found."""
        try:
            typ = self.typ.get_type()
            method = getattr(typ, self.attr_name)
            if self.secondary_attr_name is not None:
                method = getattr(method, self.secondary_attr_name)
            # don't use unbound methods in py2
            if inspect.ismethod(method) and method.__self__ is None:
                method = method.__func__
            return method
        except AttributeError:
            return None

    def is_type(self, typ: type) -> bool:
        return isinstance(self.get_method(), typ)

    def get_type(self) -> type:
        return type(self.get_method())

    def get_type_value(self) -> Value:
        return KnownValue(type(self.get_method()))

    def substitute_typevars(self, typevars: TypeVarMap) -> "Value":
        return UnboundMethodValue(
            self.attr_name,
            self.typ.substitute_typevars(typevars),
            self.secondary_attr_name,
        )

    def __str__(self) -> str:
        return "<method %s%s on %s>" % (
            self.attr_name,
            f".{self.secondary_attr_name}" if self.secondary_attr_name else "",
            self.typ,
        )


@dataclass(unsafe_hash=True)
class TypedValue(Value):
    """Value for which we know the type. This is equivalent to simple type
    annotations: an annotation of ``int`` will yield ``TypedValue(int)`` during
    type inference.

    """

    typ: Any
    """The underlying type."""
    type_object: TypeObject = field(init=False, repr=False, hash=False, compare=False)

    def __post_init__(self) -> None:
        self.type_object = TypeObject.make(self.typ)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if self.type_object.is_thrift_enum:
            # Special case: Thrift enums. These are conceptually like
            # enums, but they are ints at runtime.
            return self.can_assign_thrift_enum(other, ctx)
        elif isinstance(other, KnownValue):
            if safe_isinstance(other.val, self.typ):
                return {}
            if ctx.make_type_object(type(other.val)).is_assignable_to_type(self.typ):
                return {}
        elif isinstance(other, TypedValue):
            if ctx.make_type_object(other.typ).is_assignable_to_type(self.typ):
                return {}
        elif isinstance(other, SubclassValue):
            if isinstance(other.typ, TypedValue) and isinstance(
                other.typ.typ, self.typ
            ):
                return {}
            elif isinstance(other.typ, TypeVarValue) or other.typ is UNRESOLVED_VALUE:
                return {}
        elif isinstance(other, UnboundMethodValue):
            if self.typ in {Callable, collections.abc.Callable, object}:
                return {}
        return super().can_assign(other, ctx)

    def can_assign_thrift_enum(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if other is UNRESOLVED_VALUE:
            return {}
        elif isinstance(other, KnownValue):
            if not isinstance(other.val, int):
                return CanAssignError(f"{other} is not an int")
            if other.val in self.typ._VALUES_TO_NAMES:
                return {}
        elif isinstance(other, TypedValue):
            if other.type_object.is_assignable_to_type(
                self.typ
            ) or other.type_object.is_assignable_to_type(int):
                return {}
        elif isinstance(other, MultiValuedValue):
            tv_maps = []
            for val in other.vals:
                tv_map = self.can_assign(val, ctx)
                if isinstance(tv_map, CanAssignError):
                    # Adding an additional layer here isn't helpful
                    return tv_map
                tv_maps.append(tv_map)
            if not tv_maps:
                return CanAssignError(f"Cannot assign {other} to Thrift enum {self}")
            return unify_typevar_maps(tv_maps)
        elif isinstance(other, AnnotatedValue):
            return self.can_assign_thrift_enum(other.value, ctx)
        return CanAssignError(f"Cannot assign {other} to Thrift enum {self}")

    def get_generic_args_for_type(
        self, typ: Union[type, super], ctx: CanAssignContext
    ) -> Optional[List[Value]]:
        if isinstance(self, GenericValue):
            args = self.args
        else:
            args = ()
        generic_bases = ctx.get_generic_bases(self.typ, args)
        if typ in generic_bases:
            return list(generic_bases[typ].values())
        return None

    def get_generic_arg_for_type(
        self, typ: Union[type, super], ctx: CanAssignContext, index: int
    ) -> Value:
        args = self.get_generic_args_for_type(typ, ctx)
        if args and index < len(args):
            return args[index]
        return UNRESOLVED_VALUE

    def is_type(self, typ: type) -> bool:
        return self.type_object.is_assignable_to_type(typ)

    def get_type(self) -> type:
        return self.typ

    def get_type_value(self) -> Value:
        return KnownValue(self.typ)

    def __str__(self) -> str:
        return stringify_object(self.typ)


@dataclass(unsafe_hash=True, init=False)
class NewTypeValue(TypedValue):
    """A wrapper around an underlying type.

    Corresponds to ``typing.NewType``.

    This is a subclass of :class:`TypedValue`. Currently only NewTypes over simple,
    non-generic types are supported.

    """

    name: str
    """Name of the ``NewType``."""
    newtype: Any
    """Underlying ``NewType`` object."""

    def __init__(self, newtype: Any) -> None:
        super().__init__(newtype.__supertype__)
        self.name = newtype.__name__
        self.newtype = newtype

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, NewTypeValue):
            if self.newtype is other.newtype:
                return {}
            return CanAssignError(f"NewTypes {self} and {other} are not compatible")
        # Alow e.g. int for a NewType over int, but not a subtype of int such as an
        # IntEnum
        elif isinstance(other, TypedValue):
            if self.typ is not other.typ:
                return CanAssignError(f"Cannot assign {other} to {self}")
        elif isinstance(other, KnownValue):
            if self.typ is not type(other.val):
                return CanAssignError(f"Cannot assign {other} to {self}")
        return super().can_assign(other, ctx)

    def __str__(self) -> str:
        return "NewType(%r, %s)" % (self.name, stringify_object(self.typ))


@dataclass(unsafe_hash=True, init=False)
class GenericValue(TypedValue):
    """Subclass of :class:`TypedValue` that can represent generics.

    For example, ``List[int]`` is represented as ``GenericValue(list, [TypedValue(int)])``.

    """

    args: Tuple[Value, ...]
    """The generic arguments to the type."""

    def __init__(self, typ: type, args: Iterable[Value]) -> None:
        super().__init__(typ)
        self.args = tuple(args)

    def __str__(self) -> str:
        if self.typ is tuple:
            args = [*self.args, "..."]
        else:
            args = self.args
        args_str = ", ".join(str(arg) for arg in args)
        return f"{stringify_object(self.typ)}[{args_str}]"

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, TypedValue) and isinstance(other.typ, type):
            generic_args = other.get_generic_args_for_type(self.typ, ctx)
            # If we don't think it's a generic base, try super;
            # runtime isinstance() may disagree.
            if generic_args is None or len(self.args) != len(generic_args):
                return super().can_assign(other, ctx)
            tv_maps = []
            for i, (my_arg, their_arg) in enumerate(zip(self.args, generic_args)):
                tv_map = my_arg.can_assign(their_arg, ctx)
                if isinstance(tv_map, CanAssignError):
                    return CanAssignError(
                        f"In generic argument {i} to {self}", [tv_map]
                    )
                tv_maps.append(tv_map)
            if not tv_maps:
                return CanAssignError(f"Cannot assign {other} to {self}")
            return unify_typevar_maps(tv_maps)

        return super().can_assign(other, ctx)

    def get_arg(self, index: int) -> Value:
        try:
            return self.args[index]
        except IndexError:
            return UNRESOLVED_VALUE

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for arg in self.args:
            yield from arg.walk_values()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return GenericValue(
            self.typ, [arg.substitute_typevars(typevars) for arg in self.args]
        )


@dataclass(unsafe_hash=True, init=False)
class SequenceIncompleteValue(GenericValue):
    """A :class:`TypedValue` subclass representing a sequence of known type and length.

    For example, the expression ``[int(self.foo)]`` may be typed as
    ``SequenceIncompleteValue(list, [TypedValue(int)])``.

    This is only used for ``set``, ``list``, and ``tuple``.

    """

    members: Tuple[Value, ...]
    """The elements of the sequence."""

    def __init__(self, typ: type, members: Sequence[Value]) -> None:
        if members:
            args = (unite_values(*members),)
        else:
            args = (UNRESOLVED_VALUE,)
        super().__init__(typ, args)
        self.members = tuple(members)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, SequenceIncompleteValue):
            if not issubclass(other.typ, self.typ):
                return CanAssignError(
                    f"Cannot assign {stringify_object(other.typ)} to"
                    f" {stringify_object(self.typ)}"
                )
            my_len = len(self.members)
            their_len = len(other.members)
            if my_len != their_len:
                type_str = stringify_object(self.typ)
                return CanAssignError(
                    f"Cannot assign {type_str} of length {their_len} to {type_str} of"
                    f" length {my_len}"
                )
            if my_len == 0:
                return {}  # they're both empty
            tv_maps = []
            for i, (my_member, their_member) in enumerate(
                zip(self.members, other.members)
            ):
                tv_map = my_member.can_assign(their_member, ctx)
                if isinstance(tv_map, CanAssignError):
                    return CanAssignError(
                        f"Types for member {i} are incompatible", [tv_map]
                    )
                tv_maps.append(tv_map)
            return unify_typevar_maps(tv_maps)
        return super().can_assign(other, ctx)

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return SequenceIncompleteValue(
            self.typ, [member.substitute_typevars(typevars) for member in self.members]
        )

    def __str__(self) -> str:
        members = ", ".join(str(m) for m in self.members)
        if self.typ is tuple:
            return f"tuple[{members}]"
        return f"<{stringify_object(self.typ)} containing [{members}]>"

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for member in self.members:
            yield from member.walk_values()


@dataclass(unsafe_hash=True, init=False)
class DictIncompleteValue(GenericValue):
    """A :class:`TypedValue` representing a dictionary of known size.

    For example, the expression ``{'foo': int(self.bar)}`` may be typed as
    ``DictIncompleteValue([(KnownValue('foo'), TypedValue(int))])``.

    """

    items: List[Tuple[Value, Value]]
    """List of pairs representing the keys and values of the dict."""

    def __init__(self, items: List[Tuple[Value, Value]]) -> None:
        if items:
            key_type = unite_values(*[key for key, _ in items])
            value_type = unite_values(*[value for _, value in items])
        else:
            key_type = value_type = UNRESOLVED_VALUE
        super().__init__(dict, (key_type, value_type))
        self.items = items

    def __str__(self) -> str:
        items = ", ".join(f"{key}: {value}" for key, value in self.items)
        return f"<{stringify_object(self.typ)} containing {{{items}}}>"

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for key, value in self.items:
            yield from key.walk_values()
            yield from value.walk_values()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return DictIncompleteValue(
            [
                (key.substitute_typevars(typevars), value.substitute_typevars(typevars))
                for key, value in self.items
            ]
        )


@dataclass(init=False)
class TypedDictValue(GenericValue):
    """Equivalent to ``typing.TypedDict``; a dictionary with a known set of string keys.

    Currently does not handle the distinction between required and non-required keys.

    """

    items: Dict[str, Value]
    """The items of the ``TypedDict``."""

    def __init__(self, items: Dict[str, Value]) -> None:
        if items:
            value_type = unite_values(*items.values())
        else:
            value_type = UNRESOLVED_VALUE
        super().__init__(dict, (TypedValue(str), value_type))
        self.items = items

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, DictIncompleteValue):
            their_len = len(other.items)
            my_len = len(self.items)
            if my_len == 0:
                return {}
            if their_len < my_len:
                return CanAssignError(
                    f"Cannot assign dict of length {their_len} to dict of length"
                    f" {my_len}"
                )
            known_part = {
                key.val: value
                for key, value in other.items
                if isinstance(key, KnownValue) and isinstance(key.val, str)
            }
            has_unknowns = len(known_part) < len(other.items)
            tv_maps = []
            for key, value in self.items.items():
                if key not in known_part:
                    if not has_unknowns:
                        return CanAssignError(f"Key {key} is missing in {other}")
                else:
                    tv_map = value.can_assign(known_part[key], ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(f"Types for key {key} are incompatible")
                    tv_maps.append(tv_map)
            return unify_typevar_maps(tv_maps)
        elif isinstance(other, TypedDictValue):
            tv_maps = []
            for key, value in self.items.items():
                if key not in other.items:
                    return CanAssignError(f"Key {key} is missing in {other}")
                tv_map = value.can_assign(other.items[key], ctx)
                if isinstance(tv_map, CanAssignError):
                    return CanAssignError(f"Types for key {key} are incompatible")
                tv_maps.append(tv_map)
            return unify_typevar_maps(tv_maps)
        elif isinstance(other, KnownValue) and isinstance(other.val, dict):
            tv_maps = []
            for key, value in self.items.items():
                if key not in other.val:
                    return CanAssignError(f"Key {key} is missing in {other}")
                tv_map = value.can_assign(KnownValue(other.val[key]), ctx)
                if isinstance(tv_map, CanAssignError):
                    return CanAssignError(f"Types for key {key} are incompatible")
                tv_maps.append(tv_map)
            return unify_typevar_maps(tv_maps)
        return super().can_assign(other, ctx)

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return TypedDictValue(
            {
                key: value.substitute_typevars(typevars)
                for key, value in self.items.items()
            }
        )

    def __str__(self) -> str:
        items = [f'"{key}": {value}' for key, value in self.items.items()]
        return "TypedDict({%s})" % ", ".join(items)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items)))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for value in self.items.values():
            yield from value.walk_values()


@dataclass(unsafe_hash=True, init=False)
class AsyncTaskIncompleteValue(GenericValue):
    """A :class:`GenericValue` representing an async task.

    This should probably just be replaced with ``GenericValue``.

    """

    value: Value
    """The value returned by the task on completion."""

    def __init__(self, typ: type, value: Value) -> None:
        super().__init__(typ, (value,))
        self.value = value

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return AsyncTaskIncompleteValue(
            self.typ, self.value.substitute_typevars(typevars)
        )

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.value.walk_values()


@dataclass(unsafe_hash=True, init=False)
class CallableValue(TypedValue):
    """Equivalent to the ``Callable`` type.

    This is a thin wrapper around :class:`pyanalyze.signature.Signature`.

    """

    signature: "pyanalyze.signature.Signature"

    def __init__(self, signature: "pyanalyze.signature.Signature") -> None:
        super().__init__(collections.abc.Callable)
        self.signature = signature

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return CallableValue(self.signature.substitute_typevars(typevars))

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.signature.walk_values()

    def get_asynq_value(self) -> Value:
        """Return the CallableValue for the .asynq attribute of an AsynqCallable."""
        sig = self.signature.get_asynq_value()
        return CallableValue(sig)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if not isinstance(other, MultiValuedValue):
            signature = ctx.signature_from_value(other)
            if signature is None:
                return CanAssignError(f"{other} is not a callable type")
            if isinstance(signature, pyanalyze.signature.Signature):
                return self.signature.can_assign(signature, ctx)

        return super().can_assign(other, ctx)

    def __str__(self) -> str:
        is_asynq = "Asynq" if self.signature.is_asynq else ""
        return_value = self.signature.signature.return_annotation
        if self.signature.is_ellipsis_args:
            args = "..."
        else:
            parts = ["["]
            added_star = False
            for i, param in enumerate(self.signature.signature.parameters.values()):
                if i > 0:
                    parts.append(", ")
                if param.kind is pyanalyze.signature.SigParameter.POSITIONAL_ONLY:
                    parts.append(str(param.get_annotation()))
                elif (
                    param.kind is pyanalyze.signature.SigParameter.POSITIONAL_OR_KEYWORD
                ):
                    parts.append(f"{param.name}: {param.get_annotation()}")
                elif param.kind is pyanalyze.signature.SigParameter.KEYWORD_ONLY:
                    if not added_star:
                        parts.append("*, ")
                        added_star = True
                    parts.append(f"{param.name}: {param.get_annotation()}")
                elif param.kind is pyanalyze.signature.SigParameter.VAR_POSITIONAL:
                    added_star = True
                    parts.append(f"*{param.get_annotation()}")
                elif param.kind is pyanalyze.signature.SigParameter.VAR_KEYWORD:
                    parts.append(f"**{param.get_annotation()}")
                if param.default is not pyanalyze.signature.EMPTY:
                    parts.append(" = ...")
            parts.append("]")
            args = "".join(parts)
        return f"{is_asynq}Callable[{args}, {return_value}]"


@dataclass(frozen=True)
class SubclassValue(Value):
    """Equivalent of ``Type[]``.

    The `typ` attribute can be either a :class:`TypedValue` or a
    :class:`TypeVarValue`. The former is equivalent to ``Type[int]``
    and represents the ``int`` class or a subclass. The latter is
    equivalent to ``Type[T]`` where ``T`` is a type variable.
    The third legal argument to ``Type[]`` is ``Any``, but
    ``Type[Any]`` is represented as ``TypedValue(type)``.

    """

    typ: Union[TypedValue, "TypeVarValue"]
    """The underlying type."""

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return self.make(self.typ.substitute_typevars(typevars))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        yield from self.typ.walk_values()

    def is_type(self, typ: type) -> bool:
        if isinstance(self.typ, TypedValue):
            return safe_issubclass(self.typ.typ, typ)
        return False

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, SubclassValue):
            return self.typ.can_assign(other.typ, ctx)
        elif isinstance(other, KnownValue):
            if isinstance(other.val, type):
                if isinstance(self.typ, TypedValue) and issubclass(
                    other.val, self.typ.typ
                ):
                    return {}
                elif isinstance(self.typ, TypeVarValue):
                    return {self.typ.typevar: TypedValue(other.val)}
        elif isinstance(other, TypedValue):
            if other.typ is type:
                return {}
            # metaclass
            elif issubclass(other.typ, type) and (
                (
                    isinstance(self.typ, TypedValue)
                    and isinstance(self.typ.typ, other.typ)
                )
                or isinstance(self.typ, (TypeVarValue))
            ):
                return {}
        return super().can_assign(other, ctx)

    def get_type(self) -> Optional[type]:
        if isinstance(self.typ, TypedValue):
            return type(self.typ.typ)
        else:
            return None

    def get_type_value(self) -> Value:
        typ = self.get_type()
        if typ is not None:
            return KnownValue(typ)
        else:
            return UNRESOLVED_VALUE

    def __str__(self) -> str:
        return f"Type[{self.typ}]"

    @classmethod
    def make(cls, origin: Value) -> Value:
        if isinstance(origin, MultiValuedValue):
            return unite_values(*[cls.make(val) for val in origin.vals])
        elif origin is UNRESOLVED_VALUE:
            # Type[Any] is equivalent to plain type
            return TypedValue(type)
        elif isinstance(origin, (TypeVarValue, TypedValue)):
            return cls(origin)
        else:
            return UNRESOLVED_VALUE


@dataclass(frozen=True, order=False)
class MultiValuedValue(Value):
    """Equivalent of ``typing.Union``. Represents the union of multiple values."""

    raw_vals: InitVar[Iterable[Value]]
    vals: Tuple[Value, ...] = field(init=False)
    """The underlying values of the union."""

    def __post_init__(self, raw_vals: Iterable[Value]) -> None:
        object.__setattr__(
            self,
            "vals",
            tuple(chain.from_iterable(flatten_values(val) for val in raw_vals)),
        )

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        if not self.vals:
            return self
        return MultiValuedValue(
            [val.substitute_typevars(typevars) for val in self.vals]
        )

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, MultiValuedValue):
            tv_maps = []
            for val in other.vals:
                tv_map = self.can_assign(val, ctx)
                if isinstance(tv_map, CanAssignError):
                    # Adding an additional layer here isn't helpful
                    return tv_map
                tv_maps.append(tv_map)
            if not tv_maps:
                return CanAssignError(f"Cannot assign {other} to {self}")
            return unify_typevar_maps(tv_maps)
        elif isinstance(other, AnnotatedValue):
            return self.can_assign(other.value, ctx)
        else:
            my_vals = self.vals
            # Optimization for large unions of literals. We could perhaps cache this set,
            # but that's more complicated. Empirically this is already much faster.
            # The number 20 is arbitrary. I noticed the bottleneck in production on a
            # Union with nearly 500 values.
            if isinstance(other, KnownValue) and len(my_vals) > 20:
                try:
                    # Include the type to avoid e.g. 1 and True matching
                    known_values = {
                        (subval.val, type(subval.val))
                        for subval in my_vals
                        if isinstance(subval, KnownValue)
                    }
                except TypeError:
                    pass  # not hashable
                else:
                    try:
                        is_present = (other.val, type(other.val)) in known_values
                    except TypeError:
                        pass  # not hashable
                    else:
                        if is_present:
                            return {}
                        else:
                            # Make remaining check not consider the KnownValues again
                            my_vals = [
                                subval
                                for subval in my_vals
                                if not isinstance(subval, KnownValue)
                            ]

            tv_maps = []
            errors = []
            for val in my_vals:
                tv_map = val.can_assign(other, ctx)
                if isinstance(tv_map, CanAssignError):
                    errors.append(tv_map)
                else:
                    tv_maps.append(tv_map)
            # Ignore any branches that don't match
            if not tv_maps:
                return CanAssignError("Cannot assign to Union", errors)
            # Include only typevars that appear in all branches; i.e., prefer
            # branches that don't set typevars.
            typevars = collections.Counter(tv for tv_map in tv_maps for tv in tv_map)
            num_tv_maps = len(tv_maps)
            return {
                tv: unite_values(*[tv_map[tv] for tv_map in tv_maps])
                for tv, count in typevars.items()
                if count == num_tv_maps
            }

    def get_type_value(self) -> Value:
        if not self.vals:
            return self
        return MultiValuedValue([val.get_type_value() for val in self.vals])

    def __eq__(self, other: Value) -> Union[bool, Literal[NotImplemented]]:
        if not isinstance(other, MultiValuedValue):
            return NotImplemented
        if self.vals == other.vals:
            return True
        # try to put the values in a set so different objects that happen to have different order
        # compare equal, but don't worry if some aren't hashable
        try:
            left_vals = set(self.vals)
            right_vals = set(other.vals)
        except Exception:
            return False
        return left_vals == right_vals

    def __ne__(self, other: Value) -> bool:
        return not (self == other)

    def __str__(self) -> str:
        if not self.vals:
            return "NoReturn"
        return "Union[%s]" % ", ".join(map(str, self.vals))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for val in self.vals:
            yield from val.walk_values()


NO_RETURN_VALUE = MultiValuedValue([])
"""The empty union, equivalent to ``typing.NoReturn``."""


@dataclass(frozen=True)
class ReferencingValue(Value):
    """Value that is a reference to another value (used to implement globals)."""

    scope: Any
    name: str

    def __str__(self) -> str:
        return f"<reference to {self.name}>"


@dataclass(frozen=True)
class TypeVarValue(Value):
    """Value representing a ``typing.TypeVar``.

    Currently, bounds, value restrictions, and variance are ignored.

    """

    typevar: TypeVar

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return typevars.get(self.typevar, self)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        return {self.typevar: other}

    def get_fallback_value(self) -> Value:
        # TODO: support bounds and bases here to do something smarter
        return UNRESOLVED_VALUE


class Extension:
    """An extra piece of information about a type that can be stored in
    an :class:`AnnotatedValue`."""

    __slots__ = ()

    def substitute_typevars(self, typevars: TypeVarMap) -> "Extension":
        return self

    def walk_values(self) -> Iterable[Value]:
        return []


@dataclass
class ParameterTypeGuardExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that the parameter named `varname` is of type `guarded_type`.

    Corresponds to :class:`pyanalyze.extensions.ParameterTypeGuard`.

    """

    varname: str
    guarded_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        guarded_type = self.guarded_type.substitute_typevars(typevars)
        return ParameterTypeGuardExtension(self.varname, guarded_type)

    def walk_values(self) -> Iterable[Value]:
        yield from self.guarded_type.walk_values()


@dataclass
class TypeGuardExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that the first function argument is of type `guarded_type`.

    Corresponds to :class:`pyanalyze.extensions.TypeGuard`, or ``typing.TypeGuard``.

    """

    guarded_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        guarded_type = self.guarded_type.substitute_typevars(typevars)
        return TypeGuardExtension(guarded_type)

    def walk_values(self) -> Iterable[Value]:
        yield from self.guarded_type.walk_values()


@dataclass
class HasAttrGuardExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that the function argument named `varname` has an attribute
    named `attribute_name` of type `attribute_type`.

    Corresponds to :class:`pyanalyze.extensions.HasAttrGuard`.

    """

    varname: str
    attribute_name: Value
    attribute_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        return HasAttrGuardExtension(
            self.varname,
            self.attribute_name.substitute_typevars(typevars),
            self.attribute_type.substitute_typevars(typevars),
        )

    def walk_values(self) -> Iterable[Value]:
        yield from self.attribute_name.walk_values()
        yield from self.attribute_type.walk_values()


@dataclass
class HasAttrExtension(Extension):
    """Attached to an object to indicate that it has the given attribute.

    These cannot be created directly from user code, only through the
    :class:`pyanalyze.extension.HasAttrGuard` mechanism. This is
    because of potential code like this::

        def f(x: Annotated[object, HasAttr["y", int]]) -> None:
            return x.y

    Here, we would correctly type check the function body, but we currently
    have no way to enforce that the function is only called with arguments that
    obey the constraint.

    """

    attribute_name: Value
    attribute_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        return HasAttrExtension(
            self.attribute_name.substitute_typevars(typevars),
            self.attribute_type.substitute_typevars(typevars),
        )

    def walk_values(self) -> Iterable[Value]:
        yield from self.attribute_name.walk_values()
        yield from self.attribute_type.walk_values()


@dataclass(frozen=True)
class WeakExtension(Extension):
    """Used to indicate that a generic argument to a container may be widened.

    This is used only in conjuction with the special casing for functions
    like ``list.extend``. After code like ``lst = [1, 2]; lst.extend([i for in range(5)])``
    we may end up inferring a type like ``List[Literal[0, 1, 2, 3, 4]]``, but that is
    too narrow and leads to false positives if later code puts a different int in
    the list. If the generic argument is instead annotated with ``WeakExtension``, we
    widen the type to accommodate later appends.

    The ``TestGenericMutators.test_weak_value`` test case is an example.

    """


@dataclass(frozen=True)
class AnnotatedValue(Value):
    """Value representing a `PEP 593 <https://www.python.org/dev/peps/pep-0593/>`_ Annotated object.

    Pyanalyze uses ``Annotated`` types to represent types with some extra
    information added to them in the form of :class:`Extension` objects.

    """

    value: Value
    """The underlying value."""
    metadata: Sequence[Union[Value, Extension]]
    """The extensions associated with this value."""

    def is_type(self, typ: type) -> bool:
        return self.value.is_type(typ)

    def get_type(self) -> Optional[type]:
        return self.value.get_type()

    def get_type_value(self) -> Value:
        return self.value.get_type_value()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        metadata = [val.substitute_typevars(typevars) for val in self.metadata]
        return AnnotatedValue(self.value.substitute_typevars(typevars), metadata)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        return self.value.can_assign(other, ctx)

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.value.walk_values()
        for val in self.metadata:
            yield from val.walk_values()

    def get_metadata_of_type(self, typ: Type[T]) -> Iterable[T]:
        """Return any metadata of the given type."""
        for data in self.metadata:
            if isinstance(data, typ):
                yield data

    def has_metadata_of_type(self, typ: Type[Extension]) -> bool:
        """Return whether there is metadat of the given type."""
        return any(isinstance(data, typ) for data in self.metadata)

    def __str__(self) -> str:
        return f"Annotated[{self.value}, {', '.join(map(str, self.metadata))}]"


@dataclass(frozen=True)
class VariableNameValue(Value):
    """Value that is stored in a variable associated with a particular kind of value.

    For example, any variable named `uid` will get resolved into a ``VariableNameValue``
    of type `uid`,
    and if it gets passed into a function that takes an argument called `aid`,
    the call will be rejected.

    This was created for a legacy codebase without type annotations. If possible, prefer
    using NewTypes or other more explicit types.

    There should only be a limited set of ``VariableNameValue`` objects,
    created through the pyanalyze configuration.

    """

    varnames: List[str]

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if not isinstance(other, VariableNameValue):
            return {}
        if other == self:
            return {}
        return CanAssignError(f"Types {self} and {other} are different")

    def __str__(self) -> str:
        return "<variable name: %s>" % ", ".join(self.varnames)

    @classmethod
    def from_varname(
        cls, varname: str, varname_map: Dict[str, "VariableNameValue"]
    ) -> Optional["VariableNameValue"]:
        """Returns the VariableNameValue corresponding to a variable name.

        If there is no VariableNameValue that corresponds to the variable name, returns None.

        """
        if varname in varname_map:
            return varname_map[varname]
        if "_" in varname:
            parts = varname.split("_")
            if parts[-1] == "id":
                shortened_varname = "_".join(parts[-2:])
            else:
                shortened_varname = parts[-1]
            return varname_map.get(shortened_varname)
        return None


def flatten_values(val: Value, *, unwrap_annotated: bool = False) -> Iterable[Value]:
    """Flatten a :class:`MultiValuedValue` into its constituent values.

    We don't need to do this recursively because the
    :class:`MultiValuedValue` constructor applies this to its arguments.

    if `unwrap_annotated` is true, produces the underlying values for
    :class:`AnnotatedValue` objects.

    """
    if isinstance(val, MultiValuedValue):
        yield from val.vals
    elif isinstance(val, AnnotatedValue) and isinstance(val.value, MultiValuedValue):
        yield from val.value.vals
    elif unwrap_annotated and isinstance(val, AnnotatedValue):
        yield val.value
    else:
        yield val


def unify_typevar_maps(tv_maps: Sequence[TypeVarMap]) -> TypeVarMap:
    raw_map = defaultdict(list)
    for tv_map in tv_maps:
        for tv, value in tv_map.items():
            raw_map[tv].append(value)
    return {tv: unite_values(*values) for tv, values in raw_map.items()}


def make_weak(val: Value) -> Value:
    return annotate_value(val, [WeakExtension()])


def annotate_value(origin: Value, metadata: Sequence[Union[Value, Extension]]) -> Value:
    if not metadata:
        return origin
    if isinstance(origin, AnnotatedValue):
        # Flatten it
        metadata = [*origin.metadata, *metadata]
        origin = origin.value
    # Make sure order is consistent; conceptually this is a set but
    # sets have unpredictable iteration order.
    hashable_vals = OrderedDict()
    unhashable_vals = []
    uncomparable_vals = []
    for item in metadata:
        try:
            # Don't readd it to preserve original ordering.
            if item not in hashable_vals:
                hashable_vals[item] = None
        except Exception:
            try:
                if item not in unhashable_vals:
                    unhashable_vals.append(item)
            except Exception:
                uncomparable_vals.append(item)
    metadata = list(hashable_vals) + unhashable_vals + uncomparable_vals
    return AnnotatedValue(origin, metadata)


def unite_values(*values: Value) -> Value:
    """Unite multiple values into a single :class:`Value`.

    This collapses equal values and returns a :class:`MultiValuedValue`
    if multiple remain.

    """
    if not values:
        return NO_RETURN_VALUE
    # Make sure order is consistent; conceptually this is a set but
    # sets have unpredictable iteration order.
    hashable_vals = OrderedDict()
    unhashable_vals = []
    uncomparable_vals = []
    for value in values:
        if isinstance(value, MultiValuedValue):
            subvals = value.vals
        elif isinstance(value, AnnotatedValue) and isinstance(
            value.value, MultiValuedValue
        ):
            subvals = value.value.vals
        else:
            subvals = [value]
        for subval in subvals:
            try:
                # Don't readd it to preserve original ordering.
                if subval not in hashable_vals:
                    hashable_vals[subval] = None
            except Exception:
                try:
                    if subval not in unhashable_vals:
                        unhashable_vals.append(subval)
                except Exception:
                    uncomparable_vals.append(subval)
    existing = list(hashable_vals) + unhashable_vals + uncomparable_vals
    num = len(existing)
    if num == 0:
        return NO_RETURN_VALUE
    if num == 1:
        return existing[0]
    else:
        return MultiValuedValue(existing)


def boolean_value(value: Optional[Value]) -> Optional[bool]:
    """Given a Value, returns whether the object is statically known to be truthy.

    Returns None if its truth value cannot be determined.

    """
    if isinstance(value, KnownValue):
        try:
            # don't pretend to know the boolean value of mutable types
            # since we may have missed a change
            if not isinstance(value.val, KNOWN_MUTABLE_TYPES):
                return bool(value.val)
        except Exception:
            # Its __bool__ threw an exception. Just give up.
            return None
    return None


T = TypeVar("T")
IterableValue = GenericValue(collections.abc.Iterable, [TypeVarValue(T)])


def concrete_values_from_iterable(
    value: Value, ctx: CanAssignContext
) -> Union[None, Value, Sequence[Value]]:
    """Return the exact values that can be extracted from an iterable.

    Three possible return types:

    - ``None`` if the argument is not iterable
    - A sequence of :class:`Value` if we know the exact types in the iterable
    - A single :class:`Value` if we just know that the iterable contains this
      value, but not the precise number of them.

    Examples:

    - ``int`` -> ``None``
    - ``tuple[int, str]`` -> ``(int, str)``
    - ``tuple[int, ...]`` -> ``int``

    """
    if isinstance(value, MultiValuedValue):
        subvals = [concrete_values_from_iterable(val, ctx) for val in value.vals]
        if any(subval is None for subval in subvals):
            return None
        value_subvals = [subval for subval in subvals if isinstance(subval, Value)]
        seq_subvals = [
            subval
            for subval in subvals
            if subval is not None and not isinstance(subval, Value)
        ]
        if not value_subvals and len(set(map(len, seq_subvals))) == 1:
            return [unite_values(*vals) for vals in zip(*seq_subvals)]
        return unite_values(*value_subvals, *chain.from_iterable(seq_subvals))
    elif isinstance(value, AnnotatedValue):
        return concrete_values_from_iterable(value.value, ctx)
    value = replace_known_sequence_value(value)
    if isinstance(value, SequenceIncompleteValue) and value.typ is tuple:
        return value.members
    tv_map = IterableValue.can_assign(value, ctx)
    if not isinstance(tv_map, CanAssignError):
        return tv_map.get(T, UNRESOLVED_VALUE)
    return None


def unpack_values(
    value: Value,
    ctx: CanAssignContext,
    target_length: int,
    post_starred_length: Optional[int] = None,
) -> Union[Sequence[Value], CanAssignError]:
    """Implement iterable unpacking.

    If `post_starred_length` is None, return a list of `target_length`
    values, or :class:`CanAssignError` if value is not an iterable of
    the expected length. If `post_starred_length` is not None,
    return a list of `target_length + 1 + post_starred_length` values. This implements
    unpacking like ``a, b, *c, d = ...``.

    """
    if isinstance(value, MultiValuedValue):
        subvals = [
            unpack_values(val, ctx, target_length, post_starred_length)
            for val in value.vals
        ]
        good_subvals = []
        for subval in subvals:
            if isinstance(subval, CanAssignError):
                return CanAssignError(f"Cannot unpack {value}", [subval])
            good_subvals.append(subval)
        if not good_subvals:
            return _create_unpacked_list(
                UNRESOLVED_VALUE, target_length, post_starred_length
            )
        return [unite_values(*vals) for vals in zip(*good_subvals)]
    elif isinstance(value, AnnotatedValue):
        return unpack_values(value.value, ctx, target_length, post_starred_length)
    value = replace_known_sequence_value(value)

    # We treat the different sequence types differently here.
    # - Tuples are  immutable so we can always unpack and show
    #   an error if the length doesn't match.
    # - Sets have randomized order so unpacking into specific values
    #   doesn't make sense. We just fallback to the behavior for
    #   general iterables.
    # - Dicts do have deterministic order but unpacking them doesn't
    #   seem like a common use case. They're also mutable, so if we
    #   did decide to unpack, we'd have to do something similar to
    #   what we do for lists.
    # - Lists can be sensibly unpacked but they are also mutable. Therefore,
    #   we try first to unpack into specific values, and if that doesn't
    #   work due to a length mismatch we fall back to the generic
    #   iterable approach. We experimented both with treating lists
    #   like tuples and with always falling back, and both approaches
    #   led to false positives.
    if isinstance(value, SequenceIncompleteValue):
        if value.typ is tuple:
            return _unpack_value_sequence(
                value, value.members, target_length, post_starred_length
            )
        elif value.typ is list:
            vals = _unpack_value_sequence(
                value, value.members, target_length, post_starred_length
            )
            if not isinstance(vals, CanAssignError):
                return vals

    tv_map = IterableValue.can_assign(value, ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map
    iterable_type = tv_map.get(T, UNRESOLVED_VALUE)
    return _create_unpacked_list(iterable_type, target_length, post_starred_length)


def _create_unpacked_list(
    iterable_type: Value, target_length: int, post_starred_length: Optional[int]
) -> List[Value]:
    if post_starred_length is not None:
        return [
            *([iterable_type] * target_length),
            GenericValue(list, [iterable_type]),
            *([iterable_type] * post_starred_length),
        ]
    else:
        return [iterable_type] * target_length


def _unpack_value_sequence(
    value: Value,
    members: Sequence[Value],
    target_length: int,
    post_starred_length: Optional[int],
) -> Union[Sequence[Value], CanAssignError]:
    actual_length = len(members)
    if post_starred_length is None:
        if actual_length != target_length:
            return CanAssignError(
                f"{value} is of length {actual_length} (expected {target_length})"
            )
        return members
    if actual_length < target_length + post_starred_length:
        return CanAssignError(
            f"{value} is of length {actual_length} (expected at least"
            f" {target_length + post_starred_length})"
        )
    head = members[:target_length]
    if post_starred_length > 0:
        body = SequenceIncompleteValue(
            list, members[target_length:-post_starred_length]
        )
        tail = members[-post_starred_length:]
    else:
        body = SequenceIncompleteValue(list, members[target_length:])
        tail = []
    return [*head, body, *tail]


def replace_known_sequence_value(value: Value) -> Value:
    if isinstance(value, AnnotatedValue):
        value = value.value
    if isinstance(value, KnownValue):
        if isinstance(value.val, (list, tuple, set)):
            return SequenceIncompleteValue(
                type(value.val), [KnownValue(elt) for elt in value.val]
            )
        elif isinstance(value.val, dict):
            return DictIncompleteValue(
                [(KnownValue(k), KnownValue(v)) for k, v in value.val.items()]
            )
    return value


def extract_typevars(value: Value) -> Iterable["TypeVar"]:
    for val in value.walk_values():
        if isinstance(val, TypeVarValue):
            yield val.typevar


def stringify_object(obj: Any) -> str:
    # Stringify arbitrary Python objects such as methods and types.
    try:
        objclass = getattr(obj, "__objclass__", None)
        if objclass is not None:
            return f"{stringify_object(objclass)}.{obj.__name__}"
        if obj.__module__ == BUILTIN_MODULE:
            return obj.__name__
        elif hasattr(obj, "__qualname__"):
            return f"{obj.__module__}.{obj.__qualname__}"
        else:
            return f"{obj.__module__}.{obj.__name__}"
    except Exception:
        return repr(obj)
