"""

Value classes represent the value of an expression in a Python program. Values
are the key data type used in pyanalyze's internals.

Values are instances of a subclass of :class:`Value`. This module defines
these subclasses and some related utilities.

:func:`dump_value` can be used to show inferred values during type checking. Examples::

    from typing import Any
    from pyanalyze import dump_value

    def function(x: int, y: list[int], z: Any):
        dump_value(1)  # Literal[1]
        dump_value(x)  # int
        dump_value(y)  # list[int]
        dump_value(z)  # Any

"""

import collections.abc
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field, InitVar
import enum
import inspect
from itertools import chain
from types import FunctionType
import qcore
import textwrap
from typing import (
    Any,
    Callable,
    ContextManager,
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
    cast,
)
from typing_extensions import Literal, Protocol

import pyanalyze
from pyanalyze.extensions import CustomCheck

from .safe import all_of_type, safe_issubclass

T = TypeVar("T")
# __builtin__ in Python 2 and builtins in Python 3
BUILTIN_MODULE = str.__module__
KNOWN_MUTABLE_TYPES = (list, set, dict, deque)

TypeVarMap = Mapping["TypeVar", "Value"]
GenericBases = Mapping[Union[type, str], TypeVarMap]


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
        if isinstance(other, AnyValue):
            ctx.record_any_used()
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
            return other.can_be_assigned(self, ctx)
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

    def simplify(self) -> "Value":
        """Simplify this Value to reduce excessive detail."""
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

    def make_type_object(
        self, typ: Union[type, super, str]
    ) -> "pyanalyze.type_object.TypeObject":
        """Return a :class:`pyanalyze.type_object.TypeObject` for this concrete type."""
        raise NotImplementedError

    def get_generic_bases(
        self, typ: Union[type, str], generic_args: Sequence["Value"] = ()
    ) -> GenericBases:
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

    def get_signature(
        self, obj: object
    ) -> Optional["pyanalyze.signature.ConcreteSignature"]:
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

    def get_attribute_from_value(self, root_value: "Value", attribute: str) -> "Value":
        return UNINITIALIZED_VALUE

    def can_assume_compatibility(
        self,
        left: "pyanalyze.type_object.TypeObject",
        right: "pyanalyze.type_object.TypeObject",
    ) -> bool:
        return False

    def assume_compatibility(
        self,
        left: "pyanalyze.type_object.TypeObject",
        right: "pyanalyze.type_object.TypeObject",
    ) -> ContextManager[None]:
        return qcore.empty_context

    def has_used_any_match(self) -> bool:
        """Whether Any was used to secure a match."""
        return False

    def record_any_used(self) -> None:
        """Record that Any was used to secure a match."""

    def reset_any_used(self) -> ContextManager[None]:
        """Context that resets the value used by :meth:`has_used_any_match` and
        :meth:`record_any_match`."""
        return qcore.empty_context


@dataclass(frozen=True)
class CanAssignError:
    """A type checking error message with nested details.

    This exists in order to produce more useful error messages
    when there is a mismatch between complex types.

    """

    message: str = ""
    children: List["CanAssignError"] = field(default_factory=list)

    def display(self, depth: int = 2) -> str:
        """Display all errors in a human-readable format."""
        child_result = "".join(
            child.display(depth=depth + 2) for child in self.children
        )
        if self.message:
            message = textwrap.indent(self.message, " " * depth)
            return f"{message}\n{child_result}"
        else:
            return child_result

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


class AnySource(enum.Enum):
    """Sources of Any values."""

    default = 1
    """Any that has not been categorized."""
    explicit = 2
    """The user wrote 'Any' in an annotation."""
    error = 3
    """An error occurred."""
    unreachable = 4
    """Value that is inferred to be unreachable."""
    inference = 5
    """Insufficiently powerful type inference."""
    unannotated = 6
    """Unannotated code."""
    variable_name = 7
    """A :class:`VariableNameValue`."""
    from_another = 8
    """An Any derived from another Any, for example as an attribute."""
    generic_argument = 9
    """Missing type argument to a generic class."""
    marker = 10
    """Marker object used internally."""
    incomplete_annotation = 11
    """A special form like ClassVar without a type argument."""
    multiple_overload_matches = 12
    """Multiple matching overloads."""


@dataclass(frozen=True)
class AnyValue(Value):
    """An unknown value, equivalent to ``typing.Any``."""

    source: AnySource
    """The source of this value, such as a user-defined annotation
    or a previous error."""

    def __str__(self) -> str:
        if self.source is AnySource.default:
            return "Any"
        return f"Any[{self.source.name}]"

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        return {}  # Always allowed


UNRESOLVED_VALUE = AnyValue(AnySource.default)
"""The default instance of :class:`AnyValue`.

In the future, this should be replaced with instances of
`AnyValue` with a specific source.

"""


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

    def get_type_object(
        self, ctx: Optional[CanAssignContext] = None
    ) -> "pyanalyze.type_object.TypeObject":
        if ctx is not None:
            return ctx.make_type_object(type(self.val))
        return pyanalyze.type_object.TypeObject(type(self.val))

    def get_type_value(self) -> Value:
        return KnownValue(type(self.val))

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        # Make Literal[function] equivalent to a Callable type
        if isinstance(self.val, FunctionType):
            signature = ctx.get_signature(self.val)
            if signature is not None:
                return CallableValue(signature).can_assign(other, ctx)
        if isinstance(other, KnownValue):
            if self.val is other.val:
                return {}
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
        try:
            return hash((type(self.val), self.val))
        except TypeError:
            # If the value is not directly hashable, hash it by identity instead. This breaks
            # the rule that x == y should imply hash(x) == hash(y), but hopefully that will
            # be fine.
            return hash((type(self.val), id(self.val)))

    def __str__(self) -> str:
        if self.val is None:
            return "None"
        else:
            return "Literal[%r]" % (self.val,)

    def substitute_typevars(self, typevars: TypeVarMap) -> "KnownValue":
        if not typevars or not callable(self.val):
            return self
        return KnownValueWithTypeVars(self.val, typevars)

    def simplify(self) -> Value:
        val = replace_known_sequence_value(self)
        if isinstance(val, KnownValue):
            return TypedValue(type(val.val))
        return val.simplify()


@dataclass(frozen=True)
class KnownValueWithTypeVars(KnownValue):
    """Subclass of KnownValue that records a TypeVar substitution."""

    typevars: TypeVarMap = field(compare=False)
    """TypeVars substituted on this value."""


@dataclass(frozen=True)
class UnboundMethodValue(Value):
    """Value that represents a method on an underlying :class:`Value`.

    Despite the name this really represents a method bound to a value. For
    example, given ``s: str``, ``s.strip`` will be inferred as
    ``UnboundMethodValue("strip", Composite(TypedValue(str), "s"))``.

    """

    attr_name: str
    """Name of the method."""
    composite: "pyanalyze.stacked_scopes.Composite"
    """Value the method is bound to."""
    secondary_attr_name: Optional[str] = None
    """Used when an attribute is accessed on an existing ``UnboundMethodValue``.

    This is mostly useful in conjunction with asynq, where we might use
    ``object.method.asynq``. In that case, we would infer an ``UnboundMethodValue``
    with `secondary_attr_name` set to ``"asynq"``.

    """
    typevars: Optional[TypeVarMap] = field(default=None, compare=False)
    """Extra TypeVars applied to this method."""

    def get_method(self) -> Optional[Any]:
        """Return the runtime callable for this ``UnboundMethodValue``, or
        None if it cannot be found."""
        try:
            typ = self.composite.value.get_type()
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

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        signature = self.get_signature(ctx)
        if signature is None:
            return {}
        return CallableValue(signature).can_assign(other, ctx)

    def get_signature(
        self, ctx: CanAssignContext
    ) -> Optional["pyanalyze.signature.ConcreteSignature"]:
        signature = ctx.signature_from_value(self)
        if signature is None:
            return None
        if isinstance(signature, pyanalyze.signature.BoundMethodSignature):
            signature = signature.get_signature()
        if isinstance(signature, pyanalyze.signature.PropertyArgSpec):
            return None
        return signature

    def substitute_typevars(self, typevars: TypeVarMap) -> "Value":
        return UnboundMethodValue(
            self.attr_name,
            self.composite.substitute_typevars(typevars),
            self.secondary_attr_name,
            typevars=typevars
            if self.typevars is None
            else {**self.typevars, **typevars},
        )

    def __str__(self) -> str:
        return "<method %s%s on %s>" % (
            self.attr_name,
            f".{self.secondary_attr_name}" if self.secondary_attr_name else "",
            self.composite.value,
        )


@dataclass(unsafe_hash=True)
class TypedValue(Value):
    """Value for which we know the type. This is equivalent to simple type
    annotations: an annotation of ``int`` will yield ``TypedValue(int)`` during
    type inference.

    """

    typ: Union[type, str]
    """The underlying type, or a fully qualified reference to one."""
    _type_object: Optional["pyanalyze.type_object.TypeObject"] = field(
        init=False, repr=False, hash=False, compare=False, default=None
    )

    def get_type_object(
        self, ctx: Optional[CanAssignContext] = None
    ) -> "pyanalyze.type_object.TypeObject":
        if self._type_object is None:
            if ctx is None:
                # TODO: remoove this behavior and make ctx required
                return pyanalyze.type_object.TypeObject(self.typ)
            self._type_object = ctx.make_type_object(self.typ)
        return self._type_object

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        self_tobj = self.get_type_object(ctx)
        if self_tobj.is_thrift_enum:
            # Special case: Thrift enums. These are conceptually like
            # enums, but they are ints at runtime.
            return self.can_assign_thrift_enum(other, ctx)
        elif isinstance(other, KnownValue):
            if self_tobj.is_instance(other.val):
                return {}
            return self_tobj.can_assign(self, other, ctx)
        elif isinstance(other, TypedValue):
            return self_tobj.can_assign(self, other, ctx)
        elif isinstance(other, SubclassValue):
            if (
                isinstance(other.typ, TypedValue)
                and isinstance(self.typ, type)
                and isinstance(other.typ.typ, self.typ)
            ):
                return {}
            elif isinstance(other.typ, (TypeVarValue, AnyValue)):
                return {}
        elif isinstance(other, UnboundMethodValue):
            if self_tobj.is_exactly(
                {cast(type, Callable), collections.abc.Callable, object}
            ):
                return {}
        return super().can_assign(other, ctx)

    def can_assign_thrift_enum(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, AnyValue):
            ctx.record_any_used()
            return {}
        elif isinstance(other, KnownValue):
            if not isinstance(other.val, int):
                return CanAssignError(f"{other} is not an int")
            assert hasattr(self.typ, "_VALUES_TO_NAMES"), f"{self} is not a Thrift enum"
            if other.val in self.typ._VALUES_TO_NAMES:
                return {}
        elif isinstance(other, TypedValue):
            tobj = other.get_type_object(ctx)
            if tobj.is_assignable_to_type(int):
                return {}
            return self.get_type_object(ctx).can_assign(self, other, ctx)
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
        self, typ: Union[type, super, str], ctx: CanAssignContext
    ) -> Optional[List[Value]]:
        if isinstance(self, GenericValue):
            args = self.args
        else:
            args = ()
        if isinstance(self.typ, super):
            generic_bases = ctx.get_generic_bases(self.typ.__self_class__, args)
        else:
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
        return AnyValue(AnySource.generic_argument)

    def is_type(self, typ: type) -> bool:
        return self.get_type_object().is_assignable_to_type(typ)

    def get_type(self) -> Optional[type]:
        if isinstance(self.typ, str):
            return None
        return self.typ

    def get_type_value(self) -> Value:
        if isinstance(self.typ, str):
            return AnyValue(AnySource.inference)
        return KnownValue(self.typ)

    def __str__(self) -> str:
        if self._type_object is not None:
            return str(self._type_object)
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

    def __init__(self, typ: Union[type, str], args: Iterable[Value]) -> None:
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
        other = replace_known_sequence_value(other)
        if isinstance(other, TypedValue) and not isinstance(other.typ, super):
            generic_args = other.get_generic_args_for_type(self.typ, ctx)
            # If we don't think it's a generic base, try super;
            # runtime isinstance() may disagree.
            if generic_args is None or len(self.args) != len(generic_args):
                return super().can_assign(other, ctx)
            tv_maps = []
            for i, (my_arg, their_arg) in enumerate(zip(self.args, generic_args)):
                tv_map = my_arg.can_assign(their_arg, ctx)
                if isinstance(tv_map, CanAssignError):
                    return self.maybe_specify_error(i, other, tv_map, ctx)
                tv_maps.append(tv_map)
            if not tv_maps:
                return CanAssignError(f"Cannot assign {other} to {self}")
            return unify_typevar_maps(tv_maps)

        return super().can_assign(other, ctx)

    def maybe_specify_error(
        self, i: int, other: Value, error: CanAssignError, ctx: CanAssignContext
    ) -> CanAssignError:
        expected = self.get_arg(i)
        if isinstance(other, DictIncompleteValue) and self.typ in {
            dict,
            collections.abc.Mapping,
            collections.abc.MutableMapping,
        }:
            if i == 0:
                for pair in reversed(other.kv_pairs):
                    can_assign = expected.can_assign(pair.key, ctx)
                    if isinstance(can_assign, CanAssignError):
                        return CanAssignError(
                            f"In key of key-value pair {pair}", [can_assign]
                        )
            elif i == 1:
                for pair in reversed(other.kv_pairs):
                    can_assign = expected.can_assign(pair.value, ctx)
                    if isinstance(can_assign, CanAssignError):
                        return CanAssignError(
                            f"In value of key-value pair {pair}", [can_assign]
                        )
        elif isinstance(other, TypedDictValue) and self.typ in {
            dict,
            collections.abc.Mapping,
            collections.abc.MutableMapping,
        }:
            if i == 0:
                for key in other.items:
                    can_assign = expected.can_assign(KnownValue(key), ctx)
                    if isinstance(can_assign, CanAssignError):
                        return CanAssignError(f"In TypedDict key {key!r}", [can_assign])
            elif i == 1:
                for key, (_, value) in other.items.items():
                    can_assign = expected.can_assign(value, ctx)
                    if isinstance(can_assign, CanAssignError):
                        return CanAssignError(f"In TypedDict key {key!r}", [can_assign])
        elif isinstance(other, SequenceIncompleteValue) and self.typ in {
            list,
            set,
            tuple,
            collections.abc.Iterable,
            collections.abc.Sequence,
            collections.abc.MutableSequence,
            collections.abc.Container,
            collections.abc.Collection,
        }:
            for i, key in enumerate(other.members):
                can_assign = expected.can_assign(key, ctx)
                if isinstance(can_assign, CanAssignError):
                    return CanAssignError(f"In element {i}", [can_assign])

        return CanAssignError(f"In generic argument {i} to {self}", [error])

    def get_arg(self, index: int) -> Value:
        try:
            return self.args[index]
        except IndexError:
            return AnyValue(AnySource.generic_argument)

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for arg in self.args:
            yield from arg.walk_values()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return GenericValue(
            self.typ, [arg.substitute_typevars(typevars) for arg in self.args]
        )

    def simplify(self) -> Value:
        return GenericValue(self.typ, [arg.simplify() for arg in self.args])


@dataclass(unsafe_hash=True, init=False)
class SequenceIncompleteValue(GenericValue):
    """A :class:`TypedValue` subclass representing a sequence of known type and length.

    For example, the expression ``[int(self.foo)]`` may be typed as
    ``SequenceIncompleteValue(list, [TypedValue(int)])``.

    This is only used for ``set``, ``list``, and ``tuple``.

    """

    members: Tuple[Value, ...]
    """The elements of the sequence."""

    def __init__(self, typ: Union[type, str], members: Sequence[Value]) -> None:
        if members:
            args = (unite_values(*members),)
        else:
            args = (AnyValue(AnySource.unreachable),)
        super().__init__(typ, args)
        self.members = tuple(members)

    @classmethod
    def make_or_known(
        cls, typ: type, members: Sequence[Value]
    ) -> Union[KnownValue, "SequenceIncompleteValue"]:
        if all_of_type(members, KnownValue):
            return KnownValue(typ(member.val for member in members))
        else:
            return SequenceIncompleteValue(typ, members)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, SequenceIncompleteValue):
            tv_map = self.get_type_object(ctx).can_assign(self, other, ctx)
            if isinstance(tv_map, CanAssignError):
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

    def walk_values(self) -> Iterable[Value]:
        yield self
        for member in self.members:
            yield from member.walk_values()

    def simplify(self) -> GenericValue:
        if self.typ is tuple:
            return SequenceIncompleteValue(
                tuple, [member.simplify() for member in self.members]
            )
        members = [member.simplify() for member in self.members]
        return GenericValue(self.typ, [unite_values(*members)])


@dataclass(frozen=True)
class KVPair:
    """Represents a single entry in a :class:`DictIncompleteValue`."""

    key: Value
    """Represents the key."""
    value: Value
    """Represents the value."""
    is_many: bool = False
    """Whether this key-value pair represents possibly multiple keys."""
    is_required: bool = True
    """Whether this key-value pair is definitely present."""

    def substitute_typevars(self, typevars: TypeVarMap) -> "KVPair":
        return KVPair(
            self.key.substitute_typevars(typevars),
            self.value.substitute_typevars(typevars),
            self.is_many,
            self.is_required,
        )

    def __str__(self) -> str:
        query = "" if self.is_required else "?"
        text = f"{self.key}{query}: {self.value}"
        if self.is_many:
            return f"**{{{text}}}"
        else:
            return text


@dataclass(unsafe_hash=True, init=False)
class DictIncompleteValue(GenericValue):
    """A :class:`TypedValue` representing a dictionary of known size.

    For example, the expression ``{'foo': int(self.bar)}`` may be typed as
    ``DictIncompleteValue(dict, [KVPair(KnownValue('foo'), TypedValue(int))])``.

    """

    kv_pairs: Tuple[KVPair, ...]
    """Sequence of :class:`KVPair` objects representing the keys and values of the dict."""

    def __init__(self, typ: Union[type, str], kv_pairs: Sequence[KVPair]) -> None:
        if kv_pairs:
            key_type = unite_values(*[pair.key for pair in kv_pairs])
            value_type = unite_values(*[pair.value for pair in kv_pairs])
        else:
            key_type = value_type = AnyValue(AnySource.unreachable)
        super().__init__(typ, (key_type, value_type))
        self.kv_pairs = tuple(kv_pairs)

    def __str__(self) -> str:
        items = ", ".join(map(str, self.kv_pairs))
        return f"<{stringify_object(self.typ)} containing {{{items}}}>"

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for pair in self.kv_pairs:
            yield from pair.key.walk_values()
            yield from pair.value.walk_values()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return DictIncompleteValue(
            self.typ, [pair.substitute_typevars(typevars) for pair in self.kv_pairs]
        )

    def simplify(self) -> GenericValue:
        keys = [pair.key.simplify() for pair in self.kv_pairs]
        values = [pair.value.simplify() for pair in self.kv_pairs]
        return GenericValue(self.typ, [unite_values(*keys), unite_values(*values)])

    @property
    def items(self) -> Sequence[Tuple[Value, Value]]:
        """Sequence of pairs representing the keys and values of the dict."""
        return [(pair.key, pair.value) for pair in self.kv_pairs]

    def get_value(self, key: Value, ctx: CanAssignContext) -> Value:
        """Return the :class:`Value` for a specific key."""
        possible_values = []
        covered_keys: Set[Value] = set()
        for pair in reversed(self.kv_pairs):
            if not pair.is_many:
                if isinstance(pair.key, AnnotatedValue):
                    my_key = pair.key.value
                else:
                    my_key = pair.key
                if isinstance(my_key, KnownValue):
                    if my_key == key and pair.is_required:
                        return unite_values(*possible_values, pair.value)
                    elif my_key in covered_keys:
                        continue
                    elif pair.is_required:
                        covered_keys.add(my_key)
            if key.is_assignable(pair.key, ctx) or pair.key.is_assignable(key, ctx):
                possible_values.append(pair.value)
        if not possible_values:
            return UNINITIALIZED_VALUE
        return unite_values(*possible_values)


@dataclass(init=False)
class TypedDictValue(GenericValue):
    """Equivalent to ``typing.TypedDict``; a dictionary with a known set of string keys."""

    items: Dict[str, Tuple[bool, Value]]
    """The items of the ``TypedDict``. Required items are represented as (True, value) and optional
    ones as (False, value)."""

    def __init__(self, items: Dict[str, Tuple[bool, Value]]) -> None:
        if items:
            value_type = unite_values(*[val for _, val in items.values()])
        else:
            value_type = AnyValue(AnySource.unreachable)
        super().__init__(dict, (TypedValue(str), value_type))
        self.items = items

    def num_required_keys(self) -> int:
        return sum(1 for required, _ in self.items.values() if required)

    def all_keys_required(self) -> bool:
        return all(required for required, _ in self.items.values())

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, DictIncompleteValue):
            tv_maps = []
            for key, (is_required, value) in self.items.items():
                their_value = other.get_value(KnownValue(key), ctx)
                if their_value is UNINITIALIZED_VALUE:
                    if is_required:
                        return CanAssignError(f"Key {key} is missing in {other}")
                    else:
                        continue
                tv_map = value.can_assign(their_value, ctx)
                if isinstance(tv_map, CanAssignError):
                    return CanAssignError(
                        f"Types for key {key} are incompatible", children=[tv_map]
                    )
                tv_maps.append(tv_map)
            return unify_typevar_maps(tv_maps)
        elif isinstance(other, TypedDictValue):
            tv_maps = []
            for key, (is_required, value) in self.items.items():
                if key not in other.items:
                    if is_required:
                        return CanAssignError(f"Key {key} is missing in {other}")
                else:
                    tv_map = value.can_assign(other.items[key][1], ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"Types for key {key} are incompatible", children=[tv_map]
                        )
                    tv_maps.append(tv_map)
            return unify_typevar_maps(tv_maps)
        elif isinstance(other, KnownValue) and isinstance(other.val, dict):
            tv_maps = []
            for key, (is_required, value) in self.items.items():
                if key not in other.val:
                    if is_required:
                        return CanAssignError(f"Key {key} is missing in {other}")
                else:
                    tv_map = value.can_assign(KnownValue(other.val[key]), ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"Types for key {key} are incompatible", children=[tv_map]
                        )
                    tv_maps.append(tv_map)
            return unify_typevar_maps(tv_maps)
        return super().can_assign(other, ctx)

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return TypedDictValue(
            {
                key: (is_required, value.substitute_typevars(typevars))
                for key, (is_required, value) in self.items.items()
            }
        )

    def __str__(self) -> str:
        items = [
            f'"{key}": {value if required else "NotRequired[" + str(value) + "]"}'
            for key, (required, value) in self.items.items()
        ]
        return "TypedDict({%s})" % ", ".join(items)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items)))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for _, value in self.items.values():
            yield from value.walk_values()


@dataclass(unsafe_hash=True, init=False)
class AsyncTaskIncompleteValue(GenericValue):
    """A :class:`GenericValue` representing an async task.

    This should probably just be replaced with ``GenericValue``.

    """

    value: Value
    """The value returned by the task on completion."""

    def __init__(self, typ: Union[type, str], value: Value) -> None:
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

    signature: "pyanalyze.signature.ConcreteSignature"

    def __init__(self, signature: "pyanalyze.signature.ConcreteSignature") -> None:
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
        if not isinstance(other, (MultiValuedValue, AnyValue)):
            signature = ctx.signature_from_value(other)
            if signature is None:
                return CanAssignError(f"{other} is not a callable type")
            if isinstance(signature, pyanalyze.signature.BoundMethodSignature):
                signature = signature.get_signature()
            if isinstance(
                signature,
                (
                    pyanalyze.signature.Signature,
                    pyanalyze.signature.OverloadedSignature,
                ),
            ):
                return self.signature.can_assign(signature, ctx)

        return super().can_assign(other, ctx)

    def __str__(self) -> str:
        return str(self.signature)


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
        if isinstance(self.typ, TypedValue) and isinstance(self.typ.typ, type):
            return safe_issubclass(self.typ.typ, typ)
        return False

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, SubclassValue):
            return self.typ.can_assign(other.typ, ctx)
        elif isinstance(other, KnownValue):
            if isinstance(other.val, type):
                if isinstance(self.typ, TypedValue):
                    self_tobj = self.typ.get_type_object(ctx)
                    return self_tobj.can_assign(self, TypedValue(other.val), ctx)
                elif isinstance(self.typ, TypeVarValue):
                    return {self.typ.typevar: TypedValue(other.val)}
        elif isinstance(other, TypedValue):
            if other.typ is type:
                return {}
            # metaclass
            tobj = other.get_type_object(ctx)
            if tobj.is_assignable_to_type(type) and (
                (
                    isinstance(self.typ, TypedValue)
                    and tobj.is_metatype_of(self.typ.get_type_object(ctx))
                )
                or isinstance(self.typ, TypeVarValue)
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
            return AnyValue(AnySource.inference)

    def __str__(self) -> str:
        return f"Type[{self.typ}]"

    @classmethod
    def make(cls, origin: Value) -> Value:
        if isinstance(origin, MultiValuedValue):
            return unite_values(*[cls.make(val) for val in origin.vals])
        elif isinstance(origin, AnyValue):
            # Type[Any] is equivalent to plain type
            return TypedValue(type)
        elif isinstance(origin, (TypeVarValue, TypedValue)):
            return cls(origin)
        else:
            return AnyValue(AnySource.inference)


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
        if isinstance(other, TypeVarValue):
            other = other.get_fallback_value()
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
        elif isinstance(other, AnyValue):
            ctx.record_any_used()
            return {}
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
        literals: List[KnownValue] = []
        has_none = False
        others: List[Value] = []
        for val in self.vals:
            if val == KnownValue(None):
                has_none = True
            elif isinstance(val, KnownValue):
                literals.append(val)
            else:
                others.append(val)
        if not others:
            if has_none:
                literals.append(KnownValue(None))
            body = ", ".join(repr(val.val) for val in literals)
            return f"Literal[{body}]"
        else:
            if not literals and has_none and len(others) == 1:
                return f"Optional[{others[0]}]"
            elements = [str(val) for val in others]
            if literals:
                body = ", ".join(repr(val.val) for val in literals)
                elements.append(f"Literal[{body}]")
            if has_none:
                elements.append("None")
            return f"Union[{', '.join(elements)}]"

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for val in self.vals:
            yield from val.walk_values()

    def simplify(self) -> Value:
        return unite_values(*[val.simplify() for val in self.vals])


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
    bound: Optional[Value] = None
    constraints: Sequence[Value] = ()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return typevars.get(self.typevar, self)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if self == other:
            return {}
        if self.bound is not None:
            can_assign = self.bound.can_assign(other, ctx)
            if isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"Value of TypeVar {self} cannot be {other}", [can_assign]
                )
            return {**can_assign, self.typevar: other}
        elif self.constraints:
            can_assigns = [
                constraint.can_assign(other, ctx) for constraint in self.constraints
            ]
            if all_of_type(can_assigns, CanAssignError):
                return CanAssignError(f"Cannot assign to {self}", list(can_assigns))
            possibilities = [
                constraint
                for constraint, can_assign in zip(self.constraints, can_assigns)
                if not isinstance(can_assign, CanAssignError)
            ]
            if len(possibilities) == 1:
                (solution,) = possibilities
            else:
                # Inferring something else produces too many issues for now.
                solution = AnyValue(AnySource.inference)
            return {self.typevar: solution}
        return {self.typevar: other}

    def can_be_assigned(self, left: Value, ctx: CanAssignContext) -> CanAssign:
        if left == self:
            return {}
        if self.bound is not None:
            # TODO not sure this is right, but it helps test cases in
            # test_annotations.py behave as expected.
            can_assign = self.bound.can_assign(left, ctx)
            if isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"Value of TypeVar {self} cannot be {left}", [can_assign]
                )
            return {**can_assign, self.typevar: left}
        elif self.constraints:
            can_assigns = [
                left.can_assign(constraint, ctx) for constraint in self.constraints
            ]
            if all_of_type(can_assigns, CanAssignError):
                return CanAssignError(f"Cannot assign to {self}", list(can_assigns))
            possibilities = [
                constraint
                for constraint, can_assign in zip(self.constraints, can_assigns)
                if not isinstance(can_assign, CanAssignError)
            ]
            if len(possibilities) == 1:
                (solution,) = possibilities
            else:
                # Inferring something else produces too many issues for now.
                solution = AnyValue(AnySource.inference)
            return {self.typevar: solution}
        return {self.typevar: left}

    def get_fallback_value(self) -> Value:
        if self.bound is not None:
            return self.bound
        elif self.constraints:
            return unite_values(*self.constraints)
        return AnyValue(AnySource.inference)

    def get_type_value(self) -> Value:
        return self.get_fallback_value().get_type_value()

    def __str__(self) -> str:
        if self.bound is not None:
            return f"{self.typevar} <: {self.bound}"
        elif self.constraints:
            constraints = ", ".join(map(str, self.constraints))
            return f"{self.typevar} in ({constraints})"
        return str(self.typevar)


class Extension:
    """An extra piece of information about a type that can be stored in
    an :class:`AnnotatedValue`."""

    __slots__ = ()

    def substitute_typevars(self, typevars: TypeVarMap) -> "Extension":
        return self

    def walk_values(self) -> Iterable[Value]:
        return []


@dataclass(frozen=True)
class CustomCheckExtension(Extension):
    custom_check: CustomCheck

    def __str__(self) -> str:
        # This extra wrapper class just adds noise
        return str(self.custom_check)

    def substitute_typevars(self, typevars: TypeVarMap) -> "Extension":
        return CustomCheckExtension(self.custom_check.substitute_typevars(typevars))

    def walk_values(self) -> Iterable[Value]:
        yield from self.custom_check.walk_values()


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    metadata: Tuple[Union[Value, Extension], ...]
    """The extensions associated with this value."""

    def __init__(
        self, value: Value, metadata: Sequence[Union[Value, Extension]]
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "metadata", tuple(metadata))

    def is_type(self, typ: type) -> bool:
        return self.value.is_type(typ)

    def get_type(self) -> Optional[type]:
        return self.value.get_type()

    def get_type_value(self) -> Value:
        return self.value.get_type_value()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        metadata = tuple(val.substitute_typevars(typevars) for val in self.metadata)
        return AnnotatedValue(self.value.substitute_typevars(typevars), metadata)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        can_assign = self.value.can_assign(other, ctx)
        if isinstance(can_assign, CanAssignError):
            return can_assign
        tv_maps = [can_assign]
        for custom_check in self.get_metadata_of_type(CustomCheckExtension):
            custom_can_assign = custom_check.custom_check.can_assign(other, ctx)
            if isinstance(custom_can_assign, CanAssignError):
                return custom_can_assign
            tv_maps.append(custom_can_assign)
        return unify_typevar_maps(tv_maps)

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

    def get_custom_check_of_type(self, typ: Type[T]) -> Iterable[T]:
        """Return any CustomChecks of the given type in the metadata."""
        for custom_check in self.get_metadata_of_type(CustomCheckExtension):
            if isinstance(custom_check.custom_check, typ):
                yield custom_check.custom_check

    def has_metadata_of_type(self, typ: Type[Extension]) -> bool:
        """Return whether there is metadat of the given type."""
        return any(isinstance(data, typ) for data in self.metadata)

    def __str__(self) -> str:
        return f"Annotated[{self.value}, {', '.join(map(str, self.metadata))}]"

    def simplify(self) -> Value:
        return AnnotatedValue(self.value.simplify(), self.metadata)


@dataclass(frozen=True)
class VariableNameValue(AnyValue):
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

    def __init__(self, varnames: Iterable[str]) -> None:
        super().__init__(AnySource.variable_name)
        object.__setattr__(self, "varnames", tuple(varnames))

    varnames: Tuple[str, ...]

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if not isinstance(other, VariableNameValue):
            if isinstance(other, AnyValue):
                ctx.record_any_used()
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
    if isinstance(origin, MultiValuedValue):
        return MultiValuedValue(
            [annotate_value(subval, metadata) for subval in origin.vals]
        )
    if isinstance(origin, AnnotatedValue):
        # Flatten it
        metadata = (*origin.metadata, *metadata)
        origin = origin.value
    # Make sure order is consistent; conceptually this is a set but
    # sets have unpredictable iteration order.
    hashable_vals = OrderedDict()
    unhashable_vals = []
    for item in metadata:
        try:
            # Don't readd it to preserve original ordering.
            if item not in hashable_vals:
                hashable_vals[item] = None
        except Exception:
            unhashable_vals.append(item)
    metadata = (*hashable_vals, *unhashable_vals)
    return AnnotatedValue(origin, metadata)


def unite_and_simplify(*values: Value, limit: int) -> Value:
    united = unite_values(*values)
    if not isinstance(united, MultiValuedValue) or len(united.vals) < limit:
        return united
    simplified = [val.simplify() for val in united.vals]
    return unite_values(*simplified)


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
    for value in values:
        if isinstance(value, MultiValuedValue):
            subvals = value.vals
        elif isinstance(value, AnnotatedValue) and isinstance(
            value.value, MultiValuedValue
        ):
            subvals = [
                annotate_value(subval, value.metadata) for subval in value.value.vals
            ]
        else:
            subvals = [value]
        for subval in subvals:
            try:
                # Don't readd it to preserve original ordering.
                if subval not in hashable_vals:
                    hashable_vals[subval] = None
            except Exception:
                unhashable_vals.append(subval)
    existing = list(hashable_vals) + unhashable_vals
    num = len(existing)
    if num == 0:
        return NO_RETURN_VALUE
    if num == 1:
        return existing[0]
    else:
        return MultiValuedValue(existing)


T = TypeVar("T")
IterableValue = GenericValue(collections.abc.Iterable, [TypeVarValue(T)])


class GetItemProto(Protocol[T]):
    def __getitem__(self, i: int) -> T:
        raise NotImplementedError


GetItemProtoValue = GenericValue(GetItemProto, [TypeVarValue(T)])


def concrete_values_from_iterable(
    value: Value, ctx: CanAssignContext
) -> Union[CanAssignError, Value, Sequence[Value]]:
    """Return the exact values that can be extracted from an iterable.

    Three possible return types:

    - :class:`CanAssignError` if the argument is not iterable
    - A sequence of :class:`Value` if we know the exact types in the iterable
    - A single :class:`Value` if we just know that the iterable contains this
      value, but not the precise number of them.

    Examples:

    - ``int`` -> ``CanAssignError``
    - ``tuple[int, str]`` -> ``(int, str)``
    - ``tuple[int, ...]`` -> ``int``

    """
    value = replace_known_sequence_value(value)
    if isinstance(value, MultiValuedValue):
        subvals = [concrete_values_from_iterable(val, ctx) for val in value.vals]
        errors = [subval for subval in subvals if isinstance(subval, CanAssignError)]
        if errors:
            return CanAssignError(
                "At least one member of Union is not iterable", errors
            )
        value_subvals = [subval for subval in subvals if isinstance(subval, Value)]
        seq_subvals = [
            subval
            for subval in subvals
            if not isinstance(subval, (Value, CanAssignError))
        ]
        if not value_subvals and len(set(map(len, seq_subvals))) == 1:
            return [unite_values(*vals) for vals in zip(*seq_subvals)]
        return unite_values(*value_subvals, *chain.from_iterable(seq_subvals))
    if isinstance(value, SequenceIncompleteValue):
        return value.members
    elif isinstance(value, TypedDictValue):
        if all(required for required, _ in value.items.items()):
            return [KnownValue(key) for key in value.items]
        return MultiValuedValue([KnownValue(key) for key in value.items])
    elif isinstance(value, DictIncompleteValue):
        if all(pair.is_required and not pair.is_many for pair in value.kv_pairs):
            return [pair.key for pair in value.kv_pairs]
    elif isinstance(value, KnownValue):
        if isinstance(value.val, (str, bytes, range)):
            return [KnownValue(c) for c in value.val]
    elif value is NO_RETURN_VALUE:
        return NO_RETURN_VALUE
    iter_tv_map = IterableValue.can_assign(value, ctx)
    if not isinstance(iter_tv_map, CanAssignError):
        return iter_tv_map.get(T, AnyValue(AnySource.generic_argument))
    getitem_tv_map = GetItemProtoValue.can_assign(value, ctx)
    if not isinstance(getitem_tv_map, CanAssignError):
        return getitem_tv_map.get(T, AnyValue(AnySource.generic_argument))
    return iter_tv_map


K = TypeVar("K")
V = TypeVar("V")
MappingValue = GenericValue(collections.abc.Mapping, [TypeVarValue(K), TypeVarValue(V)])

EMPTY_DICTS = (KnownValue({}), DictIncompleteValue(dict, []))


def kv_pairs_from_mapping(
    value_val: Value, ctx: CanAssignContext
) -> Union[Sequence[KVPair], CanAssignError]:
    """Return the :class:`KVPair` objects that can be extracted from this value,
    or a :class:`CanAssignError` on error."""
    value_val = replace_known_sequence_value(value_val)
    # Special case: if we have a Union including an empty dict, just get the
    # pairs from the rest of the union and make them all non-required.
    if isinstance(value_val, MultiValuedValue) and any(
        subval in EMPTY_DICTS for subval in value_val.vals
    ):
        other_val = unite_values(
            *[subval for subval in value_val.vals if subval not in EMPTY_DICTS]
        )
        pairs = kv_pairs_from_mapping(other_val, ctx)
        if isinstance(pairs, CanAssignError):
            return pairs
        return [
            KVPair(pair.key, pair.value, pair.is_many, is_required=False)
            for pair in pairs
        ]
    if isinstance(value_val, DictIncompleteValue):
        return value_val.kv_pairs
    elif isinstance(value_val, TypedDictValue):
        return [
            KVPair(KnownValue(key), value, is_required=required)
            for key, (required, value) in value_val.items.items()
        ]
    else:
        can_assign = MappingValue.can_assign(value_val, ctx)
        if isinstance(can_assign, CanAssignError):
            return can_assign
        key_type = can_assign.get(K, AnyValue(AnySource.generic_argument))
        value_type = can_assign.get(V, AnyValue(AnySource.generic_argument))
        return [KVPair(key_type, value_type, is_many=True)]


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
                AnyValue(AnySource.error)
                if subvals
                else AnyValue(AnySource.unreachable),
                target_length,
                post_starred_length,
            )
        return [unite_values(*vals) for vals in zip(*good_subvals)]
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
    iterable_type = tv_map.get(T, AnyValue(AnySource.generic_argument))
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
    """Simplify a Value in a way that is easier to handle for most typechecking use cases.

    Does the following:

    - Replace AnnotatedValue with its inner type
    - Replace TypeVarValue with its fallback type
    - Replace KnownValues representing list, tuples, sets, or dicts with
      SequenceIncompleteValue or DictIncompleteValue.

    """
    if isinstance(value, AnnotatedValue):
        return replace_known_sequence_value(value.value)
    if isinstance(value, TypeVarValue):
        return replace_known_sequence_value(value.get_fallback_value())
    if isinstance(value, KnownValue):
        if isinstance(value.val, (list, tuple, set)):
            return SequenceIncompleteValue(
                type(value.val), [KnownValue(elt) for elt in value.val]
            )
        elif isinstance(value.val, dict):
            return DictIncompleteValue(
                type(value.val),
                [KVPair(KnownValue(k), KnownValue(v)) for k, v in value.val.items()],
            )
    return value


def extract_typevars(value: Value) -> Iterable["TypeVar"]:
    for val in value.walk_values():
        if isinstance(val, TypeVarValue):
            yield val.typevar


def stringify_object(obj: Any) -> str:
    # Stringify arbitrary Python objects such as methods and types.
    if isinstance(obj, str):
        return obj
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
