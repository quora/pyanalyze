"""

Implementation of value classes, which represent values found while analyzing an AST.

"""

import ast
from collections import OrderedDict
from dataclasses import dataclass, field, InitVar
import inspect
from itertools import chain
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    TypeVar,
)
from unittest import mock

# __builtin__ in Python 2 and builtins in Python 3
BUILTIN_MODULE = str.__module__

TypeOrTuple = Union[type, Tuple[type, ...]]
TypeVarMap = Mapping["TypeVar", "Value"]


class Value(object):
    """Class that represents the value of a variable."""

    def is_value_compatible(self, val: "Value") -> bool:
        """Returns whether the given value is compatible with this value.

        val must be a more precise type than (or the same type as) self.

        Callers should always go through NameCheckVisitor.is_value_compatible,
        because this function may raise errors since it can call into user code.

        """
        return True

    def is_type(self, typ: TypeOrTuple) -> bool:
        """Returns whether this value is an instance of the given type."""
        return False

    def get_type(self) -> Optional[type]:
        """Returns the type of this value, or None if it is not known."""
        return None

    def get_type_value(self) -> "Value":
        """Return the type of this object as used for dunder lookups."""
        return self

    def walk_values(self) -> Iterable["Value"]:
        """Walk over all values contained in this value."""
        yield self

    def apply_typevars(
        self, value: "Value", typevars: TypeVarMap
    ) -> Tuple["Value", TypeVarMap]:
        """Greedily apply the typevars in the map to this value.

        Return a tuple of a new value and newly applied typevars.

        """
        return self, {}


@dataclass(frozen=True)
class UnresolvedValue(Value):
    """Value that we cannot resolve further."""

    def __str__(self) -> str:
        return "Any"


UNRESOLVED_VALUE = UnresolvedValue()


@dataclass(frozen=True)
class UninitializedValue(Value):
    """Value for variables that have not been initialized.

    Usage of variables with this value should be an error.

    """

    def __str__(self) -> str:
        return "<uninitialized>"


UNINITIALIZED_VALUE = UninitializedValue()


@dataclass(frozen=True)
class NoReturnValue(Value):
    """Value that indicates that a function will never return."""

    def __str__(self) -> str:
        return "NoReturn"

    def is_value_compatible(self, val: Value) -> bool:
        # You can't assign anything to NoReturn
        return False


NO_RETURN_VALUE = NoReturnValue()


@dataclass(frozen=True)
class KnownValue(Value):
    """Variable with a known value."""

    val: Any
    source_node: Optional[ast.AST] = field(
        default=None, repr=False, hash=False, compare=False
    )

    def is_value_compatible(self, val: Value) -> bool:
        if isinstance(val, TypeVarValue):
            val = val.get_fallback_value()
        if isinstance(val, KnownValue):
            return self.val == val.val
        elif isinstance(val, TypedValue):
            if self.is_type(val.typ):
                return True
            return are_types_compatible(type(self.val), val.typ)
        elif isinstance(val, MultiValuedValue):
            return all(self.is_value_compatible(v) for v in val.vals)
        elif isinstance(val, SubclassValue):
            return isinstance(self.val, type) and issubclass(self.val, val.typ)
        else:
            return True

    def is_type(self, typ: TypeOrTuple) -> bool:
        return isinstance(self.val, typ)

    def get_type(self) -> type:
        return type(self.val)

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

    def get_type_value(self) -> Value:
        return KnownValue(type(self.val))


@dataclass(frozen=True)
class UnboundMethodValue(Value):
    """Value that represents an unbound method.

    That is, we know that this value is this method, but we don't have the instance it is called on.

    """

    attr_name: str
    typ: Value
    secondary_attr_name: Optional[str] = None

    def get_method(self) -> Optional[Any]:
        """Returns the method object for this UnboundMethodValue."""
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

    def is_type(self, typ: TypeOrTuple) -> bool:
        return isinstance(self.get_method(), typ)

    def get_type(self) -> type:
        return type(self.get_method())

    def get_type_value(self) -> Value:
        return KnownValue(type(self.get_method()))

    def __str__(self) -> str:
        return "<method %s%s on %s>" % (
            self.attr_name,
            ".%s" % (self.secondary_attr_name,) if self.secondary_attr_name else "",
            self.typ,
        )


@dataclass(unsafe_hash=True)
class TypedValue(Value):
    """Value for which we know the type.

    There are several subclasses of this that are used when we have an _incomplete value_: we don't
    know the full value, but have more information than just the type. For example, in this
    function:

        def fn(a, b):
            x = [a, b]

    we don't have enough information to make x a KnownValue, but we know that it is a list of two
    elements. In this case, we set it to
    SequenceIncompleteValue(list, [UNRESOLVED_VALUE, UNRESOLVED_VALUE]).

    """

    typ: Any

    def is_value_compatible(self, val: Value) -> bool:
        if isinstance(val, TypeVarValue):
            val = val.get_fallback_value()
        if hasattr(self.typ, "_VALUES_TO_NAMES"):
            # Special case: Thrift enums. These are conceptually like
            # enums, but they are ints at runtime.
            return self.is_value_compatible_thrift_enum(val)
        elif isinstance(val, KnownValue):
            if isinstance(val.val, self.typ):
                return True
            return are_types_compatible(self.typ, type(val.val))
        elif isinstance(val, TypedValue):
            if issubclass(val.typ, self.typ):
                return True
            return are_types_compatible(self.typ, val.typ)
        elif isinstance(val, MultiValuedValue):
            return all(self.is_value_compatible(v) for v in val.vals)
        elif isinstance(val, SubclassValue):
            # Valid only if self.typ is a metaclass of val.typ.
            return isinstance(val.typ, self.typ)
        else:
            return True

    def is_value_compatible_thrift_enum(self, val: Value) -> bool:
        if isinstance(val, KnownValue):
            if not isinstance(val.val, int):
                return False
            return val.val in self.typ._VALUES_TO_NAMES
        elif isinstance(val, TypedValue):
            return issubclass(val.typ, (self.typ, int))
        elif isinstance(val, MultiValuedValue):
            return all(self.is_value_compatible_thrift_enum(v) for v in val.vals)
        elif isinstance(val, (SubclassValue, UnboundMethodValue)):
            return False
        elif val is UNRESOLVED_VALUE or val is NO_RETURN_VALUE:
            return True
        else:
            return False

    def is_type(self, typ: TypeOrTuple) -> bool:
        return issubclass(self.typ, typ)

    def get_type(self) -> type:
        return self.typ

    def get_type_value(self) -> Value:
        return KnownValue(self.typ)

    def __str__(self) -> str:
        return _stringify_type(self.typ)


@dataclass(unsafe_hash=True, init=False)
class NewTypeValue(TypedValue):
    """A wrapper around an underlying type.

    Corresponds to typing.NewType.

    """

    name: str
    newtype: Any

    def __init__(self, newtype: Any) -> None:
        super(NewTypeValue, self).__init__(newtype.__supertype__)
        self.name = newtype.__name__
        self.newtype = newtype

    def is_value_compatible(self, val: Value) -> bool:
        if isinstance(val, NewTypeValue):
            return self.newtype is val.newtype
        else:
            return super(NewTypeValue, self).is_value_compatible(val)

    def __str__(self) -> str:
        return "NewType(%r, %s)" % (self.name, _stringify_type(self.typ))


@dataclass(unsafe_hash=True, init=False)
class GenericValue(TypedValue):
    """A TypedValue representing a generic.

    For example, List[int] is represented as GenericValue(list, [TypedValue(int)]).

    """

    args: Tuple[Value, ...]

    def __init__(self, typ: type, args: Iterable[Value]) -> None:
        super().__init__(typ)
        self.args = tuple(args)

    def __str__(self) -> str:
        if self.typ is tuple:
            args = list(self.args) + ["..."]
        else:
            args = self.args
        return "%s[%s]" % (
            _stringify_type(self.typ),
            ", ".join(str(arg) for arg in args),
        )

    def is_value_compatible(self, val: Value) -> bool:
        if isinstance(val, TypeVarValue):
            val = val.get_fallback_value()
        if isinstance(val, GenericValue):
            if not super(GenericValue, self).is_value_compatible(val):
                return False
            # For now we treat all generics as covariant and
            # assume that their type arguments match.
            for arg1, arg2 in zip(self.args, val.args):
                if not arg1.is_value_compatible(arg2):
                    return False
            return True
        else:
            return super(GenericValue, self).is_value_compatible(val)

    def get_arg(self, index: int) -> Value:
        try:
            return self.args[index]
        except IndexError:
            return UNRESOLVED_VALUE

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for arg in self.args:
            yield from arg.walk_values()

    def apply_typevars(
        self, val: "Value", typevars: TypeVarMap
    ) -> Tuple["Value", TypeVarMap]:
        if (
            isinstance(val, GenericValue)
            and self.is_value_compatible(val)
            and self.typ is val.typ
            and len(self.args) == len(val.args)
        ):
            new_args = []
            for my_arg, their_arg in zip(self.args, val.args):
                new_val, new_typevars = my_arg.apply_typevars(their_arg, typevars)
                typevars = {**typevars, **new_typevars}
                new_args.append(new_val)
            return GenericValue(self.typ, new_args), typevars
        else:
            return self, {}


@dataclass(unsafe_hash=True, init=False)
class SequenceIncompleteValue(GenericValue):
    """A TypedValue representing a sequence whose members are not completely known.

    For example, the expression [int(self.foo)] may be typed as
    SequenceValue(list, [TypedValue(int)])

    """

    members: Tuple[Value, ...]

    def __init__(self, typ: type, members: Sequence[Value]) -> None:
        if members:
            args = (unite_values(*members),)
        else:
            args = (UNRESOLVED_VALUE,)
        super().__init__(typ, args)
        self.members = tuple(members)

    def is_value_compatible(self, val: Value) -> bool:
        if isinstance(val, SequenceIncompleteValue):
            if not issubclass(val.typ, self.typ):
                return False
            if len(val.members) != len(self.members):
                return False
            return all(
                my_member.is_value_compatible(other_member)
                for my_member, other_member in zip(self.members, val.members)
            )
        else:
            return super(SequenceIncompleteValue, self).is_value_compatible(val)

    def __str__(self) -> str:
        if self.typ is tuple:
            return "tuple[%s]" % (", ".join(str(m) for m in self.members))
        return "<%s containing [%s]>" % (
            _stringify_type(self.typ),
            ", ".join(map(str, self.members)),
        )

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for member in self.members:
            yield from member.walk_values()


@dataclass(unsafe_hash=True, init=False)
class DictIncompleteValue(GenericValue):
    """A TypedValue representing a dictionary whose keys and values are not completely known.

    For example, the expression {'foo': int(self.bar)} may be typed as
    DictIncompleteValue([(KnownValue('foo'), TypedValue(int))]).

    """

    items: List[Tuple[Value, Value]]

    def __init__(self, items: List[Tuple[Value, Value]]) -> None:
        if items:
            key_type = unite_values(*[key for key, _ in items])
            value_type = unite_values(*[value for _, value in items])
        else:
            key_type = value_type = UNRESOLVED_VALUE
        super().__init__(dict, (key_type, value_type))
        self.items = items

    def __str__(self) -> str:
        return "<%s containing {%s}>" % (
            _stringify_type(self.typ),
            ", ".join("%s: %s" % (key, value) for key, value in self.items),
        )

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for key, value in self.items:
            yield from key.walk_values()
            yield from value.walk_values()


@dataclass(init=False)
class TypedDictValue(GenericValue):
    items: Dict[str, Value]

    def __init__(self, items: Dict[str, Value]) -> None:
        if items:
            value_type = unite_values(*items.values())
        else:
            value_type = UNRESOLVED_VALUE
        super(TypedDictValue, self).__init__(dict, (TypedValue(str), value_type))
        self.items = items

    def is_value_compatible(self, val: Value) -> bool:
        if isinstance(val, DictIncompleteValue):
            if len(val.items) < len(self.items):
                return False
            known_part = {
                key.val: value
                for key, value in val.items
                if isinstance(key, KnownValue) and isinstance(key.val, str)
            }
            has_unknowns = len(known_part) < len(val.items)
            for key, value in self.items.items():
                if key not in known_part:
                    if not has_unknowns:
                        return False
                else:
                    if not value.is_value_compatible(known_part[key]):
                        return False
            return True
        elif isinstance(val, TypedDictValue):
            for key, value in self.items.items():
                if key not in val.items:
                    return False
                if not value.is_value_compatible(val.items[key]):
                    return False
            return True
        elif isinstance(val, KnownValue) and isinstance(val.val, dict):
            for key, value in self.items.items():
                if key not in val.val:
                    return False
                if not value.is_value_compatible(KnownValue(val.val[key])):
                    return False
            return True
        else:
            return super(TypedDictValue, self).is_value_compatible(val)

    def __str__(self) -> str:
        items = ['"%s": %s' % (key, value) for key, value in self.items.items()]
        return "TypedDict({%s})" % ", ".join(items)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items)))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for value in self.items.values():
            yield from value.walk_values()


@dataclass(unsafe_hash=True, init=False)
class AsyncTaskIncompleteValue(GenericValue):
    """A TypedValue representing an async task.

    value is the value that the task object wraps.

    """

    value: Value

    def __init__(self, typ: type, value: Value) -> None:
        super(AsyncTaskIncompleteValue, self).__init__(typ, (value,))
        self.value = value

    def walk_values(self) -> Iterable["Value"]:
        yield self
        yield from self.value.walk_values()


@dataclass(frozen=True)
class AwaitableIncompleteValue(Value):
    """A Value representing a Python 3 awaitable, e.g. a coroutine object."""

    value: Value

    def get_type_value(self) -> Value:
        return UNRESOLVED_VALUE

    def __str__(self) -> str:
        return "Awaitable[%s]" % (self.value,)

    def walk_values(self) -> Iterable["Value"]:
        yield self
        yield from self.value.walk_values()


@dataclass(frozen=True)
class SubclassValue(Value):
    """Value that is either a type or its subclass."""

    typ: type

    def is_type(self, typ: TypeOrTuple) -> bool:
        try:
            return issubclass(self.typ, typ)
        except Exception:
            return False

    def is_value_compatible(self, val: Value) -> bool:
        if isinstance(val, TypeVarValue):
            val = val.get_fallback_value()
        if isinstance(val, MultiValuedValue):
            return all(self.is_value_compatible(subval) for subval in val.vals)
        elif isinstance(val, SubclassValue):
            return issubclass(val.typ, self.typ)
        elif isinstance(val, KnownValue):
            if isinstance(val.val, type):
                return issubclass(val.val, self.typ)
            else:
                return False
        elif isinstance(val, TypedValue):
            if val.typ is type:
                return True
            elif issubclass(val.typ, type):
                # metaclass
                return isinstance(self.typ, val.typ)
            else:
                return False
        elif val is UNRESOLVED_VALUE or val is NO_RETURN_VALUE:
            return True
        else:
            return False

    def get_type(self) -> type:
        return type(self.typ)

    def get_type_value(self) -> Value:
        return KnownValue(type(self.typ))

    def __str__(self) -> str:
        return "Type[%s]" % (_stringify_type(self.typ),)


@dataclass(frozen=True, order=False)
class MultiValuedValue(Value):
    """Variable for which multiple possible values have been recorded."""

    raw_vals: InitVar[Iterable[Value]]
    vals: Tuple[Value, ...] = field(init=False)

    def __post_init__(self, raw_vals: Iterable[Value]) -> None:
        object.__setattr__(
            self,
            "vals",
            tuple(chain.from_iterable(flatten_values(val) for val in raw_vals)),
        )

    def is_value_compatible(self, val: Value) -> bool:
        if isinstance(val, MultiValuedValue):
            return all(
                any(subval.is_value_compatible(other_subval) for subval in self.vals)
                for other_subval in val.vals
            )
        else:
            return any(subval.is_value_compatible(val) for subval in self.vals)

    def get_type_value(self) -> Value:
        return MultiValuedValue([val.get_type_value() for val in self.vals])

    def __eq__(self, other: Value) -> Union[bool, type(NotImplemented)]:
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
        return "Union[%s]" % ", ".join(map(str, self.vals))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for val in self.vals:
            yield from val.walk_values()


@dataclass(frozen=True)
class ReferencingValue(Value):
    """Value that is a reference to another value (used to implement globals)."""

    scope: Any
    name: str

    def __str__(self) -> str:
        return "<reference to %s>" % (self.name,)


@dataclass(frozen=True)
class TypeVarValue(Value):
    """Value representing a type variable."""

    typevar: TypeVar

    def apply_typevars(
        self, value: "Value", typevars: TypeVarMap
    ) -> Tuple["Value", TypeVarMap]:
        if self.typevar in typevars:
            # Ignore the passed value, we'll error elsewhere
            return (typevars[self.typevar], {})
        return value, {self.typevar: value}

    def get_fallback_value(self) -> Value:
        # TODO: support bounds and bases here to do something smarter
        return UNRESOLVED_VALUE


@dataclass(frozen=True)
class VariableNameValue(Value):
    """Value that is stored in a variable associated with a particular kind of value.

    For example, any variable named 'uid' will get resolved into a VariableNameValue of type uid,
    and if it gets passed into a function that takes an argument called 'aid',
    is_value_compatible will return False.

    This was created for a legacy codebase without type annotations. If possible, prefer
    using NewTypes or other more explicit types.

    """

    varnames: List[str]

    def is_value_compatible(self, val: Value) -> bool:
        if not isinstance(val, VariableNameValue):
            return True
        return val == self

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


def flatten_values(val: Value) -> Iterable[Value]:
    """Flatten a MultiValuedValue into a single value.

    We don't need to do this recursively because the
    MultiValuedValue constructor applies this to its arguments.

    """
    if isinstance(val, MultiValuedValue):
        for subval in val.vals:
            yield subval
    else:
        yield val


def unite_values(*values: Value) -> Value:
    if not values:
        return UNRESOLVED_VALUE
    # Make sure order is consistent; conceptually this is a set but
    # sets have unpredictable iteration order.
    hashable_vals = OrderedDict()
    unhashable_vals = []
    uncomparable_vals = []
    for value in values:
        if isinstance(value, MultiValuedValue):
            subvals = value.vals
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
    if len(existing) == 1:
        return existing[0]
    else:
        return MultiValuedValue(existing)


def are_types_compatible(t1: type, t2: type) -> bool:
    """Special cases for type compatibility checks."""
    # As a special case, the Python type system treats int as
    # a subtype of float.
    if t1 is float and t2 is int:
        return True
    # Everything is compatible with mock objects
    if issubclass(t2, mock.NonCallableMock):
        return True
    return False


def boolean_value(value: Optional[Value]) -> Optional[bool]:
    """Given a Value, returns whether the object is statically known to be truthy.

    Returns None if its truth value cannot be determined.

    """
    if isinstance(value, KnownValue):
        try:
            return bool(value.val)
        except Exception:
            # Its __bool__ threw an exception. Just give up.
            return None
    return None


def extract_typevars(value: Value) -> Iterable["TypeVar"]:
    for val in value.walk_values():
        if isinstance(val, TypeVarValue):
            yield val.typevar


def _stringify_type(typ: type) -> str:
    try:
        if typ.__module__ == BUILTIN_MODULE:
            return typ.__name__
        elif hasattr(typ, "__qualname__"):
            return "%s.%s" % (typ.__module__, typ.__qualname__)
        else:
            return "%s.%s" % (typ.__module__, typ.__name__)
    except Exception:
        return repr(typ)
