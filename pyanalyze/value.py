from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

"""

Implementation of value classes, which represent values found while analyzing an AST.

"""

import attr
from collections import OrderedDict
import inspect
from itertools import chain
from typing import Dict, List, Optional, Tuple

try:
    from unittest import mock
except ImportError:
    import mock

# __builtin__ in Python 2 and builtins in Python 3
BUILTIN_MODULE = str.__module__


class Value(object):
    """Class that represents the value of a variable."""

    def is_value_compatible(self, val):
        """Returns whether the given value is compatible with this value.

        val must be a more precise type than (or the same type as) self.

        Callers should always go through NameCheckVisitor.is_value_compatible,
        because this function may raise errors since it can call into user code.

        """
        return True

    def is_type(self, typ):
        """Returns whether this value is an instance of the given type."""
        return False

    def get_type(self):
        """Returns the type of this value, or None if it is not known."""
        return None

    def get_type_value(self):
        """Return the type of this object as used for dunder lookups."""
        return self


@attr.s(frozen=True)
class UnresolvedValue(Value):
    """Value that we cannot resolve further."""

    def __str__(self):
        return "Any"


UNRESOLVED_VALUE = UnresolvedValue()


@attr.s(frozen=True)
class UninitializedValue(Value):
    """Value for variables that have not been initialized.

    Usage of variables with this value should be an error.

    """

    def __str__(self):
        return "<uninitialized>"


UNINITIALIZED_VALUE = UninitializedValue()


@attr.s(frozen=True)
class NoReturnValue(Value):
    """Value that indicates that a function will never return."""

    def __str__(self):
        return "NoReturn"

    def is_value_compatible(self, val):
        # You can't assign anything to NoReturn
        return False


NO_RETURN_VALUE = NoReturnValue()


@attr.s(frozen=True, hash=False, cmp=False)
class KnownValue(Value):
    """Variable with a known value."""

    val = attr.ib()
    source_node = attr.ib(default=None, repr=False, hash=False, cmp=False)

    def is_value_compatible(self, val):
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

    def is_type(self, typ):
        return isinstance(self.val, typ)

    def get_type(self):
        return type(self.val)

    def __eq__(self, other):
        return (
            isinstance(other, KnownValue)
            and type(self.val) is type(other.val)
            and self.val == other.val
        )

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        # For Python 2, make sure b'' and u'' are hashed differently.
        return hash((type(self.val), self.val))

    def __str__(self):
        if self.val is None:
            return "None"
        else:
            return "Literal[%r]" % (self.val,)

    def get_type_value(self):
        return KnownValue(type(self.val))


@attr.s(frozen=True)
class UnboundMethodValue(Value):
    """Value that represents an unbound method.

    That is, we know that this value is this method, but we don't have the instance it is called on.

    """

    attr_name = attr.ib(type=str)
    typ = attr.ib(type=type)
    secondary_attr_name = attr.ib(default=None, type=Optional[str])

    def get_method(self):
        """Returns the method object for this UnboundMethodValue."""
        try:
            method = getattr(self.typ, self.attr_name)
            if self.secondary_attr_name is not None:
                method = getattr(method, self.secondary_attr_name)
            # don't use unbound methods in py2
            if inspect.ismethod(method) and method.__self__ is None:
                method = method.__func__
            return method
        except AttributeError:
            return None

    def is_type(self, typ):
        return isinstance(self.get_method(), typ)

    def get_type(self):
        return type(self.get_method())

    def get_type_value(self):
        return KnownValue(type(self.get_method()))

    def __str__(self):
        return "<method %s%s on %s>" % (
            self.attr_name,
            ".%s" % (self.secondary_attr_name,) if self.secondary_attr_name else "",
            _stringify_type(self.typ),
        )


@attr.s(hash=True)
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

    typ = attr.ib()

    def is_value_compatible(self, val):
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

    def is_value_compatible_thrift_enum(self, val):
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

    def is_type(self, typ):
        return issubclass(self.typ, typ)

    def get_type(self):
        return self.typ

    def get_type_value(self):
        return KnownValue(self.typ)

    def __str__(self):
        return _stringify_type(self.typ)


@attr.s(hash=True, init=False)
class NewTypeValue(TypedValue):
    """A wrapper around an underlying type.

    Corresponds to typing.NewType.

    """

    name = attr.ib(type=str)
    newtype = attr.ib()

    def __init__(self, newtype):
        super(NewTypeValue, self).__init__(newtype.__supertype__)
        self.name = newtype.__name__
        self.newtype = newtype

    def is_value_compatible(self, val):
        if isinstance(val, NewTypeValue):
            return self.newtype is val.newtype
        else:
            return super(NewTypeValue, self).is_value_compatible(val)

    def __str__(self):
        return "NewType(%r, %s)" % (self.name, _stringify_type(self.typ))


@attr.s(hash=True)
class GenericValue(TypedValue):
    """A TypedValue representing a generic.

    For example, List[int] is represented as GenericValue(list, [TypedValue(int)]).

    """

    args = attr.ib(converter=tuple, type=Tuple[Value, ...])

    def __str__(self):
        if self.typ is tuple:
            args = list(self.args) + ["..."]
        else:
            args = self.args
        return "%s[%s]" % (
            _stringify_type(self.typ),
            ", ".join(str(arg) for arg in args),
        )

    def is_value_compatible(self, val):
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

    def get_arg(self, index):
        try:
            return self.args[index]
        except IndexError:
            return UNRESOLVED_VALUE


@attr.s(hash=True, init=False)
class SequenceIncompleteValue(GenericValue):
    """A TypedValue representing a sequence whose members are not completely known.

    For example, the expression [int(self.foo)] may be typed as
    SequenceValue(list, [TypedValue(int)])

    """

    members = attr.ib(type=Tuple[Value, ...])

    def __init__(self, typ, members):
        if members:
            args = (unite_values(*members),)
        else:
            args = (UNRESOLVED_VALUE,)
        super(SequenceIncompleteValue, self).__init__(typ, args)
        self.members = tuple(members)

    def is_value_compatible(self, val):
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

    def __str__(self):
        if self.typ is tuple:
            return "tuple[%s]" % (", ".join(str(m) for m in self.members))
        return "<%s containing [%s]>" % (
            _stringify_type(self.typ),
            ", ".join(map(str, self.members)),
        )


@attr.s(hash=True, init=False)
class DictIncompleteValue(GenericValue):
    """A TypedValue representing a dictionary whose keys and values are not completely known.

    For example, the expression {'foo': int(self.bar)} may be typed as
    DictIncompleteValue([(KnownValue('foo'), TypedValue(int))]).

    """

    items = attr.ib(type=List[Tuple[Value, Value]])

    def __init__(self, items):
        if items:
            key_type = unite_values(*[key for key, _ in items])
            value_type = unite_values(*[value for _, value in items])
        else:
            key_type = value_type = UNRESOLVED_VALUE
        super(DictIncompleteValue, self).__init__(dict, (key_type, value_type))
        self.items = items

    def __str__(self):
        return "<%s containing {%s}>" % (
            _stringify_type(self.typ),
            ", ".join("%s: %s" % (key, value) for key, value in self.items),
        )


@attr.s(init=False)
class TypedDictValue(GenericValue):
    items = attr.ib(type=Dict[str, Value])

    def __init__(self, items):
        if items:
            value_type = unite_values(*items.values())
        else:
            value_type = UNRESOLVED_VALUE
        super(TypedDictValue, self).__init__(dict, (TypedValue(str), value_type))
        self.items = items

    def is_value_compatible(self, val):
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

    def __str__(self):
        items = ['"%s": %s' % (key, value) for key, value in self.items.items()]
        return "TypedDict({%s})" % ", ".join(items)

    def __hash__(self):
        return hash(tuple(sorted(self.items)))


@attr.s(hash=True, init=False)
class AsyncTaskIncompleteValue(GenericValue):
    """A TypedValue representing an async task.

    value is the value that the task object wraps.

    """

    value = attr.ib(type=Value)

    def __init__(self, typ, value):
        super(AsyncTaskIncompleteValue, self).__init__(typ, (value,))
        self.value = value


@attr.s(frozen=True)
class AwaitableIncompleteValue(Value):
    """A Value representing a Python 3 awaitable, e.g. a coroutine object."""

    value = attr.ib(type=Value)

    def get_type_value(self):
        return UNRESOLVED_VALUE

    def __str__(self):
        return "Awaitable[%s]" % (self.value,)


@attr.s(frozen=True)
class SubclassValue(Value):
    """Value that is either a type or its subclass."""

    typ = attr.ib(type=type)

    def is_type(self, typ):
        try:
            return issubclass(self.typ, typ)
        except Exception:
            return False

    def is_value_compatible(self, val):
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

    def get_type(self):
        return type(self.typ)

    def get_type_value(self):
        return KnownValue(type(self.typ))

    def __str__(self):
        return "Type[%s]" % (_stringify_type(self.typ),)


@attr.s(cmp=False, frozen=True, hash=True)
class MultiValuedValue(Value):
    """Variable for which multiple possible values have been recorded."""

    vals = attr.ib(
        converter=lambda vals: tuple(
            chain.from_iterable(flatten_values(val) for val in vals)
        )
    )

    def is_value_compatible(self, val):
        if isinstance(val, MultiValuedValue):
            return all(
                any(subval.is_value_compatible(other_subval) for subval in self.vals)
                for other_subval in val.vals
            )
        else:
            return any(subval.is_value_compatible(val) for subval in self.vals)

    def get_type_value(self):
        return MultiValuedValue([val.get_type_value() for val in self.vals])

    def __eq__(self, other):
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

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "Union[%s]" % ", ".join(map(str, self.vals))


@attr.s(frozen=True)
class ReferencingValue(Value):
    """Value that is a reference to another value (used to implement globals)."""

    scope = attr.ib()
    name = attr.ib(type=str)

    def __str__(self):
        return "<reference to %s>" % (self.name,)


# TODO(jelle): merge with NewTypeValuue
@attr.s(frozen=True)
class VariableNameValue(Value):
    """Value that is stored in a variable associated with a particular kind of value.

    For example, any variable named 'uid' will get resolved into a VariableNameValue of type uid,
    and if it gets passed into a function that takes an argument called 'aid',
    is_value_compatible will return False.

    """

    varnames = attr.ib(type=List[str])

    def is_value_compatible(self, val):
        if not isinstance(val, VariableNameValue):
            return True
        return val == self

    def __str__(self):
        return "<variable name: %s>" % ", ".join(self.varnames)

    @classmethod
    def from_varname(cls, varname, varname_map):
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


def flatten_values(val):
    """Flatten a MultiValuedValue into a single value.

    We don't need to do this recursively because the
    MultiValuedValue constructor applies this to its arguments.

    """
    if isinstance(val, MultiValuedValue):
        for subval in val.vals:
            yield subval
    else:
        yield val


def unite_values(*values):
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


def are_types_compatible(t1, t2):
    """Special cases for type compatibility checks."""
    # As a special case, the Python type system treats int as
    # a subtype of float.
    if t1 is float and t2 is int:
        return True
    # Everything is compatible with mock objects
    if issubclass(t2, mock.NonCallableMock):
        return True
    return False


def boolean_value(value):
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


def _stringify_type(typ):
    try:
        if typ.__module__ == BUILTIN_MODULE:
            return typ.__name__
        elif hasattr(typ, "__qualname__"):
            return "%s.%s" % (typ.__module__, typ.__qualname__)
        else:
            return "%s.%s" % (typ.__module__, typ.__name__)
    except Exception:
        return repr(typ)
