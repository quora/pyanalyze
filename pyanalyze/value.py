"""

Implementation of value classes, which represent values found while analyzing an AST.

"""

import collections.abc
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field, InitVar
import inspect
from itertools import chain
from types import FunctionType
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

from .safe import safe_isinstance
from .type_object import TypeObject

T = TypeVar("T")
# __builtin__ in Python 2 and builtins in Python 3
BUILTIN_MODULE = str.__module__
KNOWN_MUTABLE_TYPES = (list, set, dict, deque)

TypeVarMap = Mapping["TypeVar", "Value"]

_type_object_cache: Dict[type, TypeObject] = {}


class CanAssignContext:
    def get_additional_bases(self, typ: Union[type, super]) -> Set[type]:
        return set()

    def make_type_object(self, typ: Union[type, super]) -> TypeObject:
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
        return {}

    def get_signature(
        self, obj: object, is_asynq: bool = False
    ) -> Optional["pyanalyze.signature.Signature"]:
        return None


@dataclass(frozen=True)
class CanAssignError:
    message: str
    children: List["CanAssignError"] = field(default_factory=list)

    def display(self, depth: int = 2) -> str:
        result = f"{' ' * depth}{self.message}\n"
        result += "".join(child.display(depth=depth + 2) for child in self.children)
        return result

    def __str__(self) -> str:
        return self.display()


# Return value of CanAssign
CanAssign = Union[TypeVarMap, CanAssignError]


class Value:
    """Class that represents the value of a variable."""

    __slots__ = ()

    def is_type(self, typ: type) -> bool:
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

    def substitute_typevars(self, typevars: TypeVarMap) -> "Value":
        """Substitute the typevars in the map to produce a new Value."""
        return self

    def can_assign(self, other: "Value", ctx: CanAssignContext) -> CanAssign:
        """Whether other can be assigned to self.

        If yes, return a (possibly empty) map with TypeVar substitutions. If not,
        return None.

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

    def is_assignable(self, other: "Value", ctx: CanAssignContext) -> bool:
        """Similar to can_assign but just returns a bool."""
        return isinstance(self.can_assign(other, ctx), dict)


@dataclass(frozen=True)
class UnresolvedValue(Value):
    """Value that we cannot resolve further."""

    def __str__(self) -> str:
        return "Any"

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        return {}  # Always allowed


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
class KnownValue(Value):
    """Variable with a known value."""

    val: Any

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

    def is_type(self, typ: type) -> bool:
        return isinstance(self.get_method(), typ)

    def get_type(self) -> type:
        return type(self.get_method())

    def get_type_value(self) -> Value:
        return KnownValue(type(self.get_method()))

    def substitute_typevars(self, typevars: TypeVarMap) -> "Value":
        """Substitute the typevars in the map to produce a new Value."""
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

    Corresponds to typing.NewType.

    """

    name: str
    newtype: Any

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
    """A TypedValue representing a generic.

    For example, List[int] is represented as GenericValue(list, [TypedValue(int)]).

    """

    args: Tuple[Value, ...]

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
    """A TypedValue representing a sequence whose members are not completely known.

    For example, the expression [int(self.foo)] may be typed as
    SequenceIncompleteValue(list, [TypedValue(int)])

    """

    members: Tuple[Value, ...]

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
    items: Dict[str, Value]

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
    """A TypedValue representing an async task.

    value is the value that the task object wraps.

    """

    value: Value

    def __init__(self, typ: type, value: Value) -> None:
        super(AsyncTaskIncompleteValue, self).__init__(typ, (value,))
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
        # TODO: unify with _get_argspec_from_value() in NameCheckVisitor
        signature = None
        if isinstance(other, CallableValue):
            signature = other.signature
        elif isinstance(other, KnownValue):
            signature = ctx.get_signature(
                other.val, is_asynq=hasattr(other.val, "asynq")
            )
        elif isinstance(other, SubclassValue) and isinstance(other.typ, TypedValue):
            signature = ctx.get_signature(other.typ.typ)
        elif isinstance(other, UnboundMethodValue):
            method = other.get_method()
            if method is not None:
                unbound_signature = ctx.get_signature(method)
                maybe_bound = pyanalyze.signature.make_bound_method(
                    unbound_signature, other.typ
                )
                if isinstance(maybe_bound, pyanalyze.signature.BoundMethodSignature):
                    signature = maybe_bound.get_signature()
                else:
                    signature = maybe_bound
        elif isinstance(other, TypedValue):
            typ = other.typ
            if typ is collections.abc.Callable or typ is FunctionType:
                return {}
            if not hasattr(typ, "__call__") or (
                getattr(typ.__call__, "__objclass__", None) is type
                and not issubclass(typ, type)
            ):
                return CanAssignError(f"{other} is not a callable type")
            call_fn = typ.__call__
            unbound_signature = ctx.get_signature(call_fn)
            bound_method = pyanalyze.signature.make_bound_method(
                unbound_signature, other
            )
            if bound_method is not None:
                signature = bound_method.get_signature()
        if signature is not None:
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
    """Value that is either a type or its subclass."""

    typ: Union[TypedValue, "TypeVarValue"]

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return self.make(self.typ.substitute_typevars(typevars))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        yield from self.typ.walk_values()

    def is_type(self, typ: type) -> bool:
        if isinstance(self.typ, TypedValue):
            try:
                return issubclass(self.typ.typ, typ)
            except Exception:
                return False
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
    """Variable for which multiple possible values have been recorded."""

    raw_vals: InitVar[Iterable[Value]]
    vals: Tuple[Value, ...] = field(init=False)

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


@dataclass(frozen=True)
class ReferencingValue(Value):
    """Value that is a reference to another value (used to implement globals)."""

    scope: Any
    name: str

    def __str__(self) -> str:
        return f"<reference to {self.name}>"


@dataclass(frozen=True)
class TypeVarValue(Value):
    """Value representing a type variable."""

    typevar: TypeVar

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return typevars.get(self.typevar, self)

    def can_assign(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        return {self.typevar: other}

    def get_fallback_value(self) -> Value:
        # TODO: support bounds and bases here to do something smarter
        return UNRESOLVED_VALUE


class Extension:
    __slots__ = ()

    def substitute_typevars(self, typevars: TypeVarMap) -> "Extension":
        return self

    def walk_values(self) -> Iterable[Value]:
        return []


@dataclass
class ParameterTypeGuardExtension(Extension):
    varname: str
    guarded_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        guarded_type = self.guarded_type.substitute_typevars(typevars)
        return ParameterTypeGuardExtension(self.varname, guarded_type)

    def walk_values(self) -> Iterable[Value]:
        yield from self.guarded_type.walk_values()


@dataclass
class TypeGuardExtension(Extension):
    guarded_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        guarded_type = self.guarded_type.substitute_typevars(typevars)
        return TypeGuardExtension(guarded_type)

    def walk_values(self) -> Iterable[Value]:
        yield from self.guarded_type.walk_values()


@dataclass
class HasAttrGuardExtension(Extension):
    """Returned by a function to indicate that varname has the given attribute."""

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
    """Attached to a function to indicate that it has the given attribute.

    These cannot be created directly from user code, only through the
    HasAttrGuard mechanism. The main reason is that in code like this:

        def f(x: Annotated[object, HasAttr["y", int]]) -> None:
            return x.y

    We would correctly type check the function body, but we currently
    have no way to enforce that it is only called with arguments that
    obey the constraint. If we fix that, we might as well fully implement
    Protocols.

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
    like list.extend. After code like `lst = [1, 2]; lst.extend([i for in range(5)])`
    we may end up inferring a type like `List[Literal[0, 1, 2, 3, 4]]`, but that is
    too narrow and leads to false positives if later code puts a different int in
    the list. If the generic argument is instead annotated with WeakExtension, we
    widen the type to accommodate later appends.

    The TestGenericMutators.test_weak_value test case is an example.

    """


@dataclass(frozen=True)
class AnnotatedValue(Value):
    """Value representing a PEP 593 Annotated object."""

    value: Value
    metadata: Sequence[Union[Value, Extension]]

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
        for data in self.metadata:
            if isinstance(data, typ):
                yield data

    def has_metadata_of_type(self, typ: Type[Extension]) -> bool:
        return any(isinstance(data, typ) for data in self.metadata)

    def __str__(self) -> str:
        return f"Annotated[{self.value}, {', '.join(map(str, self.metadata))}]"


@dataclass(frozen=True)
class VariableNameValue(Value):
    """Value that is stored in a variable associated with a particular kind of value.

    For example, any variable named 'uid' will get resolved into a VariableNameValue of type uid,
    and if it gets passed into a function that takes an argument called 'aid',
    can_assign will return None.

    This was created for a legacy codebase without type annotations. If possible, prefer
    using NewTypes or other more explicit types.

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
    """Flatten a MultiValuedValue into a single value.

    We don't need to do this recursively because the
    MultiValuedValue constructor applies this to its arguments.

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
    - None if the argument is not iterable
    - A sequence of Values if we know the exact types in the iterable
    - A single Value if we just know that the iterable contains this
      value, but not the precise number of them.

    Examples:
    - tuple[int, str] -> (int, str)
    - tuple[int, ...] -> int
    - int -> None

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

    If post_starred_length is None, return a list of target_length values, or CanAssignError
    if value is not an iterable of the expected length. If post_starred_length is not None,
    return a list of target_length + 1 + post_starred_length values. This implements
    unpacking like `a, b, *c, d = ...`.

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
    """Stringify arbitrary Python objects such as methods and types."""
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
