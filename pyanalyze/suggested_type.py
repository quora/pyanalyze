"""

Suggest types for untyped code.

"""
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .safe import safe_getattr, safe_isinstance
from .error_code import ErrorCode
from .node_visitor import Failure, ErrorContext
from .value import (
    NO_RETURN_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignError,
    GenericValue,
    KnownValue,
    SequenceIncompleteValue,
    SubclassValue,
    TypedDictValue,
    TypedValue,
    Value,
    MultiValuedValue,
    VariableNameValue,
    replace_known_sequence_value,
    stringify_object,
    unite_values,
)
from .signature import Signature

CallArgs = Mapping[str, Value]
FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


@dataclass
class CallableData:
    node: FunctionNode
    ctx: ErrorContext
    sig: Signature
    calls: List[CallArgs] = field(default_factory=list)

    def check(self) -> Iterator[Failure]:
        if not self.calls:
            return
        for param in _extract_params(self.node):
            if param.annotation is not None:
                continue
            sig_param = self.sig.parameters.get(param.arg)
            if sig_param is None or not isinstance(sig_param.annotation, AnyValue):
                continue  # e.g. inferred type for self
            all_values = [call[param.arg] for call in self.calls]
            all_values = [prepare_type(v) for v in all_values]
            all_values = [v for v in all_values if not isinstance(v, AnyValue)]
            if not all_values:
                continue
            suggested = unite_values(*all_values)
            if not should_suggest_type(suggested):
                continue
            detail, metadata = display_suggested_type(suggested)
            failure = self.ctx.show_error(
                param,
                f"Suggested type for parameter {param.arg}",
                ErrorCode.suggested_parameter_type,
                detail=detail,
                # Otherwise we record it twice in tests. We should ultimately
                # refactor error tracking to make it less hacky for things that
                # show errors outside of files.
                save=False,
                extra_metadata=metadata,
            )
            if failure is not None:
                yield failure


@dataclass
class CallableTracker:
    callable_to_data: Dict[object, CallableData] = field(default_factory=dict)
    callable_to_calls: Dict[object, List[CallArgs]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record_callable(
        self, node: FunctionNode, callable: object, sig: Signature, ctx: ErrorContext
    ) -> None:
        """Record when we encounter a callable."""
        self.callable_to_data[callable] = CallableData(node, ctx, sig)

    def record_call(self, callable: object, arguments: Mapping[str, Value]) -> None:
        """Record the actual arguments passed in in a call."""
        self.callable_to_calls[callable].append(arguments)

    def check(self) -> List[Failure]:
        failures = []
        for callable, calls in self.callable_to_calls.items():
            if callable in self.callable_to_data:
                data = self.callable_to_data[callable]
                data.calls += calls
                failures += data.check()
        return failures


def display_suggested_type(value: Value) -> Tuple[str, Optional[Dict[str, Any]]]:
    value = prepare_type(value)
    if isinstance(value, MultiValuedValue) and value.vals:
        cae = CanAssignError("Union", [CanAssignError(str(val)) for val in value.vals])
    else:
        cae = CanAssignError(str(value))
    # If the type is simple enough, add extra_metadata for autotyping to apply.
    if isinstance(value, TypedValue) and type(value) is TypedValue:
        # For now, only for exactly TypedValue
        if value.typ is FunctionType:
            # It will end up suggesting builtins.function, which doesn't
            # exist, and we should be using a Callable type instead anyway.
            metadata = None
        else:
            suggested_type = stringify_object(value.typ)
            imports = []
            if isinstance(value.typ, str):
                if "." in value.typ:
                    imports.append(value.typ)
            elif safe_getattr(value.typ, "__module__", None) != "builtins":
                imports.append(suggested_type.split(".")[0])
            metadata = {"suggested_type": suggested_type, "imports": imports}
    else:
        metadata = None
    return str(cae), metadata


def should_suggest_type(value: Value) -> bool:
    # Literal[<some function>] isn't useful. In the future we should suggest a
    # Callable type.
    if isinstance(value, KnownValue) and isinstance(value.val, FunctionType):
        return False
    # These generally aren't useful.
    if isinstance(value, TypedValue) and value.typ in (FunctionType, type):
        return False
    if isinstance(value, AnyValue):
        return False
    if isinstance(value, MultiValuedValue) and len(value.vals) > 5:
        # Big unions probably aren't useful
        return False
    # We emptied out a Union
    if value is NO_RETURN_VALUE:
        return False
    return True


def prepare_type(value: Value) -> Value:
    """Simplify a type to turn it into a suggestion."""
    if isinstance(value, AnnotatedValue):
        return prepare_type(value.value)
    elif isinstance(value, SequenceIncompleteValue):
        if value.typ is tuple:
            return SequenceIncompleteValue(
                tuple, [prepare_type(elt) for elt in value.members]
            )
        else:
            return GenericValue(value.typ, [prepare_type(arg) for arg in value.args])
    elif isinstance(value, (TypedDictValue, CallableValue)):
        return value
    elif isinstance(value, GenericValue):
        # TODO maybe turn DictIncompleteValue into TypedDictValue?
        return GenericValue(value.typ, [prepare_type(arg) for arg in value.args])
    elif isinstance(value, VariableNameValue):
        return AnyValue(AnySource.unannotated)
    elif isinstance(value, KnownValue):
        if value.val is None:
            return value
        elif safe_isinstance(value.val, type):
            return SubclassValue(TypedValue(value.val))
        elif callable(value.val):
            return value  # TODO get the signature instead and return a CallableValue?
        value = replace_known_sequence_value(value)
        if isinstance(value, KnownValue):
            return TypedValue(type(value.val))
        else:
            return prepare_type(value)
    elif isinstance(value, MultiValuedValue):
        vals = [prepare_type(subval) for subval in value.vals]
        # Throw out Anys
        vals = [val for val in vals if not isinstance(val, AnyValue)]
        type_literals: List[Tuple[Value, type]] = []
        rest: List[Value] = []
        for subval in vals:
            if (
                isinstance(subval, SubclassValue)
                and isinstance(subval.typ, TypedValue)
                and safe_isinstance(subval.typ.typ, type)
            ):
                type_literals.append((subval, subval.typ.typ))
            else:
                rest.append(subval)
        if type_literals:
            shared_type = get_shared_type([typ for _, typ in type_literals])
            if shared_type is object:
                type_val = TypedValue(type)
            else:
                type_val = SubclassValue(TypedValue(shared_type))
            return unite_values(type_val, *rest)
        return unite_values(*[v for v, _ in type_literals], *rest)
    else:
        return value


def get_shared_type(types: Sequence[type]) -> type:
    mros = [t.mro() for t in types]
    first, *rest = mros
    rest_sets = [set(mro) for mro in rest]
    for candidate in first:
        if all(candidate in mro for mro in rest_sets):
            return candidate
    assert False, "should at least have found object"


# We exclude *args and **kwargs by default because it's not currently possible
# to give useful types for them.
def _extract_params(
    node: FunctionNode, *, include_var: bool = False
) -> Iterator[ast.arg]:
    yield from node.args.args
    if include_var and node.args.vararg is not None:
        yield node.args.vararg
    yield from node.args.kwonlyargs
    if include_var and node.args.kwarg is not None:
        yield node.args.kwarg
