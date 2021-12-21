"""

Suggest types for untyped code.

"""
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Sequence, Union

from pyanalyze.safe import safe_isinstance

from .error_code import ErrorCode
from .node_visitor import Failure
from .value import (
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
    unite_values,
)
from .reexport import ErrorContext
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
            suggested = display_suggested_type(unite_values(*all_values))
            failure = self.ctx.show_error(
                param,
                f"Suggested type for parameter {param.arg}",
                ErrorCode.suggested_parameter_type,
                detail=suggested,
                # Otherwise we record it twice in tests. We should ultimately
                # refactor error tracking to make it less hacky for things that
                # show errors outside of files.
                save=False,
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


def display_suggested_type(value: Value) -> str:
    value = prepare_type(value)
    if isinstance(value, MultiValuedValue) and value.vals:
        cae = CanAssignError("Union", [CanAssignError(str(val)) for val in value.vals])
    else:
        cae = CanAssignError(str(value))
    return str(cae)


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
        if value.val is None or safe_isinstance(value.val, type):
            return value
        elif callable(value.val):
            return value  # TODO get the signature instead and return a CallableValue?
        value = replace_known_sequence_value(value)
        if isinstance(value, KnownValue):
            return TypedValue(type(value.val))
        else:
            return prepare_type(value)
    elif isinstance(value, MultiValuedValue):
        vals = [prepare_type(subval) for subval in value.vals]
        type_literals = [
            v
            for v in vals
            if isinstance(v, KnownValue) and safe_isinstance(v.val, type)
        ]
        if len(type_literals) > 1:
            types = [v.val for v in type_literals if isinstance(v.val, type)]
            shared_type = get_shared_type(types)
            type_val = SubclassValue(TypedValue(shared_type))
            others = [
                v
                for v in vals
                if not isinstance(v, KnownValue) or not safe_isinstance(v.val, type)
            ]
            return unite_values(type_val, *others)
        return unite_values(*vals)
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


def _extract_params(node: FunctionNode) -> Iterator[ast.arg]:
    yield from node.args.args
    if node.args.vararg is not None:
        yield node.args.vararg
    yield from node.args.kwonlyargs
    if node.args.kwarg is not None:
        yield node.args.kwarg
