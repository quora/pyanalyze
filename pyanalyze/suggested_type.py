"""

Suggest types for untyped code.

"""
import ast
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Mapping, Union

from .error_code import ErrorCode
from .node_visitor import Failure
from .value import AnyValue, CanAssignError, Value, MultiValuedValue, unite_values
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
            suggested = display_suggested_type(unite_values(*all_values))
            failure = self.ctx.show_error(
                param,
                f"Suggested type for parameter {param.arg}",
                ErrorCode.suggested_parameter_type,
                detail=suggested,
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
    value = value.simplify()
    if isinstance(value, MultiValuedValue) and value.vals:
        cae = CanAssignError("Union", [CanAssignError(str(val)) for val in value.vals])
    else:
        cae = CanAssignError(str(value))
    return str(cae)


def _extract_params(node: FunctionNode) -> Iterator[ast.arg]:
    yield from node.args.args
    if node.args.vararg is not None:
        yield node.args.vararg
    yield from node.args.kwonlyargs
    if node.args.kwarg is not None:
        yield node.args.kwarg
