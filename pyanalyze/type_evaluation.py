"""

Implementation of type evaluation.

"""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import textwrap
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, Union

from .extensions import get_type_evaluation
from .value import AnySource, AnyValue, CanAssignError, Value, unite_values

# None means the variable is unset

VarMap = Mapping[str, Optional[Value]]


class InvalidEvaluation(Exception):
    pass


class Context:
    variables: VarMap
    evaluate_type: Callable[[ast.AST], Value]

    @contextmanager
    def narrow_variables(self, varmap: Optional[VarMap]) -> Iterator[None]:
        if varmap is None:
            yield
            return
        old_varmap = self.variables
        new_varmap = {**old_varmap, **varmap}
        try:
            self.variables = new_varmap
            yield
        finally:
            self.variables = old_varmap


@dataclass
class Evaluator:
    func: Callable[..., Any]
    node: ast.FunctionDef
    globals: Mapping[str, object]


@dataclass
class UnionCombinedReturn:
    left: "EvalReturn"
    right: "EvalReturn"


@dataclass
class AnyCombinedReturn:
    left: "EvalReturn"
    right: "EvalReturn"


EvalReturn = Union[None, Value, CanAssignError, UnionCombinedReturn, AnyCombinedReturn]


def may_be_none(ret: EvalReturn) -> bool:
    if ret is None:
        return True
    elif isinstance(ret, (Value, CanAssignError)):
        return False
    else:
        return may_be_none(ret.left) or may_be_none(ret.right)


@dataclass
class ConditionReturn:
    # These are None if there is no match, and a (possibly empty)
    # map of new variable values if there is a match.
    left_varmap: Optional[VarMap] = None
    right_varmap: Optional[VarMap] = None
    is_any_match: bool = False


@dataclass
class ConditionEvaluator(ast.NodeVisitor):
    ctx: Context

    def visit_Call(self, node: ast.Call) -> ConditionReturn:
        if not isinstance(node.func, ast.Name):
            raise InvalidEvaluation("Unexpected call")
        name = node.func.id
        if name == "is_set":
            if node.keywords or len(node.args) != 1:
                raise InvalidEvaluation("is_set() takes a single argument")
            if not isinstance(node.args[0], ast.Name):
                raise InvalidEvaluation("Argument to is_set() must be a variable")
            variable = node.args[0].id
            if variable not in self.ctx.variables:
                raise InvalidEvaluation(f"{variable} is not a variable")
            match = self.ctx.variables[variable] is not None
            if match:
                return ConditionReturn(left_varmap={})
            else:
                return ConditionReturn(right_varmap={})
        elif name == "isinstance":
            raise NotImplementedError
        else:
            raise InvalidEvaluation(f"Invalid function {name}")


@dataclass
class TypeEvaluator(ast.NodeVisitor):
    ctx: Context

    def visit_FunctionDef(self, node: ast.FunctionDef) -> EvalReturn:
        return self.visit_block(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> EvalReturn:
        return self.visit_block(node.body)

    def visit_block(self, statements: Sequence[ast.stmt]) -> EvalReturn:
        for stmt in statements:
            result = self.visit(stmt)
            if not may_be_none(result):
                return result
        return None

    def visit_Return(self, node: ast.Return) -> EvalReturn:
        if node.value is None:
            raise InvalidEvaluation("return statement must have a value")
        return self.ctx.evaluate_type(node.value)

    def visit_Raise(self, node: ast.Raise) -> EvalReturn:
        if (
            not isinstance(node.exc, ast.Call)
            or not isinstance(node.exc.func, ast.Name)
            or node.exc.func.id != "Exception"
            or node.exc.keywords
            or len(node.exc.args) != 1
            or not isinstance(node.exc.args[0], ast.Str)
        ):
            raise InvalidEvaluation(
                "raise statement must be of the form 'raise Exception(message)'"
            )
        return CanAssignError(node.exc.args[0].s)

    def visit_If(self, node: ast.If) -> EvalReturn:
        visitor = ConditionEvaluator(self.ctx)
        condition = visitor.visit(node.test)
        if condition.is_any_match:
            with self.ctx.narrow_variables(condition.left_varmap):
                left_result = self.visit_block(node.body)
            with self.ctx.narrow_variables(condition.right_varmap):
                right_result = self.visit_block(node.orelse)
            return AnyCombinedReturn(left_result, right_result)
        else:
            if condition.left_varmap is not None:
                with self.ctx.narrow_variables(condition.left_varmap):
                    left_result = self.visit_block(node.body)
            else:
                left_result = None
            if condition.right_varmap is not None:
                with self.ctx.narrow_variables(condition.right_varmap):
                    right_result = self.visit_block(node.orelse)
            else:
                right_result = None
            if condition.left_varmap is not None:
                if condition.right_varmap is not None:
                    return UnionCombinedReturn(left_result, right_result)
                else:
                    return left_result
            else:
                if condition.right_varmap is not None:
                    return right_result
                else:
                    raise InvalidEvaluation("Condition must either match or not match")


def evaluate(evaluator: Evaluator, ctx: Context) -> Union[Value, CanAssignError]:
    visitor = TypeEvaluator(ctx)
    try:
        result = visitor.visit(evaluator.node)
        return _evaluate_ret(result)
    except InvalidEvaluation as e:
        return CanAssignError(
            "Internal error in type evaluator", [CanAssignError(str(e))]
        )


def _evaluate_ret(ret: EvalReturn) -> Union[Value, CanAssignError]:
    if ret is None:
        raise InvalidEvaluation("Evaluator failed to return")
    elif isinstance(ret, AnyCombinedReturn):
        left = _evaluate_ret(ret.left)
        right = _evaluate_ret(ret.right)

        # If one branch is an error, pick the other one
        if isinstance(left, CanAssignError):
            if isinstance(right, CanAssignError):
                return CanAssignError(children=[left, right])
            return right
        if isinstance(right, CanAssignError):
            return left

        if left == right:
            return left
        else:
            return AnyValue(AnySource.multiple_overload_matches)
    elif isinstance(ret, UnionCombinedReturn):
        left = _evaluate_ret(ret.left)
        if isinstance(left, CanAssignError):
            return left
        right = _evaluate_ret(ret.right)
        if isinstance(right, CanAssignError):
            return right
        return unite_values(left, right)
    else:
        return ret


def get_evaluator(func: Callable[..., Any]) -> Optional[Evaluator]:
    try:
        key = f"{func.__module__}.{func.__qualname__}"
    except AttributeError:
        return None
    evaluation_func = get_type_evaluation(key)
    if evaluation_func is None:
        return None
    lines, _ = inspect.getsourcelines(evaluation_func)
    code = textwrap.dedent("".join(lines))
    body = ast.parse(code)
    if not body.body:
        return None
    evaluator = body.body[0]
    if not isinstance(evaluator, ast.FunctionDef):
        raise InvalidEvaluation(f"Cannot locate {func}")
    return Evaluator(evaluation_func, evaluator, evaluation_func.__globals__)
