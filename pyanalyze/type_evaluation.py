"""

Implementation of type evaluation.

"""

import ast
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import textwrap
from typing import (
    Any,
    Callable,
    Container,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .extensions import get_type_evaluation
from .value import (
    AnySource,
    AnyValue,
    CanAssignContext,
    CanAssignError,
    KnownValue,
    MultiValuedValue,
    Value,
    can_assign_and_used_any,
    unannotate,
    unite_values,
    unify_typevar_maps,
    TypeVarMap,
)

VarMap = Mapping[str, Value]


class InvalidEvaluation(Exception):
    pass


class Context:
    variables: VarMap
    set_variables: Container[str]
    can_assign_context: CanAssignContext

    def evaluate_type(self, __node: ast.AST) -> Value:
        raise NotImplementedError

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

    def reverse(self) -> "ConditionReturn":
        return ConditionReturn(
            left_varmap=self.right_varmap,
            right_varmap=self.left_varmap,
            is_any_match=self.is_any_match,
        )


@dataclass
class ConditionEvaluator(ast.NodeVisitor):
    ctx: Context

    def visit_Call(self, node: ast.Call) -> ConditionReturn:
        if not isinstance(node.func, ast.Name):
            raise InvalidEvaluation("Unexpected call")
        name = node.func.id
        if name == "is_provided":
            if node.keywords or len(node.args) != 1:
                raise InvalidEvaluation("is_provided() takes a single argument")
            if not isinstance(node.args[0], ast.Name):
                raise InvalidEvaluation("Argument to is_provided() must be a variable")
            variable = node.args[0].id
            match = variable in self.ctx.set_variables
            if match:
                return ConditionReturn(left_varmap={})
            else:
                return ConditionReturn(right_varmap={})
        elif name == "is_of_type":
            if node.keywords or len(node.args) != 2:
                raise InvalidEvaluation("is_of_type() takes two positional arguments")
            varname_node = node.args[0]
            typ = self.ctx.evaluate_type(node.args[1])
            return self.visit_is_of_type(varname_node, typ)
        else:
            raise InvalidEvaluation(f"Invalid function {name}")

    def visit_is_of_type(self, varname_node: ast.AST, typ: Value) -> ConditionReturn:
        if not isinstance(varname_node, ast.Name):
            raise InvalidEvaluation("First argument to is_of_type() must be a name")
        val = self.get_name(varname_node)
        can_assign, used_any = can_assign_and_used_any(
            typ, val, self.ctx.can_assign_context
        )
        if isinstance(can_assign, CanAssignError):
            triple = decompose_union(typ, val, self.ctx.can_assign_context)
            if triple is not None:
                _, used_any, remaining = triple
                return ConditionReturn(
                    left_varmap={varname_node.id: typ},
                    right_varmap={varname_node.id: remaining},
                    is_any_match=used_any,
                )
            return ConditionReturn(right_varmap={})
        else:
            return ConditionReturn(
                left_varmap={varname_node.id: typ}, is_any_match=used_any
            )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ConditionReturn:
        if isinstance(node.op, ast.Not):
            ret = self.visit(node.operand)
            return ret.reverse()
        else:
            raise InvalidEvaluation("Unsupported unary operation")

    def visit_Compare(self, node: ast.Compare) -> ConditionReturn:
        if len(node.ops) != 1:
            raise InvalidEvaluation("Chained comparison is unsupported")
        op = node.ops[0]
        right = node.comparators[0]
        if isinstance(op, (ast.Is, ast.IsNot)):
            if not isinstance(right, ast.NameConstant):
                raise InvalidEvaluation(
                    "is/is not are only supported with True, False, and None"
                )
            ret = self.visit_is_of_type(node.left, KnownValue(right.value))
            if isinstance(op, ast.IsNot):
                return ret.reverse()
            return ret
        elif isinstance(op, (ast.Eq, ast.NotEq)):
            operand = self.evaluate_literal(right)
            ret = self.visit_is_of_type(node.left, operand)
            if isinstance(op, ast.NotEq):
                return ret.reverse()
            return ret
        else:
            raise InvalidEvaluation("Unsupported comparison operator")

    def evaluate_literal(self, node: ast.expr) -> KnownValue:
        if isinstance(node, ast.NameConstant):
            return KnownValue(node.value)
        elif isinstance(node, ast.Num):
            return KnownValue(node.n)
        elif isinstance(node, (ast.Str, ast.Bytes)):
            return KnownValue(node.s)
        else:
            raise InvalidEvaluation("Only literals supported")

    def get_name(self, node: ast.Name) -> Value:
        try:
            return self.ctx.variables[node.id]
        except KeyError:
            raise InvalidEvaluation(f"Invalid variable {node.id}") from None


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
    if evaluation_func is None or not hasattr(evaluation_func, "__globals__"):
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


def decompose_union(
    expected_type: Value, parent_value: Value, ctx: CanAssignContext
) -> Optional[Tuple[TypeVarMap, bool, Value]]:
    value = unannotate(parent_value)
    if isinstance(value, MultiValuedValue):
        tv_maps = []
        remaining_values = []
        union_used_any = False
        for val in value.vals:
            can_assign, subval_used_any = can_assign_and_used_any(
                expected_type, val, ctx
            )
            if isinstance(can_assign, CanAssignError):
                remaining_values.append(val)
            else:
                if subval_used_any:
                    union_used_any = True
                tv_maps.append(can_assign)
        if tv_maps:
            tv_map = unify_typevar_maps(tv_maps)
            assert (
                remaining_values
            ), f"all union members matched between {expected_type} and {parent_value}"
            return tv_map, union_used_any, unite_values(*remaining_values)
    return None
