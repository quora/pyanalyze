"""

Implementation of type evaluation.

"""

import ast
from contextlib import contextmanager
import contextlib
from dataclasses import dataclass, field
import inspect
import operator
import qcore
import sys
import textwrap
from typing_extensions import Literal
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from .extensions import get_type_evaluation
from .value import (
    NO_RETURN_VALUE,
    AnySource,
    AnyValue,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    KnownValue,
    MultiValuedValue,
    Value,
    flatten_values,
    unannotate,
    unite_values,
    unify_typevar_maps,
    TypeVarMap,
)

ARGS = qcore.MarkerObject("*args")
KWARGS = qcore.MarkerObject("**kwargs")
DEFAULT = qcore.MarkerObject("default")
UNKNOWN = qcore.MarkerObject("unknown")

# How a bound argument was filled: int for a positional arg,
# str for a keyword arg, DEFAULT for a parameter filled from the
# default, ARGS for a parameter filled from *args, KWARGS for **kwargs,
# UNKNOWN if there are multiple possibilities.
Position = Union[int, str, Literal[DEFAULT, ARGS, KWARGS, UNKNOWN]]

VarMap = Mapping[str, Value]


class Condition:
    def display(self, negated: bool = False) -> CanAssignError:
        raise NotImplementedError


class NullCondition(Condition):
    def display(self, negated: bool = False) -> CanAssignError:
        return CanAssignError()


@dataclass
class ConditionList(Condition):
    children: Sequence[Condition]

    def display(self, negated: bool = False) -> CanAssignError:
        return CanAssignError(
            children=[c.display(negated=negated) for c in self.children]
        )


@dataclass
class NotCondition(Condition):
    condition: Condition

    def display(self, negated: bool = False) -> CanAssignError:
        return self.condition.display(negated=not negated)


@dataclass
class ArgumentKindCondition(Condition):
    argument: str
    function: Literal["is_provided", "is_positional", "is_keyword"]

    def display(self, negated: bool = False) -> CanAssignError:
        maybe_not = " not" if negated else ""
        text = f"Argument {self.argument} was{maybe_not} provided"
        if self.function == "is_positional":
            return CanAssignError(f"{text} as a positional argument")
        elif self.function == "is_keyword":
            return CanAssignError(f"{text} as a keyword argument")
        else:
            return CanAssignError(text)


@dataclass
class PlatformCondition(Condition):
    actual: str
    op: Type[ast.cmpop]
    expected: object

    def display(self, negated: bool = False) -> CanAssignError:
        if negated:
            op = _OP_TO_DATA[self.op].negation
        else:
            op = self.op
        text = _OP_TO_DATA[op].text
        return CanAssignError(f"Platform ({self.actual}) {text} {self.expected}")


@dataclass
class VersionCondition(Condition):
    actual: Tuple[int, ...]
    op: Type[ast.cmpop]
    expected: object

    def _display_version(self, version: object) -> str:
        if isinstance(version, tuple):
            return ".".join(map(str, version))
        return str(version)

    def display(self, negated: bool = False) -> CanAssignError:
        if negated:
            op = _OP_TO_DATA[self.op].negation
        else:
            op = self.op
        text = _OP_TO_DATA[op].text
        return CanAssignError(
            f"Platform ({self._display_version(self.actual)})"
            f" {text} {self._display_version(self.expected)}"
        )


@dataclass
class IsOfTypeCondition(Condition):
    arg: str
    op: "_Operator"
    original_arg_type: Value
    remaining_type: Value
    expected_type: Value
    exclude_any: bool = True

    def display(self, negated: bool = False) -> CanAssignError:
        positive_text = _OP_TO_DATA[self.op].text
        negative_text = _OP_TO_DATA[_OP_TO_DATA[self.op].negation].text
        if negated:
            positive_text, negative_text = negative_text, positive_text

        original_arg_type = self.original_arg_type
        remaining_type = self.remaining_type
        matched_type = subtract_unions(original_arg_type, remaining_type)
        if negated:
            matched_type, remaining_type = remaining_type, matched_type

        if matched_type is NO_RETURN_VALUE:
            text = negative_text
            type_to_show = remaining_type
        elif remaining_type is NO_RETURN_VALUE:
            text = positive_text
            type_to_show = matched_type
        else:
            text = f"partially {positive_text}"
            type_to_show = unite_values(matched_type, remaining_type)
        if self.exclude_any:
            epilog = ""
        else:
            epilog = " (using permissive Any semantics)"
        if self.op not in (ast.LtE, ast.Gt) and isinstance(
            self.expected_type, KnownValue
        ):
            expected_type = repr(self.expected_type.val)
        else:
            expected_type = str(self.expected_type)
        return CanAssignError(
            f"Argument {self.arg} (type: {type_to_show}) {text} {expected_type}{epilog}"
        )


def subtract_unions(left: Value, right: Value) -> Value:
    if right is NO_RETURN_VALUE:
        return left
    right_vals = set(flatten_values(right))
    remaining = [
        subval
        for subval in flatten_values(unannotate(left))
        if subval not in right_vals
    ]
    return unite_values(*remaining)


_Operator = Union[Type[ast.cmpop], Literal["matches", "does not match"]]


@dataclass
class _Comparator:
    text: str
    negation: _Operator
    impl: Callable[[Any, Any], object]


def _dummy_impl(left: object, right: object) -> NoReturn:
    raise NotImplementedError


_OP_TO_DATA: Dict[_Operator, _Comparator] = {
    ast.Is: _Comparator("is", ast.IsNot, operator.is_),
    ast.IsNot: _Comparator("is not", ast.Is, operator.is_not),
    ast.Eq: _Comparator("==", ast.Eq, operator.eq),
    ast.NotEq: _Comparator("!=", ast.NotEq, operator.ne),
    ast.Gt: _Comparator(">", ast.LtE, operator.gt),
    ast.LtE: _Comparator("<=", ast.Gt, operator.le),
    ast.Lt: _Comparator("<", ast.GtE, operator.lt),
    ast.GtE: _Comparator(">=", ast.Lt, operator.ge),
    ast.In: _Comparator("in", ast.NotIn, lambda a, b: a in b),
    ast.NotIn: _Comparator("not in", ast.In, lambda a, b: a not in b),
    "matches": _Comparator("matches", "matches", _dummy_impl),
    "does not match": _Comparator("does not match", "matches", _dummy_impl),
}


@dataclass
class InvalidEvaluation:
    message: str
    node: ast.AST


@dataclass
class UserRaisedError:
    message: str
    active_conditions: Sequence[Condition]
    argument: Optional[str] = None

    def get_detail(self) -> Optional[str]:
        if self.active_conditions:
            return str(
                CanAssignError(
                    children=[cond.display() for cond in self.active_conditions]
                )
            )
        return None


EvaluateError = Union[InvalidEvaluation, UserRaisedError]


class Context:
    variables: VarMap
    positions: Mapping[str, Position]
    can_assign_context: CanAssignContext

    def evaluate_type(self, __node: ast.AST) -> Value:
        raise NotImplementedError

    def evaluate_value(self, __node: ast.AST) -> Value:
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

    def evaluate(self, ctx: Context) -> Tuple[Value, Sequence[UserRaisedError]]:
        visitor = EvaluateVisitor(ctx)
        result = visitor.run(self.node)
        errors = [e for e in visitor.errors if isinstance(e, UserRaisedError)]
        return result, errors


@dataclass
class CombinedReturn:
    children: Sequence[Optional[Value]]

    @classmethod
    def make(cls, *returns: "EvalReturn") -> "EvalReturn":
        if len(returns) == 1:
            return returns[0]
        children = []
        for child in returns:
            if isinstance(child, CombinedReturn):
                children += child.children
            else:
                children.append(child)
        return CombinedReturn(children)


EvalReturn = Union[None, Value, CombinedReturn]


@dataclass
class ConditionReturn:
    condition: Condition
    # These are None if there is no match, and a (possibly empty)
    # map of new variable values if there is a match.
    left_varmap: Optional[VarMap] = None
    right_varmap: Optional[VarMap] = None

    def reverse(self) -> "ConditionReturn":
        return ConditionReturn(
            left_varmap=self.right_varmap,
            right_varmap=self.left_varmap,
            condition=NotCondition(self.condition),
        )


@dataclass
class ConditionEvaluator(ast.NodeVisitor):
    ctx: Context
    validation_mode: bool = False
    errors: List[EvaluateError] = field(default_factory=list, init=False)

    def return_invalid(self, message: str, node: ast.AST) -> ConditionReturn:
        self.errors.append(InvalidEvaluation(message, node))
        return ConditionReturn(NullCondition())

    def visit_Call(self, node: ast.Call) -> ConditionReturn:
        if not isinstance(node.func, ast.Name):
            return self.return_invalid("Unexpected call", node.func)
        name = node.func.id
        if name in ("is_provided", "is_positional", "is_keyword"):
            if node.keywords or len(node.args) != 1:
                return self.return_invalid(f"{name}() takes a single argument", node)
            if not isinstance(node.args[0], ast.Name):
                return self.return_invalid(
                    f"Argument to {name}() must be a variable", node.args[0]
                )
            variable = node.args[0].id
            try:
                position = self.ctx.positions[variable]
            except KeyError:
                return self.return_invalid(
                    f"{variable} is not a valid variable", node.args[0]
                )
            if name == "is_provided":
                match = position is not DEFAULT and position is not UNKNOWN
            elif name == "is_positional":
                match = position is ARGS or isinstance(position, int)
            elif name == "is_keyword":
                match = position is KWARGS or isinstance(position, str)
            else:
                return self.return_invalid(name, node.func)
            condition = ArgumentKindCondition(variable, name)
            if match:
                return ConditionReturn(left_varmap={}, condition=condition)
            else:
                return ConditionReturn(
                    right_varmap={}, condition=NotCondition(condition)
                )
        elif name == "is_of_type":
            if len(node.args) != 2:
                return self.return_invalid(
                    "is_of_type() takes two positional arguments", node
                )
            varname_node = node.args[0]
            typ = self.ctx.evaluate_type(node.args[1])
            exclude_any = True
            for keyword in node.keywords:
                if keyword.arg == "exclude_any":
                    if isinstance(
                        keyword.value, ast.NameConstant
                    ) and keyword.value.value in (True, False):
                        exclude_any = keyword.value.value
                    else:
                        return self.return_invalid(
                            "exclude_any argument must be a literal bool", keyword.value
                        )
                else:
                    return self.return_invalid(
                        "Invalid keyword argument to is_of_type()", keyword
                    )
            return self.visit_is_of_type(
                varname_node, typ, "matches", exclude_any=exclude_any
            )
        else:
            return self.return_invalid(f"Invalid function {name}", node.func)

    def visit_is_of_type(
        self,
        varname_node: ast.AST,
        typ: Value,
        op: _Operator,
        *,
        exclude_any: bool = True,
    ) -> ConditionReturn:
        if not isinstance(varname_node, ast.Name):
            return self.return_invalid(
                "First argument to is_of_type() must be a name", varname_node
            )
        val = self.get_name(varname_node)
        if val is None:
            return ConditionReturn(NullCondition())
        can_assign = can_assign_maybe_exclude_any(
            typ, val, self.ctx.can_assign_context, exclude_any
        )
        if isinstance(can_assign, CanAssignError):
            pair = decompose_union(typ, val, self.ctx.can_assign_context, exclude_any)
            if pair is not None:
                _, remaining = pair
                return ConditionReturn(
                    condition=IsOfTypeCondition(
                        varname_node.id,
                        op,
                        val,
                        remaining,
                        typ,
                        exclude_any=exclude_any,
                    ),
                    left_varmap={varname_node.id: typ},
                    right_varmap={varname_node.id: remaining},
                )
            return ConditionReturn(right_varmap={}, condition=NullCondition())
        else:
            return ConditionReturn(
                left_varmap={varname_node.id: typ},
                condition=IsOfTypeCondition(
                    varname_node.id,
                    op,
                    val,
                    NO_RETURN_VALUE,
                    typ,
                    exclude_any=exclude_any,
                ),
            )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ConditionReturn:
        if isinstance(node.op, ast.Not):
            ret = self.visit(node.operand)
            return ret.reverse()
        else:
            return self.return_invalid("Unsupported unary operation", node)

    def visit_Compare(self, node: ast.Compare) -> ConditionReturn:
        if len(node.ops) != 1:
            return self.return_invalid("Chained comparison is unsupported", node)
        op = node.ops[0]
        right = node.comparators[0]
        right_operand = self.evaluate_literal(right)
        if right_operand is None:
            return ConditionReturn(NullCondition())
        if isinstance(node.left, ast.Name) and isinstance(
            op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)
        ):
            ret = self.visit_is_of_type(node.left, right_operand, type(op))
            if isinstance(op, (ast.NotEq, ast.IsNot)):
                return ret.reverse()
            return ret

        if isinstance(node.left, ast.Attribute):
            mod = self.evaluate_literal(node.left.value)
            if mod == KnownValue(sys):
                if node.left.attr == "platform":
                    left_operand = sys.platform
                elif node.left.attr == "version_info":
                    left_operand = sys.version_info
                else:
                    return self.return_invalid(
                        "Only comparisons on sys.platform and sys.version_info are"
                        " suppoorted",
                        node.left,
                    )
                data = _OP_TO_DATA[type(op)]
                try:
                    result = data.impl(left_operand, right_operand.val)
                except Exception:
                    return self.return_invalid(
                        f"Invalid sys.{node.left.attr} comparison", node
                    )
                if node.left.attr == "platform":
                    condition = PlatformCondition(
                        sys.platform, type(op), right_operand.val
                    )
                else:
                    condition = VersionCondition(
                        sys.version_info[:2], type(op), right_operand.val
                    )
                if result:
                    return ConditionReturn(condition, left_varmap={})
                else:
                    return ConditionReturn(condition, right_varmap={})

        return self.return_invalid("Unsupported comparison operator", node)

    def visit_BoolOp(self, node: ast.BoolOp) -> ConditionReturn:
        if self.validation_mode:
            for operand in node.values:
                self.visit(operand)
            return ConditionReturn(NullCondition())
        active = []
        is_and = isinstance(node.op, ast.And)
        remaining_varmaps = []
        narrowed_varmap = {}
        stack = contextlib.ExitStack()
        with stack:
            for operand in node.values:
                result = self.visit(operand)
                if is_and:
                    if result.left_varmap is None:
                        # Condition returns False
                        active.append(NotCondition(result.condition))
                        return ConditionReturn(
                            right_varmap=result.right_varmap,
                            condition=ConditionList(active),
                        )
                    elif result.right_varmap is None:
                        # Condition returns True
                        narrowed_varmap.update(result.left_varmap)
                        stack.enter_context(
                            self.ctx.narrow_variables(result.left_varmap)
                        )
                        active.append(result.condition)
                    else:
                        # Condition matches partially
                        active.append(result.condition)
                        narrowed_varmap.update(result.left_varmap)
                        stack.enter_context(
                            self.ctx.narrow_variables(result.left_varmap)
                        )
                        remaining_varmaps.append(result.right_varmap)
                else:
                    if result.left_varmap is None:
                        # Condition returns False
                        narrowed_varmap.update(result.right_varmap)
                        stack.enter_context(
                            self.ctx.narrow_variables(result.right_varmap)
                        )
                        active.append(NotCondition(result.condition))
                    elif result.right_varmap is None:
                        # Condition returns True
                        active.append(result.condition)
                        return ConditionReturn(
                            left_varmap=result.left_varmap,
                            condition=ConditionList(active),
                        )
                    else:
                        # Condition partially matches
                        active.append(result.condition)
                        narrowed_varmap.update(result.right_varmap)
                        stack.enter_context(
                            self.ctx.narrow_variables(result.right_varmap)
                        )
                        remaining_varmaps.append(result.left_varmap)

        # We got only partial matches
        if is_and:
            return ConditionReturn(
                ConditionList(active),
                left_varmap=narrowed_varmap,
                right_varmap=unite_varmaps(remaining_varmaps),
            )
        else:
            return ConditionReturn(
                ConditionList(active),
                left_varmap=unite_varmaps(remaining_varmaps),
                right_varmap=narrowed_varmap,
            )

    def evaluate_literal(self, node: ast.expr) -> Optional[KnownValue]:
        val = self.ctx.evaluate_value(node)
        if isinstance(val, KnownValue):
            return val
        self.errors.append(InvalidEvaluation("Only literals supported", node))
        return None

    def get_name(self, node: ast.Name) -> Optional[Value]:
        try:
            return self.ctx.variables[node.id]
        except KeyError:
            self.errors.append(InvalidEvaluation(f"Invalid variable {node.id}", node))
            return None


@dataclass
class EvaluateVisitor(ast.NodeVisitor):
    ctx: Context
    errors: List[EvaluateError] = field(default_factory=list)
    active_conditions: List[Condition] = field(default_factory=list)
    validation_mode: bool = False

    def run(self, node: ast.AST) -> Value:
        ret = self.visit(node)
        return self._evaluate_ret(ret, node)

    def _evaluate_ret(self, ret: EvalReturn, node: ast.AST) -> Value:
        if ret is None:
            # TODO return the func's return annotation instead
            if not self.validation_mode:
                self.add_invalid("Evaluator failed to return", node)
            return AnyValue(AnySource.error)
        elif isinstance(ret, CombinedReturn):
            children = [self._evaluate_ret(child, node) for child in ret.children]
            return unite_values(*children)
        else:
            return ret

    def add_invalid(self, message: str, node: ast.AST) -> None:
        error = InvalidEvaluation(message, node)
        self.errors.append(error)

    @contextmanager
    def add_active_condition(self, condition: Condition) -> Iterator[None]:
        self.active_conditions.append(condition)
        try:
            yield
        finally:
            popped = self.active_conditions.pop()
            assert popped == condition

    def visit_FunctionDef(self, node: ast.FunctionDef) -> EvalReturn:
        return self.visit_block(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> EvalReturn:
        return self.visit_block(node.body)

    def visit_block(self, statements: Sequence[ast.stmt]) -> EvalReturn:
        possible_returns = []
        for stmt in statements:
            result = self.visit(stmt)
            if result is None:
                continue
            if isinstance(result, Value):
                return CombinedReturn.make(*possible_returns, result)
            else:
                if all(res is not None for res in result.children):
                    return CombinedReturn.make(*possible_returns, *result.children)
                else:
                    possible_returns += [
                        res for res in result.children if res is not None
                    ]
        return CombinedReturn.make(*possible_returns, None)

    def visit_Pass(self, node: ast.Pass) -> EvalReturn:
        return None

    def visit_Return(self, node: ast.Return) -> EvalReturn:
        if node.value is None:
            self.add_invalid("return statement must have a value", node)
            return KnownValue(None)
        return self.ctx.evaluate_type(node.value)

    def visit_Expr(self, node: ast.Expr) -> EvalReturn:
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "show_error"
        ):
            call = node.value
            if len(call.args) != 1:
                self.add_invalid(
                    "show_error() takes exactly one positional argument", call
                )
                return None
            if not isinstance(call.args[0], ast.Str):
                self.add_invalid(
                    "show_error() message must be a string literal", call.args[0]
                )
                return None
            message = call.args[0].s
            argument = None
            for keyword in call.keywords:
                if keyword.arg == "argument":
                    if not isinstance(keyword.value, ast.Name):
                        self.add_invalid("argument must be a name", keyword.value)
                        return None
                    argument = keyword.value.id
                    if argument not in self.ctx.variables:
                        self.add_invalid(
                            f"{argument} is not a valid argument", keyword.value
                        )
                        return None
                else:
                    self.add_invalid(
                        "Invalid keyword argument to show_error()", keyword
                    )
                    return None
            self.errors.append(
                UserRaisedError(message, list(self.active_conditions), argument)
            )
            return None
        self.add_invalid("Invalid statement", node)
        return None

    def visit_If(self, node: ast.If) -> EvalReturn:
        visitor = ConditionEvaluator(self.ctx, self.validation_mode)
        condition = visitor.visit(node.test)
        self.errors += visitor.errors
        if self.validation_mode:
            self.visit_block(node.body)
            self.visit_block(node.orelse)
            return None
        if condition.left_varmap is not None:
            with self.ctx.narrow_variables(
                condition.left_varmap
            ), self.add_active_condition(condition.condition):
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
                return CombinedReturn.make(left_result, right_result)
            else:
                return left_result
        else:
            if condition.right_varmap is not None:
                return right_result
            else:
                self.add_invalid("Condition must either match or not match", node)
                return None

    def generic_visit(self, node: ast.AST) -> Any:
        self.add_invalid("Invalid code in type evaluator", node)
        return None


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
        return None
    return Evaluator(evaluation_func, evaluator, evaluation_func.__globals__)


def validate(node: ast.AST, ctx: Context) -> List[InvalidEvaluation]:
    visitor = EvaluateVisitor(ctx, validation_mode=True)
    visitor.run(node)
    return [error for error in visitor.errors if isinstance(error, InvalidEvaluation)]


def decompose_union(
    expected_type: Value, parent_value: Value, ctx: CanAssignContext, exclude_any: bool
) -> Optional[Tuple[TypeVarMap, Value]]:
    value = unannotate(parent_value)
    if isinstance(value, MultiValuedValue):
        tv_maps = []
        remaining_values = []
        for val in value.vals:
            can_assign = can_assign_maybe_exclude_any(
                expected_type, val, ctx, exclude_any
            )
            if isinstance(can_assign, CanAssignError):
                remaining_values.append(val)
            else:
                tv_maps.append(can_assign)
        if tv_maps:
            tv_map = unify_typevar_maps(tv_maps)
            assert (
                remaining_values
            ), f"all union members matched between {expected_type} and {parent_value}"
            return tv_map, unite_values(*remaining_values)
    return None


def can_assign_maybe_exclude_any(
    left: Value, right: Value, ctx: CanAssignContext, exclude_any: bool
) -> CanAssign:
    if exclude_any:
        with ctx.set_exclude_any():
            return left.can_assign(right, ctx)
    else:
        return left.can_assign(right, ctx)


def unite_varmaps(varmaps: Sequence[VarMap]) -> Optional[VarMap]:
    if not varmaps:
        return None
    keys = set.intersection(*[set(m) for m in varmaps])
    return {
        key: unite_values(*[varmap.get(key, NO_RETURN_VALUE) for varmap in varmaps])
        for key in keys
    }
