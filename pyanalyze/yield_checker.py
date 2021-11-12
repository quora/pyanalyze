"""

test_scope component that checks for errors related to yields in async code.

This does the following three checks:
- yielding the same thing more than once in the same yield
- yielding an async task in a non-async function
- yielding before using the result of the previous yield

"""

import ast
from ast_decompiler import decompile
import asynq
import contextlib
from dataclasses import dataclass, field
import qcore
import itertools
import logging
from typing import (
    Any,
    Dict,
    Set,
    Callable,
    ContextManager,
    Iterator,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
)

from .asynq_checker import AsyncFunctionKind
from .error_code import ErrorCode
from .value import Value, KnownValue, UnboundMethodValue, UNINITIALIZED_VALUE
from .analysis_lib import get_indentation, get_line_range_for_node
from .node_visitor import Replacement

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor


@dataclass
class YieldInfo:
    """Wrapper class for yield nodes."""

    yield_node: ast.Yield
    statement_node: ast.stmt
    lines: List[str]
    line_range: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.line_range = get_line_range_for_node(self.statement_node, self.lines)

    def is_assign_or_expr(self) -> bool:
        if not isinstance(self.statement_node, (ast.Expr, ast.Assign)):
            return False
        return self.statement_node.value is self.yield_node

    def get_indentation(self) -> int:
        return get_indentation(self.lines[self.statement_node.lineno - 1])

    def target_and_value(self) -> Tuple[List[ast.AST], List[ast.AST]]:
        """Returns a pair of a list of target nodes and a list of value nodes."""
        assert self.yield_node.value is not None
        if isinstance(self.statement_node, ast.Assign):
            # this branch is for assign statements
            # e.g. x = yield y.asynq()
            # _ = yield async_fn.asynq()
            if not isinstance(self.statement_node.targets[0], ast.Tuple):
                # target is one entity
                return ([self.statement_node.targets[0]], [self.yield_node.value])
            # target is a tuple
            elif (
                isinstance(self.yield_node.value, ast.Call)
                and isinstance(self.yield_node.value.func, ast.Name)
                and self.yield_node.value.func.id == "tuple"
                and isinstance(self.yield_node.value.args[0], ast.Tuple)
            ):
                # value is a call to tuple()
                # e.g. x, y = yield tuple((a.asynq(), b.asynq()))
                # in this case we remove the call to tuple and return
                # the plain elements of the tuple but we remove the surrounding braces
                return (
                    self.statement_node.targets[0].elts,
                    self.yield_node.value.args[0].elts,
                )

            elif isinstance(self.yield_node.value, ast.Tuple):
                # value is a tuple too, return both targets and values as such
                # but get rid of the parenthesis
                return (self.statement_node.targets[0].elts, self.yield_node.value.elts)

            # target is a tuple but only one value
            # e.g. x, y = yield f.asynq()
            # in this case we'll wrap the target with () so that they
            # can be combined with another yield
            return ([self.statement_node.targets[0]], [self.yield_node.value])

        # not an assign statement
        # e.g. yield x.asynq()
        if not isinstance(self.yield_node.value, ast.Tuple):
            # single entity yielded
            return ([ast.Name(id="_", ctx=ast.Store())], [self.yield_node.value])
        # multiple values yielded e.g. yield x.asynq(), z.asynq()
        return (
            [ast.Name(id="_", ctx=ast.Store()) for _ in self.yield_node.value.elts],
            self.yield_node.value.elts,
        )


class VarnameGenerator:
    """Class to generate a unique variable name from an AST node.

    Split off into a separate class for ease of testing.

    To construct this class, pass in a function that, given a name, returns whether it is available.
    Call .get(node) on the instance to get a variable name for an AST node.

    """

    def __init__(self, is_available: Callable[[str], bool]) -> None:
        self.is_available = is_available

    def get(self, node: ast.AST) -> str:
        """Returns a unique variable name for this node."""
        candidate = self._get_candidate(node).lstrip("_")
        return self._ensure_unique(candidate)

    def _get_candidate(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            camel_cased = _camel_case_to_snake_case(node.id)
            if camel_cased != node.id:
                return camel_cased
            else:
                return f"{node.id}_result"
        elif isinstance(node, ast.Attribute):
            if node.attr == "async" or node.attr == "asynq":
                return self._get_candidate(node.value)
            elif node.attr.endswith("_async"):
                # probably a method call like .get_async
                return self._get_candidate(node.value)
            else:
                varname = node.attr.lstrip("_")
                for prefix in ("render_", "get_"):
                    if varname.startswith(prefix):
                        return varname[len(prefix) :]
                return varname
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "async_call"
                and node.args
            ):
                return self._get_candidate(node.args[0])
            return self._get_candidate(node.func)
        else:
            return "autogenerated_var"

    def _ensure_unique(self, varname: str) -> str:
        """Ensures an autogenerated variable name is unique by appending numbers to it."""
        if self.is_available(varname):
            return varname
        else:
            for i in itertools.count(2):
                next_varname = f"{varname}{i}"
                if self.is_available(next_varname):
                    return next_varname
        assert False, "unreachable"


@dataclass
class YieldChecker:
    visitor: "NameCheckVisitor"
    variables_from_yield_result: Dict[str, bool] = field(default_factory=dict)
    in_yield_result_assignment: bool = False
    in_non_async_yield: bool = False
    last_yield_in_aug_assign: bool = False
    previous_yield: Optional[ast.Yield] = None
    statement_for_previous_yield: Optional[ast.stmt] = None
    used_varnames: Set[str] = field(default_factory=set)
    added_decorator: bool = False

    @contextlib.contextmanager
    def check_yield(
        self, node: ast.Yield, current_statement: ast.stmt
    ) -> Iterator[None]:
        assert current_statement is not None
        if self.visitor.async_kind == AsyncFunctionKind.normal:
            self._check_for_duplicate_yields(node, self.visitor.current_statement)

        in_non_async_yield = self.visitor.async_kind == AsyncFunctionKind.non_async
        with qcore.override(self, "in_non_async_yield", in_non_async_yield):
            yield

        if self.visitor.async_kind == AsyncFunctionKind.normal:
            self._maybe_show_unnecessary_yield_error(node, current_statement)

        if self.last_yield_in_aug_assign:
            self._maybe_show_unnecessary_yield_error(node, current_statement)

        self.last_yield_in_aug_assign = self._is_augassign_target(node)
        self.variables_from_yield_result = {}
        self.previous_yield = node
        self.statement_for_previous_yield = current_statement

    # Unnecessary yield checking

    def check_yield_result_assignment(self, in_yield: bool) -> ContextManager[None]:
        return qcore.override(self, "in_yield_result_assignment", in_yield)

    def record_assignment(self, name: str) -> None:
        if self.in_yield_result_assignment:
            self.variables_from_yield_result[name] = False

    def record_usage(self, name: str, node: ast.AST) -> None:
        if name in self.variables_from_yield_result:
            if self._is_augassign_target(node):
                return
            self.variables_from_yield_result[name] = True

    def reset_yield_checks(self) -> None:
        """Resets variables for the unnecessary yield check.

        This is done while visiting if statements to prevent the unnecessary yield check from
        concluding that yields in one branch of the if are unused.

        """
        self.variables_from_yield_result = {}
        self.last_yield_in_aug_assign = False

    # Missing @async detection

    def record_call(self, value: Value, node: ast.Call) -> None:
        if (
            not self.in_non_async_yield
            or not self._is_async_call(value, node.func)
            or self.added_decorator
        ):
            return

        # prevent ourselves from adding the decorator to the same function multiple times
        self.added_decorator = True
        func_node = self.visitor.node_context.nearest_enclosing(ast.FunctionDef)
        lines = self.visitor._lines()
        # this doesn't handle decorator order, it just adds @asynq() right before the def
        i = func_node.lineno - 1
        def_line = lines[i]
        indentation = len(def_line) - len(def_line.lstrip())
        replacement = Replacement([i + 1], [" " * indentation + "@asynq()\n", def_line])
        self.visitor.show_error(
            node, error_code=ErrorCode.missing_asynq, replacement=replacement
        )

    # Internal part

    def _is_augassign_target(self, node: ast.AST) -> bool:
        return (
            isinstance(self.visitor.current_statement, ast.AugAssign)
            and node is self.visitor.current_statement.value
        )

    def _is_async_call(self, value: Value, node: ast.AST) -> bool:
        # calls to something.asynq are always async
        if isinstance(node, ast.Attribute) and (
            node.attr in ("async", "asynq", "future") or node.attr.endswith("_async")
        ):
            return True
        if isinstance(value, UnboundMethodValue):
            obj = value.get_method()
        elif isinstance(value, KnownValue):
            obj = value.val
        else:
            return False
        return self.is_async_fn(obj)

    def _maybe_show_unnecessary_yield_error(
        self, node: ast.Yield, current_statement: ast.stmt
    ) -> None:
        if isinstance(current_statement, ast.Expr) and current_statement.value is node:
            return

        current_yield_result_vars = self.variables_from_yield_result

        # check if we detected anything being used out of the last yield
        self.visitor.log(logging.DEBUG, "Yield result", current_yield_result_vars)
        if len(current_yield_result_vars) > 0:
            if not any(current_yield_result_vars.values()):
                unused = list(current_yield_result_vars.keys())
                self.show_unnecessary_yield_error(unused, node, current_statement)

    def _check_for_duplicate_yields(
        self, node: ast.Yield, current_statement: ast.stmt
    ) -> None:
        if not isinstance(node.value, ast.Tuple) or len(node.value.elts) < 2:
            return

        duplicate_indices = {}  # index to first index
        seen = {}  # ast.dump result to index
        for i, member in enumerate(node.value.elts):
            # identical AST nodes don't compare equally, so just stringify them for comparison
            code = ast.dump(member)
            if code in seen:
                duplicate_indices[i] = seen[code]
            else:
                seen[code] = i

        if not duplicate_indices:
            return

        new_members = [
            elt for i, elt in enumerate(node.value.elts) if i not in duplicate_indices
        ]
        if len(new_members) == 1:
            new_value = new_members[0]
        else:
            new_value = ast.Tuple(elts=new_members)
        new_yield_node = ast.Yield(value=new_value)

        if isinstance(current_statement, ast.Expr) and current_statement.value is node:
            new_nodes = [ast.Expr(value=new_yield_node)]
        elif (
            isinstance(current_statement, ast.Assign)
            and current_statement.value is node
        ):
            if (
                len(current_statement.targets) != 1
                or not isinstance(current_statement.targets[0], ast.Tuple)
                or len(current_statement.targets[0].elts) != len(node.value.elts)
            ):
                new_nodes = None
            else:
                new_targets = []
                # these are for cases where we do something like
                #   a, b = yield f.asynq(), f.asynq()
                # we turn this into
                #   a = yield f.asynq()
                #   b = a
                extra_nodes = []
                assignment_targets = current_statement.targets[0].elts
                for i, target in enumerate(assignment_targets):
                    if i not in duplicate_indices:
                        new_targets.append(target)
                    elif not (isinstance(target, ast.Name) and target.id == "_"):
                        extra_nodes.append(
                            ast.Assign(
                                targets=[target],
                                value=assignment_targets[duplicate_indices[i]],
                            )
                        )
                if len(new_targets) == 1:
                    new_target = new_targets[0]
                else:
                    new_target = ast.Tuple(elts=new_targets)

                new_assign = ast.Assign(targets=[new_target], value=new_yield_node)
                new_nodes = [new_assign] + extra_nodes
        else:
            new_nodes = None

        if new_nodes is not None:
            lines_to_delete = self._lines_of_node(node)
            indent = self._indentation_of_node(current_statement)
            new_code = "".join(
                decompile(node, starting_indentation=indent) for node in new_nodes
            )
            new_lines = [line + "\n" for line in new_code.splitlines()]
            replacement = Replacement(lines_to_delete, new_lines)
        else:
            replacement = None

        self.visitor.show_error(
            node, error_code=ErrorCode.duplicate_yield, replacement=replacement
        )

    def show_unnecessary_yield_error(
        self, unused: Sequence[object], node: ast.Yield, current_statement: ast.stmt
    ) -> None:
        if not unused:
            message = "Unnecessary yield: += assignments can be combined"
        elif len(unused) == 1:
            message = "Unnecessary yield: %s was not used before this yield" % (
                unused[0],
            )
        else:
            unused_str = ", ".join(map(str, unused))
            message = f"Unnecessary yield: {unused_str} were not used before this yield"
        replacement = self._create_replacement_for_yield_nodes(node, current_statement)
        self.visitor.show_error(
            node, message, ErrorCode.unnecessary_yield, replacement=replacement
        )

    def _lines_of_node(self, yield_node: ast.Yield) -> List[int]:
        """Returns the lines that the given yield node occupies."""
        # see if it has a parent assign node
        if hasattr(yield_node, "parent_assign_node"):
            first_lineno = yield_node.parent_assign_node.lineno
        else:
            first_lineno = yield_node.lineno
        lines = self.visitor._lines()
        first_line = lines[first_lineno - 1]
        indent = get_indentation(first_line)
        last_lineno = first_lineno + 1
        while True:
            if last_lineno - 1 >= len(lines):
                break
            last_line = lines[last_lineno - 1]
            last_line_indent = get_indentation(last_line)
            # if it is just spaces then stop
            if last_line.isspace():
                break
            if last_line_indent > indent:
                last_lineno += 1
            elif (
                last_line_indent == indent
                and len(last_line) >= indent
                and last_line[indent] == ")"
            ):
                last_lineno += 1
            else:
                break
        return list(range(first_lineno, last_lineno))

    def _create_replacement_for_yield_nodes(
        self, second_node: ast.Yield, second_parent: ast.stmt
    ) -> Optional[Replacement]:
        """Returns one statement that does a batched yield of the given 2 yields."""
        lines = self.visitor._lines()
        assert self.previous_yield is not None
        assert self.statement_for_previous_yield is not None
        first_yield = YieldInfo(
            self.previous_yield, self.statement_for_previous_yield, lines
        )
        second_yield = YieldInfo(second_node, second_parent, lines)

        # this shouldn't happen in async code but test_scope checks for it elsewhere
        if (
            first_yield.yield_node.value is None
            or second_yield.yield_node.value is None
        ):
            return None

        # check whether there is any code between the two yield statements
        lines_in_between = list(
            range(first_yield.line_range[-1] + 1, second_yield.line_range[0])
        )
        adjacent = not lines_in_between or all(
            lines[i - 1].isspace() for i in lines_in_between
        )

        if first_yield.is_assign_or_expr() and second_yield.is_assign_or_expr():
            if adjacent:
                return self._merge_assign_nodes(first_yield, second_yield)
            else:
                # give up if the two are at different indentations
                # this probably means the second one is in a with context or try-except
                if first_yield.get_indentation() != second_yield.get_indentation():
                    return None
                # if there is intervening code, first move the first yield to right before the
                # second one
                to_delete = list(
                    range(first_yield.line_range[0], second_yield.line_range[0])
                )
                yield_lines = lines[
                    first_yield.line_range[0] - 1 : first_yield.line_range[-1]
                ]
                between_lines = lines[
                    first_yield.line_range[-1] : second_yield.line_range[0] - 1
                ]
                return Replacement(to_delete, between_lines + yield_lines)
        elif first_yield.is_assign_or_expr():
            # move the target of the second yield to right after the first one
            indentation = first_yield.get_indentation()
            new_assign_lines, replace_yield = self._move_out_var_from_yield(
                second_yield, indentation
            )

            between_lines = lines[
                first_yield.line_range[-1] : second_yield.line_range[0] - 1
            ]
            to_add = [*new_assign_lines, *between_lines]
            if replace_yield.lines_to_add:
                to_add += replace_yield.lines_to_add

            # determine lines to remove
            between_range = list(
                range(first_yield.line_range[-1] + 1, second_yield.line_range[0])
            )
            to_delete = [*between_range, *replace_yield.linenos_to_delete]

            return Replacement(to_delete, to_add)
        else:
            indentation = first_yield.get_indentation()
            lines_to_add, replace_first = self._move_out_var_from_yield(
                first_yield, indentation
            )

            if second_yield.is_assign_or_expr():
                # just move it
                second_lines = lines[
                    second_yield.line_range[0] - 1 : second_yield.line_range[-1]
                ]
                second_indentation = get_indentation(second_lines[0])
                difference = indentation - second_indentation
                if difference > 0:
                    second_lines = [" " * difference + line for line in second_lines]
                elif difference < 0:
                    second_lines = [line[(-difference):] for line in second_lines]
                lines_to_add += second_lines
                lines_for_second_yield = []
            else:
                second_assign_lines, replace_second = self._move_out_var_from_yield(
                    second_yield, indentation
                )
                lines_to_add += second_assign_lines
                lines_for_second_yield = replace_second.lines_to_add

            lines_to_add += replace_first.lines_to_add or []
            lines_to_add += lines[
                first_yield.line_range[-1] : second_yield.line_range[0] - 1
            ]
            if lines_for_second_yield:
                lines_to_add += lines_for_second_yield

            linenos_to_delete = list(
                range(first_yield.line_range[0], second_yield.line_range[-1] + 1)
            )
            return Replacement(linenos_to_delete, lines_to_add)

    def _move_out_var_from_yield(
        self, yield_info: YieldInfo, indentation: int
    ) -> Tuple[List[str], Replacement]:
        """Helper for splitting up a yield node and moving it to an earlier place.

        For example, it will help turn:

            some_code((yield get_value.asynq()))

        into:

            value = yield get_value.asynq()
            ...
            some_code(value)

        Returns a pair of a list of lines to form the new assignment code (value = ...) and a
        Replacement object implementing the second change.

        """
        assert yield_info.yield_node.value is not None
        varname = self.generate_varname_from_node(yield_info.yield_node.value)
        name_node = ast.Name(id=varname)
        replace = self.visitor.replace_node(
            yield_info.yield_node,
            name_node,
            current_statement=yield_info.statement_node,
        )

        new_assign = ast.Assign(targets=[name_node], value=yield_info.yield_node)
        new_assign_code = decompile(new_assign, starting_indentation=indentation)
        assign_lines = [line + "\n" for line in new_assign_code.splitlines()]
        return assign_lines, replace

    def _merge_assign_nodes(
        self, first_yield: YieldInfo, second_yield: YieldInfo
    ) -> Replacement:
        # the basic approach is to extract the targets (left hand side of the assignment) and the
        # values to yield (on the right hand side) independently for each of the yield nodes and
        # then combine them. But there are different cases to consider

        first_node_target_value = first_yield.target_and_value()
        second_node_target_value = second_yield.target_and_value()
        targets = first_node_target_value[0] + second_node_target_value[0]
        values = first_node_target_value[1] + second_node_target_value[1]
        yield_node = ast.Yield(value=ast.Tuple(elts=values))

        # if everything in targets is an underscore '_', then just avoid creating
        # an assignment statement
        if all(isinstance(target, ast.Name) and target.id == "_" for target in targets):
            new_node = ast.Expr(value=yield_node)
        else:
            new_node = ast.Assign(targets=[ast.Tuple(elts=targets)], value=yield_node)

        indent = self._indentation_of_node(first_yield.statement_node)
        new_code = decompile(new_node, starting_indentation=indent)
        lines_to_delete = list(
            range(first_yield.line_range[0], second_yield.line_range[-1] + 1)
        )
        return Replacement(lines_to_delete, [new_code])

    def _indentation_of_node(self, node: ast.AST) -> int:
        """Calculate the indentation of an AST node."""
        line = self.visitor._lines()[node.lineno - 1]
        # assumes that there is at least one space in the beginning which should be the case
        return get_indentation(line)

    def generate_varname_from_node(self, node: ast.AST) -> str:
        def is_available(name: str) -> bool:
            if name in self.used_varnames:
                return False
            value = self.visitor.scopes.get(name, node=None, state=None)
            return value is UNINITIALIZED_VALUE

        varname = VarnameGenerator(is_available).get(node)
        self.used_varnames.add(varname)
        return varname

    def is_async_fn(self, obj: Any) -> bool:
        if hasattr(obj, "__self__"):
            if isinstance(
                obj.__self__, asynq.decorators.AsyncDecorator
            ) or asynq.is_async_fn(obj.__self__):
                return True
        return asynq.is_async_fn(obj)


def _camel_case_to_snake_case(s: str) -> str:
    """Converts a CamelCase string to snake_case."""
    out = []
    last_was_uppercase = False
    for c in s:
        if c.isupper():
            out.append(c.lower())
        else:
            if last_was_uppercase and len(out) > 1:
                out[-1] = "_" + out[-1]
            out.append(c)
        last_was_uppercase = c.isupper()
    return "".join(out)
