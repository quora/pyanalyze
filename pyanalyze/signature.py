"""

The :class:`Signature` object and associated functionality. This
provides a way to represent rich callable objects and type check
calls.

"""


from .error_code import ErrorCode
from .safe import all_of_type
from .stacked_scopes import (
    AndConstraint,
    OrConstraint,
    Composite,
    Constraint,
    ConstraintType,
    NULL_CONSTRAINT,
    AbstractConstraint,
    Varname,
)
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    AsyncTaskIncompleteValue,
    CanAssignContext,
    GenericValue,
    HasAttrExtension,
    HasAttrGuardExtension,
    KVPair,
    KnownValue,
    MultiValuedValue,
    ParameterTypeGuardExtension,
    SequenceIncompleteValue,
    DictIncompleteValue,
    TypeGuardExtension,
    TypeVarValue,
    TypedDictValue,
    TypedValue,
    Value,
    TypeVarMap,
    CanAssign,
    CanAssignError,
    concrete_values_from_iterable,
    extract_typevars,
    flatten_values,
    replace_known_sequence_value,
    stringify_object,
    unify_typevar_maps,
    unite_values,
    NO_RETURN_VALUE,
)

import ast
import asynq
from collections import defaultdict, OrderedDict
import collections.abc
from dataclasses import dataclass, field, replace
from functools import reduce
import itertools
from types import MethodType, FunctionType
import inspect
import qcore
from qcore.helpers import safe_str
from typing import (
    Any,
    Iterable,
    NamedTuple,
    Optional,
    ClassVar,
    Sequence,
    Union,
    Callable,
    Dict,
    List,
    Set,
    TypeVar,
    Tuple,
    TYPE_CHECKING,
)
from typing_extensions import Literal

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor

EMPTY = inspect.Parameter.empty

ARGS = qcore.MarkerObject("*args")
KWARGS = qcore.MarkerObject("**kwargs")


@dataclass
class PossibleKwarg:
    """Label used for keyword arguments that may not be present at runtiime."""

    name: str


# Representation of a single argument to a call. Second member is
# None for positional args, str for keyword args,
# ARGS for *args, KWARGS for **kwargs.
Argument = Tuple[Composite, Union[None, str, Literal[ARGS], Literal[KWARGS]]]

# Arguments bound to a call. The first member of the tuple is a reference back
# to the ActualArguments: the value is a str for a kwarg, int for a positional,
# None for something we cannot infer.
BoundArgs = Dict[str, Tuple[Union[None, str, int], Composite]]


@dataclass
class ActualArguments:
    """Represents the actual arguments to a call.

    Before creating this class, we decompose ``*args`` and ``**kwargs`` arguments
    of known composition into additional positional and keyword arguments, and we
    merge multiple ``*args`` or ``**kwargs``.

    Creating the ``ActualArguments`` for a call is independent of the signature
    of the callee.

    """

    positionals: List[Composite]
    star_args: Optional[Value]  # represents the type of the elements of *args
    keywords: Dict[str, Tuple[bool, Composite]]
    star_kwargs: Optional[Value]  # represents the type of the elements of **kwargs
    kwargs_required: bool


def _maybe_or_constraint(
    left: AbstractConstraint, right: AbstractConstraint
) -> AbstractConstraint:
    if left is NULL_CONSTRAINT or right is NULL_CONSTRAINT:
        return NULL_CONSTRAINT
    return OrConstraint(left, right)


class ImplReturn(NamedTuple):
    """Return value of :term:`impl` functions.

    These functions return either a single :class:`pyanalyze.value.Value`
    object, indicating what the function returns, or an instance of this class.

    """

    return_value: Value
    """The return value of the function."""
    constraint: AbstractConstraint = NULL_CONSTRAINT
    """A :class:`pyanalyze.stacked_scopes.Constraint` indicating things that are true
    if the function returns a truthy value."""
    no_return_unless: AbstractConstraint = NULL_CONSTRAINT
    """A :class:`pyanalyze.stacked_scopes.Constraint` indicating things that are true
    unless the function does not return."""
    is_error: bool = False
    """Whether there was an error in this call. Used only for overload resolutioon."""
    used_any_for_match: bool = False
    """Whether Any was used for this match. Used only for overload resolution."""
    remaining_arguments: Optional[ActualArguments] = None
    """Arguments that still need to be processed. Used only for overload resolution."""

    @classmethod
    def unite_impl_rets(cls, rets: Sequence["ImplReturn"]) -> "ImplReturn":
        if not rets:
            return ImplReturn(NO_RETURN_VALUE)
        return ImplReturn(
            unite_values(*[r.return_value for r in rets]),
            reduce(_maybe_or_constraint, [r.constraint for r in rets]),
            reduce(_maybe_or_constraint, [r.no_return_unless for r in rets]),
        )


@dataclass
class CallContext:
    """The context passed to an :term:`impl` function."""

    vars: Dict[str, Value]
    """Dictionary of variable names passed to the function."""
    visitor: "NameCheckVisitor"
    """Using the visitor can allow various kinds of advanced logic
    in impl functions."""
    bound_args: inspect.BoundArguments
    node: ast.AST
    """AST node corresponding to the function call. Useful for
    showing errors."""

    def ast_for_arg(self, arg: str) -> Optional[ast.AST]:
        composite = self.composite_for_arg(arg)
        if composite is not None:
            return composite.node
        return None

    def varname_for_arg(self, arg: str) -> Optional[Varname]:
        """Return a :term:`varname` corresponding to the given function argument.

        This is useful for creating a :class:`pyanalyze.stacked_scopes.Constraint`
        referencing the argument.

        """
        composite = self.composite_for_arg(arg)
        if composite is not None:
            return composite.varname
        return None

    def composite_for_arg(self, arg: str) -> Optional[Composite]:
        composite = self.bound_args.arguments.get(arg)
        if isinstance(composite, Composite):
            return composite
        return None

    def show_error(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.incompatible_call,
        *,
        arg: Optional[str] = None,
        node: Optional[ast.AST] = None,
        detail: Optional[str] = None,
    ) -> None:
        """Show an error.

        If the `arg` parameter is given, we attempt to find the
        AST for that argument to the function and point the error
        to it.

        """
        if node is None and arg is not None:
            node = self.ast_for_arg(arg)
        if node is None:
            node = self.node
        self.visitor.show_error(node, message, error_code=error_code, detail=detail)


Impl = Callable[[CallContext], Union[Value, ImplReturn]]


class SigParameter(inspect.Parameter):
    """Wrapper around :class:`inspect.Parameter` that stores annotations
    as :class:`pyanalyze.value.Value` objects."""

    __slots__ = ()

    def __init__(
        self,
        name: str,
        kind: inspect._ParameterKind = inspect.Parameter.POSITIONAL_OR_KEYWORD,
        *,
        default: Union[None, Value, Literal[EMPTY]] = None,
        annotation: Union[None, Value, Literal[EMPTY]] = None,
    ) -> None:
        if default is None:
            default_composite = EMPTY
        elif isinstance(default, Value):
            default_composite = Composite(default, None, None)
        else:
            default_composite = default
        if annotation is None:
            annotation = EMPTY
        super().__init__(name, kind, default=default_composite, annotation=annotation)

    def substitute_typevars(self, typevars: TypeVarMap) -> "SigParameter":
        if self._annotation is EMPTY:
            annotation = self._annotation
        else:
            annotation = self._annotation.substitute_typevars(typevars)
        return SigParameter(
            name=self._name,
            kind=self._kind,
            default=self._default,
            annotation=annotation,
        )

    def get_annotation(self) -> Value:
        if self.annotation is EMPTY:
            return AnyValue(AnySource.unannotated)
        return self.annotation

    def __str__(self) -> str:
        # Adapted from Parameter.__str__
        kind = self.kind
        formatted = self._name

        if self._annotation is not EMPTY:
            formatted = f"{formatted}: {self._annotation}"

        if self._default is not EMPTY:
            if self._annotation is not EMPTY:
                formatted = f"{formatted} = {self._default.value}"
            else:
                formatted = f"{formatted}={self._default.value}"

        if kind is SigParameter.VAR_POSITIONAL:
            formatted = "*" + formatted
        elif kind is SigParameter.VAR_KEYWORD:
            formatted = "**" + formatted

        return formatted


@dataclass(frozen=True)
class Signature:
    """Represents the signature of a Python callable.

    This is used to type check function calls and it powers the
    :class:`pyanalyze.value.CallableValue` class.

    """

    _return_key: ClassVar[str] = "%return"

    signature: inspect.Signature
    """The underlying :class:`inspect.Signature`, storing the parameters
    and the return annotation."""
    impl: Optional[Impl] = field(default=None, compare=False)
    """:term:`impl` function for this signature."""
    callable: Optional[Callable[..., Any]] = field(default=None, compare=False)
    """The callable that this signature represents."""
    is_asynq: bool = False
    """Whether this signature represents an asynq function."""
    has_return_annotation: bool = True
    is_ellipsis_args: bool = False
    """Whether this signature represents a ``Callable[..., T]`` callable. Such
    a callable is compatible with any other callable with a compatible return type."""
    allow_call: bool = False
    """Whether type checking can call the actual function to retrieve a precise return value."""
    typevars_of_params: Dict[str, List["TypeVar"]] = field(
        init=False, default_factory=dict, repr=False, compare=False, hash=False
    )
    all_typevars: Set["TypeVar"] = field(
        init=False, default_factory=set, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        for param_name, param in self.signature.parameters.items():
            if param.annotation is EMPTY:
                continue
            typevars = list(extract_typevars(param.annotation))
            if typevars:
                self.typevars_of_params[param_name] = typevars
        if self.signature.return_annotation is not EMPTY:
            return_typevars = list(extract_typevars(self.signature.return_annotation))
            if return_typevars:
                self.typevars_of_params[self._return_key] = return_typevars
        self.all_typevars.update(
            {
                typevar
                for tv_list in self.typevars_of_params.values()
                for typevar in tv_list
            }
        )

    def _check_param_type_compatibility(
        self,
        param: SigParameter,
        var_value: Value,
        visitor: "NameCheckVisitor",
        node: ast.AST,
        typevar_map: Optional[TypeVarMap] = None,
        is_overload: bool = False,
    ) -> Tuple[Optional[TypeVarMap], bool, Optional[Value]]:
        """Check type compatibility for a single parameter.

        Returns a three-tuple:
        - A TypeVarMap if the assignment succeeded, or None if there was an error.
        - A bool indicating whether Any was used to succeed in the assignment.
        - A Value or None, used for union decomposition with overloads.

        """
        if param.annotation is not EMPTY and not (
            isinstance(param.default, Composite) and var_value is param.default.value
        ):
            if typevar_map:
                param_typ = param.annotation.substitute_typevars(typevar_map)
            else:
                param_typ = param.annotation
            tv_map, used_any = self._can_assign_and_used_any(
                param_typ, var_value, visitor
            )
            if isinstance(tv_map, CanAssignError):
                if is_overload and isinstance(var_value, MultiValuedValue):
                    tv_maps = []
                    remaining_values = []
                    union_used_any = False
                    for val in var_value.vals:
                        can_assign, subval_used_any = self._can_assign_and_used_any(
                            param_typ, val, visitor
                        )
                        if isinstance(can_assign, CanAssignError):
                            remaining_values.append(val)
                        else:
                            if subval_used_any:
                                union_used_any = True
                            tv_maps.append(can_assign)
                    if tv_maps:
                        tv_map = unify_typevar_maps(tv_maps)
                        assert remaining_values, (
                            f"all union members matched between {param_typ} and"
                            f" {var_value}"
                        )
                        return tv_map, union_used_any, unite_values(*remaining_values)
                visitor.show_error(
                    node,
                    f"Incompatible argument type for {param.name}: expected {param_typ}"
                    f" but got {var_value}",
                    ErrorCode.incompatible_argument,
                    detail=str(tv_map),
                )
                return None, False, None
            return tv_map, used_any, None
        return {}, False, None

    def _can_assign_and_used_any(
        self, param_typ: Value, var_value: Value, visitor: "NameCheckVisitor"
    ) -> Tuple[CanAssign, bool]:
        with visitor.reset_any_used():
            tv_map = param_typ.can_assign(var_value, visitor)
            used_any = visitor.has_used_any_match()
        return tv_map, used_any

    def _apply_annotated_constraints(
        self, raw_return: Union[Value, ImplReturn], bound_args: inspect.BoundArguments
    ) -> ImplReturn:
        if isinstance(raw_return, Value):
            ret = ImplReturn(raw_return)
        else:
            ret = raw_return
        constraints = []
        if ret.constraint is not NULL_CONSTRAINT:
            constraints.append(ret.constraint)
        if isinstance(ret.return_value, AnnotatedValue):
            for guard in ret.return_value.get_metadata_of_type(
                ParameterTypeGuardExtension
            ):
                if guard.varname in bound_args.arguments:
                    composite = bound_args.arguments[guard.varname]
                    if (
                        isinstance(composite, Composite)
                        and composite.varname is not None
                    ):
                        constraint = Constraint(
                            composite.varname,
                            ConstraintType.is_value_object,
                            True,
                            guard.guarded_type,
                        )
                        constraints.append(constraint)
            for guard in ret.return_value.get_metadata_of_type(TypeGuardExtension):
                # This might miss some cases where we should use the second argument instead. We'll
                # have to come up with additional heuristics if that comes up.
                if isinstance(self.callable, MethodType) or (
                    isinstance(self.callable, FunctionType)
                    and self.callable.__name__ != self.callable.__qualname__
                ):
                    index = 1
                else:
                    index = 0
                composite = bound_args.args[index]
                if isinstance(composite, Composite) and composite.varname is not None:
                    constraint = Constraint(
                        composite.varname,
                        ConstraintType.is_value_object,
                        True,
                        guard.guarded_type,
                    )
                    constraints.append(constraint)
            for guard in ret.return_value.get_metadata_of_type(HasAttrGuardExtension):
                if guard.varname in bound_args.arguments:
                    composite = bound_args.arguments[guard.varname]
                    if (
                        isinstance(composite, Composite)
                        and composite.varname is not None
                    ):
                        constraint = Constraint(
                            composite.varname,
                            ConstraintType.add_annotation,
                            True,
                            HasAttrExtension(
                                guard.attribute_name, guard.attribute_type
                            ),
                        )
                        constraints.append(constraint)
        if constraints:
            constraint = reduce(AndConstraint, constraints)
        else:
            constraint = NULL_CONSTRAINT
        return ImplReturn(ret.return_value, constraint, ret.no_return_unless)

    def bind_arguments(
        self, actual_args: ActualArguments, visitor: "NameCheckVisitor", node: ast.AST
    ) -> Optional[BoundArgs]:
        """Attempt to bind the parameters in the signature to the arguments actually passed in.

        Nomenclature:
        - parameters: the formal parameters of the callable
        - arguments: the arguments passed in in this call
        - bound arguments: the mapping of parameter names to values produced by this call

        """
        positional_index = 0
        keywords_consumed: Set[str] = set()
        bound_args: BoundArgs = {}
        star_args_consumed = False
        star_kwargs_consumed = False

        for param in self.signature.parameters.values():
            if param.kind is SigParameter.POSITIONAL_ONLY:
                if positional_index < len(actual_args.positionals):
                    bound_args[param.name] = (
                        positional_index,
                        actual_args.positionals[positional_index],
                    )
                    positional_index += 1
                elif actual_args.star_args is not None:
                    bound_args[param.name] = None, Composite(actual_args.star_args)
                    star_args_consumed = True
                elif param.default is EMPTY:
                    self.show_call_error(
                        f"Missing required positional argument: '{param.name}'",
                        node,
                        visitor,
                    )
                    return None
                else:
                    bound_args[param.name] = None, param.default
            elif param.kind is SigParameter.POSITIONAL_OR_KEYWORD:
                if positional_index < len(actual_args.positionals):
                    bound_args[param.name] = (
                        positional_index,
                        actual_args.positionals[positional_index],
                    )
                    positional_index += 1
                    if param.name in actual_args.keywords:
                        self.show_call_error(
                            f"Parameter '{param.name}' provided as both a positional"
                            " and a keyword argument",
                            node,
                            visitor,
                        )
                        return None
                elif actual_args.star_args is not None:
                    if param.name in actual_args.keywords:
                        self.show_call_error(
                            f"Parameter '{param.name}' may be filled from both *args"
                            " and a keyword argument",
                            node,
                            visitor,
                        )
                        return None
                    star_args_consumed = True
                    # It may also come from **kwargs
                    if actual_args.star_kwargs is not None:
                        value = unite_values(
                            actual_args.star_kwargs, actual_args.star_kwargs
                        )
                        star_kwargs_consumed = True
                    else:
                        value = actual_args.star_args
                    bound_args[param.name] = None, Composite(value)
                elif param.name in actual_args.keywords:
                    definitely_provided, composite = actual_args.keywords[param.name]
                    if not definitely_provided and param.default is EMPTY:
                        self.show_call_error(
                            f"Parameter '{param.name}' may not be provided by this"
                            " call",
                            node,
                            visitor,
                        )
                        return None
                    bound_args[param.name] = param.name, composite
                    keywords_consumed.add(param.name)
                elif actual_args.star_kwargs is not None:
                    bound_args[param.name] = None, Composite(actual_args.star_kwargs)
                    star_kwargs_consumed = True
                elif param.default is EMPTY:
                    self.show_call_error(
                        f"Missing required argument: '{param.name}'", node, visitor
                    )
                    return None
                else:
                    bound_args[param.name] = None, param.default
            elif param.kind is SigParameter.KEYWORD_ONLY:
                if param.name in actual_args.keywords:
                    definitely_provided, composite = actual_args.keywords[param.name]
                    if not definitely_provided and param.default is EMPTY:
                        self.show_call_error(
                            f"Parameter '{param.name}' may not be provided by this"
                            " call",
                            node,
                            visitor,
                        )
                        return None
                    bound_args[param.name] = param.name, composite
                    keywords_consumed.add(param.name)
                elif actual_args.star_kwargs is not None:
                    bound_args[param.name] = None, Composite(actual_args.star_kwargs)
                    star_kwargs_consumed = True
                    keywords_consumed.add(param.name)
                elif param.default is EMPTY:
                    self.show_call_error(
                        f"Missing required argument: '{param.name}'", node, visitor
                    )
                    return None
                else:
                    bound_args[param.name] = None, param.default
            elif param.kind is SigParameter.VAR_POSITIONAL:
                star_args_consumed = True
                positionals = []
                while positional_index < len(actual_args.positionals):
                    positionals.append(actual_args.positionals[positional_index].value)
                    positional_index += 1
                if actual_args.star_args is not None:
                    element_value = unite_values(*positionals, actual_args.star_args)
                    star_args_value = GenericValue(tuple, [element_value])
                else:
                    star_args_value = SequenceIncompleteValue(tuple, positionals)
                bound_args[param.name] = None, Composite(star_args_value)
            elif param.kind is SigParameter.VAR_KEYWORD:
                star_kwargs_consumed = True
                items = {}
                for key, (
                    definitely_provided,
                    composite,
                ) in actual_args.keywords.items():
                    if key in keywords_consumed:
                        continue
                    items[key] = (definitely_provided, composite.value)
                if actual_args.star_kwargs is not None:
                    value_value = unite_values(
                        *(val for _, val in items.values()), actual_args.star_kwargs
                    )
                    star_kwargs_value = GenericValue(
                        dict, [TypedValue(str), value_value]
                    )
                else:
                    star_kwargs_value = TypedDictValue(items)
                bound_args[param.name] = None, Composite(star_kwargs_value)
            else:
                assert False, f"unhandled param {param.kind}"

        if not star_args_consumed and positional_index != len(actual_args.positionals):
            self.show_call_error(
                f"Takes {positional_index} positional arguments but"
                f" {len(actual_args.positionals)} were given",
                node,
                visitor,
            )
            return None
        if not star_kwargs_consumed:
            extra_kwargs = set(actual_args.keywords) - keywords_consumed
            if extra_kwargs:
                extra_kwargs_str = ", ".join(map(repr, extra_kwargs))
                if len(extra_kwargs) == 1:
                    message = f"Got an unexpected keyword argument {extra_kwargs_str}"
                else:
                    message = f"Got unexpected keyword arguments {extra_kwargs_str}"
                self.show_call_error(message, node, visitor)
                return None
        if not star_args_consumed and actual_args.star_args:
            self.show_call_error("*args provided but not used", node, visitor)
            return None
        if (
            not star_kwargs_consumed
            and actual_args.star_kwargs
            and actual_args.kwargs_required
        ):
            self.show_call_error("**kwargs provided but not used", node, visitor)
            return None
        return bound_args

    def show_call_error(
        self,
        message: str,
        node: ast.AST,
        visitor: "NameCheckVisitor",
        detail: Optional[str] = None,
    ) -> None:
        if self.callable is not None:
            message = f"In call to {stringify_object(self.callable)}: {message}"
        visitor.show_error(node, message, ErrorCode.incompatible_call, detail=detail)

    def get_default_return(self, source: AnySource = AnySource.error) -> ImplReturn:
        return_value = self.signature.return_annotation
        if return_value is EMPTY:
            return ImplReturn(AnyValue(AnySource.unannotated), is_error=True)
        if self._return_key in self.typevars_of_params:
            typevar_values = {tv: AnyValue(source) for tv in self.all_typevars}
            return_value = return_value.substitute_typevars(typevar_values)
        return ImplReturn(return_value, is_error=True)

    def check_call(
        self, args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
    ) -> ImplReturn:
        """Type check a call to this Signature with the given arguments.

        This may call the :term:`impl` function or the underlying callable,
        but normally just uses :meth:`inspect.Signature.bind`.

        """
        args = list(args)
        preprocessed = preprocess_args(args, visitor, node)
        if preprocessed is None:
            return self.get_default_return()
        return self.check_call_preprocessed(preprocessed, visitor, node)

    def check_call_preprocessed(
        self,
        preprocessed: ActualArguments,
        visitor: "NameCheckVisitor",
        node: ast.AST,
        is_overload: bool = False,
    ) -> ImplReturn:
        if self.is_ellipsis_args:
            if self.allow_call:
                runtime_return = self._maybe_perform_call(preprocessed, visitor, node)
                if runtime_return is not None:
                    return ImplReturn(runtime_return)
            return_value = self.signature.return_annotation
            if return_value is EMPTY:
                return_value = AnyValue(AnySource.unannotated)
            return ImplReturn(return_value, used_any_for_match=True)

        bound_args = self.bind_arguments(preprocessed, visitor, node)
        if bound_args is None:
            return self.get_default_return()
        variables = {key: composite.value for key, (_, composite) in bound_args.items()}
        return_value = self.signature.return_annotation
        typevar_values: Dict[TypeVar, Value] = {}
        if self.all_typevars:
            tv_possible_values: Dict[TypeVar, List[Value]] = defaultdict(list)
            for param_name in self.typevars_of_params:
                if param_name == self._return_key:
                    continue
                var_value = variables[param_name]
                param = self.signature.parameters[param_name]
                tv_map, _, _ = self._check_param_type_compatibility(
                    param, var_value, visitor, node
                )
                if tv_map is None:
                    return self.get_default_return()
                else:
                    # For now, the first assignment wins.
                    for typevar, value in tv_map.items():
                        tv_possible_values[typevar].append(value)
            typevar_values = {
                typevar: unite_values(
                    *tv_possible_values.get(
                        typevar, [AnyValue(AnySource.generic_argument)]
                    )
                )
                for typevar in self.all_typevars
            }
            if self._return_key in self.typevars_of_params:
                return_value = return_value.substitute_typevars(typevar_values)

        had_error = False
        used_any = False
        new_args = None
        for name, (position, composite) in bound_args.items():
            param = self.signature.parameters[name]
            (
                tv_map,
                param_used_any,
                remaining_value,
            ) = self._check_param_type_compatibility(
                param,
                composite.value,
                visitor,
                node,
                typevar_values,
                # If position is None we can't narrow so don't bother.
                is_overload=is_overload and position is not None,
            )
            if tv_map is None:
                had_error = True
            if param_used_any:
                used_any = True
            if remaining_value is not None:
                if isinstance(position, int):
                    new_positionals = list(preprocessed.positionals)
                    new_positionals[position] = Composite(remaining_value)
                    new_args = replace(preprocessed, positionals=new_positionals)
                elif isinstance(position, str):
                    new_kwargs = dict(preprocessed.keywords)
                    existing_required, _ = new_kwargs[position]
                    new_kwargs[position] = existing_required, Composite(remaining_value)
                    new_args = replace(preprocessed, keywords=new_kwargs)
                else:
                    assert False, "position should be set"
                # You only get to do this once per call.
                is_overload = False

        inspect_bound_args = inspect.BoundArguments(
            self.signature,
            OrderedDict(
                (param, composite) for param, (_, composite) in bound_args.items()
            ),
        )
        # don't call the implementation function if we had an error, so that
        # the implementation function doesn't have to worry about basic
        # type checking
        if not had_error and self.impl is not None:
            ctx = CallContext(
                vars=variables,
                visitor=visitor,
                bound_args=inspect_bound_args,
                node=node,
            )
            return_value = self.impl(ctx)

        if self.allow_call:
            runtime_return = self._maybe_perform_call(preprocessed, visitor, node)
            if runtime_return is not None:
                if isinstance(return_value, ImplReturn):
                    return_value = ImplReturn(
                        runtime_return,
                        return_value.constraint,
                        return_value.no_return_unless,
                    )
                else:
                    return_value = runtime_return
        if return_value is EMPTY:
            ret = ImplReturn(AnyValue(AnySource.unannotated), is_error=had_error)
        else:
            ret = self._apply_annotated_constraints(return_value, inspect_bound_args)
        return ret._replace(
            is_error=had_error,
            used_any_for_match=used_any,
            remaining_arguments=new_args,
        )

    def _maybe_perform_call(
        self, actual_args: ActualArguments, visitor: "NameCheckVisitor", node: ast.AST
    ) -> Optional[Value]:
        if self.callable is None:
            return None
        args = []
        kwargs = {}
        for composite in actual_args.positionals:
            if isinstance(composite.value, KnownValue):
                args.append(composite.value.val)
            else:
                return None
        if actual_args.star_args is not None:
            values = concrete_values_from_iterable(actual_args.star_args, visitor)
            if isinstance(values, collections.abc.Sequence) and all_of_type(
                values, KnownValue
            ):
                args += [val.val for val in values]
            else:
                return None
        for kwarg, (required, composite) in actual_args.keywords.items():
            if not required:
                return None
            if isinstance(composite.value, KnownValue):
                kwargs[kwarg] = composite.value.val
            else:
                return None
        if actual_args.star_kwargs is not None:
            value = replace_known_sequence_value(actual_args.star_kwargs)
            if isinstance(value, DictIncompleteValue):
                for pair in value.kv_pairs:
                    if (
                        pair.is_required
                        and not pair.is_many
                        and isinstance(pair.key, KnownValue)
                        and isinstance(pair.key.val, str)
                        and isinstance(pair.value, KnownValue)
                    ):
                        kwargs[pair.key.val] = pair.value.val
                    else:
                        return None
            else:
                return None

        try:
            value = self.callable(*args, **kwargs)
        except Exception as e:
            message = f"Error calling {self}: {safe_str(e)}"
            visitor._show_error_if_checking(node, message, ErrorCode.incompatible_call)
            return None
        else:
            return KnownValue(value)

    def can_assign(
        self, other: "ConcreteSignature", ctx: CanAssignContext
    ) -> CanAssign:
        """Equivalent of :meth:`pyanalyze.value.Value.can_assign`. Checks
        whether another ``Signature`` is compatible with this ``Signature``.
        """
        if isinstance(other, OverloadedSignature):
            # An overloaded signature can be assigned if any of the component signatures
            # can be assigned. Strictly, an overloaded signature could satisfy a non-overloaded
            # signature through a combination of overloads, but we make no attempt to support
            # that.
            errors = []
            for sig in other.signatures:
                can_assign = self.can_assign(sig, ctx)
                if isinstance(can_assign, CanAssignError):
                    errors.append(
                        CanAssignError(f"overload {sig} is incompatible", [can_assign])
                    )
                else:
                    return can_assign
            return CanAssignError("overloaded function is incompatible", errors)
        if self.is_asynq and not other.is_asynq:
            return CanAssignError("callable is not asynq")
        their_return = other.signature.return_annotation
        my_return = self.signature.return_annotation
        return_tv_map = my_return.can_assign(their_return, ctx)
        if isinstance(return_tv_map, CanAssignError):
            return CanAssignError(
                "return annotation is not compatible", [return_tv_map]
            )
        if self.is_ellipsis_args or other.is_ellipsis_args:
            return return_tv_map
        tv_maps = [return_tv_map]
        their_params = list(other.signature.parameters.values())
        their_args = other.get_param_of_kind(SigParameter.VAR_POSITIONAL)
        if their_args is not None:
            their_args_index = their_params.index(their_args)
            args_annotation = their_args.get_annotation()
        else:
            their_args_index = -1
            args_annotation = None
        their_kwargs = other.get_param_of_kind(SigParameter.VAR_KEYWORD)
        if their_kwargs is not None:
            kwargs_annotation = their_kwargs.get_annotation()
        else:
            kwargs_annotation = None
        consumed_positional = set()
        consumed_keyword = set()
        for i, my_param in enumerate(self.signature.parameters.values()):
            my_annotation = my_param.get_annotation()
            if my_param.kind is SigParameter.POSITIONAL_ONLY:
                if i < len(their_params) and their_params[i].kind in (
                    SigParameter.POSITIONAL_ONLY,
                    SigParameter.POSITIONAL_OR_KEYWORD,
                ):
                    if (
                        my_param.default is not EMPTY
                        and their_params[i].default is EMPTY
                    ):
                        return CanAssignError(
                            f"positional-only param {my_param.name!r} has no default"
                        )

                    their_annotation = their_params[i].get_annotation()
                    tv_map = their_annotation.can_assign(my_annotation, ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"type of positional-only parameter {my_param.name!r} is"
                            " incompatible",
                            [tv_map],
                        )
                    tv_maps.append(tv_map)
                    consumed_positional.add(their_params[i].name)
                elif args_annotation is not None:
                    new_tv_maps = can_assign_var_positional(
                        my_param, args_annotation, i - their_args_index, ctx
                    )
                    if isinstance(new_tv_maps, CanAssignError):
                        return new_tv_maps
                    tv_maps += new_tv_maps
                else:
                    return CanAssignError(
                        f"positional-only parameter {i} is not accepted"
                    )
            elif my_param.kind is SigParameter.POSITIONAL_OR_KEYWORD:
                if (
                    i < len(their_params)
                    and their_params[i].kind is SigParameter.POSITIONAL_OR_KEYWORD
                ):
                    if my_param.name != their_params[i].name:
                        return CanAssignError(
                            f"param name {their_params[i].name!r} does not match"
                            f" {my_param.name!r}"
                        )
                    if (
                        my_param.default is not EMPTY
                        and their_params[i].default is EMPTY
                    ):
                        return CanAssignError(f"param {my_param.name!r} has no default")
                    their_annotation = their_params[i].get_annotation()
                    tv_map = their_annotation.can_assign(my_annotation, ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"type of parameter {my_param.name!r} is incompatible",
                            [tv_map],
                        )
                    tv_maps.append(tv_map)
                    consumed_positional.add(their_params[i].name)
                    consumed_keyword.add(their_params[i].name)
                elif (
                    i < len(their_params)
                    and their_params[i].kind is SigParameter.POSITIONAL_ONLY
                ):
                    return CanAssignError(
                        f"parameter {my_param.name!r} is not accepted as a keyword"
                        " argument"
                    )
                elif args_annotation is not None and kwargs_annotation is not None:
                    new_tv_maps = can_assign_var_positional(
                        my_param, args_annotation, i - their_args_index, ctx
                    )
                    if isinstance(new_tv_maps, CanAssignError):
                        return new_tv_maps
                    tv_maps += new_tv_maps
                    new_tv_maps = can_assign_var_keyword(
                        my_param, kwargs_annotation, ctx
                    )
                    if isinstance(new_tv_maps, CanAssignError):
                        return new_tv_maps
                    tv_maps += new_tv_maps
                else:
                    return CanAssignError(
                        f"parameter {my_param.name!r} is not accepted"
                    )
            elif my_param.kind is SigParameter.KEYWORD_ONLY:
                their_param = other.signature.parameters.get(my_param.name)
                if their_param is not None and their_param.kind in (
                    SigParameter.POSITIONAL_OR_KEYWORD,
                    SigParameter.KEYWORD_ONLY,
                ):
                    if my_param.default is not EMPTY and their_param.default is EMPTY:
                        return CanAssignError(
                            f"keyword-only param {my_param.name!r} has no default"
                        )
                    their_annotation = their_param.get_annotation()
                    tv_map = their_annotation.can_assign(my_annotation, ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"type of parameter {my_param.name!r} is incompatible",
                            [tv_map],
                        )
                    tv_maps.append(tv_map)
                    consumed_keyword.add(their_param.name)
                elif kwargs_annotation is not None:
                    new_tv_maps = can_assign_var_keyword(
                        my_param, kwargs_annotation, ctx
                    )
                    if isinstance(new_tv_maps, CanAssignError):
                        return new_tv_maps
                    tv_maps += new_tv_maps
                else:
                    return CanAssignError(
                        f"parameter {my_param.name!r} is not accepted"
                    )
            elif my_param.kind is SigParameter.VAR_POSITIONAL:
                if args_annotation is None:
                    return CanAssignError("*args are not accepted")
                tv_map = args_annotation.can_assign(my_annotation, ctx)
                if isinstance(tv_map, CanAssignError):
                    return CanAssignError("type of *args is incompatible", [tv_map])
                tv_maps.append(tv_map)
                extra_positional = [
                    param
                    for param in their_params
                    if param.name not in consumed_positional
                    and param.kind
                    in (
                        SigParameter.POSITIONAL_ONLY,
                        SigParameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                for extra_param in extra_positional:
                    tv_map = extra_param.get_annotation().can_assign(my_annotation, ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"type of param {extra_param.name!r} is incompatible with "
                            "*args type",
                            [tv_map],
                        )
                    tv_maps.append(tv_map)
            elif my_param.kind is SigParameter.VAR_KEYWORD:
                if kwargs_annotation is None:
                    return CanAssignError("**kwargs are not accepted")
                tv_map = kwargs_annotation.can_assign(my_annotation, ctx)
                if isinstance(tv_map, CanAssignError):
                    return CanAssignError("type of **kwargs is incompatible", [tv_map])
                tv_maps.append(tv_map)
                extra_keyword = [
                    param
                    for param in their_params
                    if param.name not in consumed_keyword
                    and param.kind
                    in (SigParameter.KEYWORD_ONLY, SigParameter.POSITIONAL_OR_KEYWORD)
                ]
                for extra_param in extra_keyword:
                    tv_map = extra_param.get_annotation().can_assign(my_annotation, ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"type of param {extra_param.name!r} is incompatible with "
                            "**kwargs type",
                            [tv_map],
                        )
                    tv_maps.append(tv_map)

        for param in their_params:
            if (
                param.kind is SigParameter.VAR_POSITIONAL
                or param.kind is SigParameter.VAR_KEYWORD
            ):
                continue  # ok if they have extra *args or **kwargs
            elif param.default is not EMPTY:
                continue
            elif param.kind is SigParameter.POSITIONAL_ONLY:
                if param.name not in consumed_positional:
                    return CanAssignError(
                        f"takes extra positional-only parameter {param.name!r}"
                    )
            elif param.kind is SigParameter.POSITIONAL_OR_KEYWORD:
                if (
                    param.name not in consumed_positional
                    and param.name not in consumed_keyword
                ):
                    return CanAssignError(f"takes extra parameter {param.name!r}")
            elif param.kind is SigParameter.KEYWORD_ONLY:
                if param.name not in consumed_keyword:
                    return CanAssignError(f"takes extra parameter {param.name!r}")

        return unify_typevar_maps(tv_maps)

    def get_param_of_kind(self, kind: inspect._ParameterKind) -> Optional[SigParameter]:
        for param in self.signature.parameters.values():
            if param.kind is kind:
                return param
        return None

    def substitute_typevars(self, typevars: TypeVarMap) -> "Signature":
        return Signature(
            signature=inspect.Signature(
                [
                    param.substitute_typevars(typevars)
                    for param in self.signature.parameters.values()
                ],
                return_annotation=self.signature.return_annotation.substitute_typevars(
                    typevars
                ),
            ),
            impl=self.impl,
            callable=self.callable,
            is_asynq=self.is_asynq,
            has_return_annotation=self.has_return_annotation,
            is_ellipsis_args=self.is_ellipsis_args,
            allow_call=self.allow_call,
        )

    def walk_values(self) -> Iterable[Value]:
        yield from self.signature.return_annotation.walk_values()
        for param in self.signature.parameters.values():
            if param.annotation is not EMPTY:
                yield from param.annotation.walk_values()

    def get_asynq_value(self) -> "Signature":
        """Return the :class:`Signature` for the `.asynq` attribute of an
        :class:`pyanalyze.extensions.AsynqCallable`."""
        if not self.is_asynq:
            raise TypeError("get_asynq_value() is only supported for AsynqCallable")
        return_annotation = AsyncTaskIncompleteValue(
            asynq.AsyncTask, self.signature.return_annotation
        )
        return Signature.make(
            self.signature.parameters.values(),
            return_annotation,
            impl=self.impl,
            callable=self.callable,
            has_return_annotation=self.has_return_annotation,
            is_ellipsis_args=self.is_ellipsis_args,
            is_asynq=False,
            allow_call=self.allow_call,
        )

    @classmethod
    def make(
        cls,
        parameters: Iterable[SigParameter],
        return_annotation: Optional[Value] = None,
        *,
        impl: Optional[Impl] = None,
        callable: Optional[object] = None,
        has_return_annotation: bool = True,
        is_ellipsis_args: bool = False,
        is_asynq: bool = False,
        allow_call: bool = False,
    ) -> "Signature":
        """Create a :class:`Signature` object.

        This is more convenient to use than the constructor
        because it abstracts away the creation of the underlying
        :class:`inspect.Signature`.

        """
        if return_annotation is None:
            return_annotation = AnyValue(AnySource.unannotated)
            has_return_annotation = False
        return cls(
            signature=inspect.Signature(
                parameters, return_annotation=return_annotation
            ),
            impl=impl,
            callable=callable,
            has_return_annotation=has_return_annotation,
            is_ellipsis_args=is_ellipsis_args,
            is_asynq=is_asynq,
            allow_call=allow_call,
        )

    def __str__(self) -> str:
        param_str = ", ".join(self._render_parameters())
        asynq_str = "@asynq " if self.is_asynq else ""
        rendered = f"{asynq_str}({param_str})"
        if self.signature.return_annotation is not EMPTY:
            rendered += f" -> {self.signature.return_annotation}"
        return rendered

    def _render_parameters(self) -> Iterable[str]:
        # Adapted from Signature's own __str__
        if self.is_ellipsis_args:
            yield "..."
            return
        render_pos_only_separator = False
        render_kw_only_separator = True
        for param in self.signature.parameters.values():
            formatted = str(param)

            kind = param.kind

            if kind == SigParameter.POSITIONAL_ONLY:
                render_pos_only_separator = True
            elif render_pos_only_separator:
                yield "/"
                render_pos_only_separator = False

            if kind == SigParameter.VAR_POSITIONAL:
                render_kw_only_separator = False
            elif kind == SigParameter.KEYWORD_ONLY and render_kw_only_separator:
                yield "*"
                render_kw_only_separator = False

            yield formatted

        if render_pos_only_separator:
            yield "/"

    def bind_self(self, *, preserve_impl: bool = False) -> Optional["Signature"]:
        if self.is_ellipsis_args:
            return ANY_SIGNATURE
        params = list(self.signature.parameters.values())
        if not params or params[0].kind not in (
            SigParameter.POSITIONAL_ONLY,
            SigParameter.POSITIONAL_OR_KEYWORD,
        ):
            return None
        return Signature(
            signature=inspect.Signature(
                params[1:], return_annotation=self.return_value
            ),
            # We don't carry over the implementation function by default, because it
            # may not work when passed different arguments.
            impl=self.impl if preserve_impl else None,
            callable=self.callable,
            is_asynq=self.is_asynq,
            has_return_annotation=self.has_return_value(),
            is_ellipsis_args=self.is_ellipsis_args,
            allow_call=self.allow_call,
        )

    # TODO: do we need these?
    def has_return_value(self) -> bool:
        return self.has_return_annotation

    @property
    def return_value(self) -> Value:
        return self.signature.return_annotation


ANY_SIGNATURE = Signature.make(
    [], AnyValue(AnySource.explicit), is_ellipsis_args=True, is_asynq=True
)
""":class:`Signature` that should be compatible with any other
:class:`Signature`."""


def preprocess_args(
    args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
) -> Optional[ActualArguments]:
    """Preprocess the argument list. Produces an ActualArguments object."""

    # Step 1: Split up args and kwargs if possible.
    processed_args = []
    kwargs_requireds = []
    for arg, label in args:
        if label is None or isinstance(label, str):
            processed_args.append((arg, label))
        elif label is ARGS:
            concrete_values = concrete_values_from_iterable(arg.value, visitor)
            if isinstance(concrete_values, CanAssignError):
                visitor.show_error(
                    node,
                    f"{arg.value} is not iterable",
                    ErrorCode.incompatible_call,
                    detail=str(concrete_values),
                )
                return None
            elif isinstance(concrete_values, Value):
                # We don't know the precise types. Repack it in a tuple because
                # at this point it doesn't matter what the precise runtime type is.
                new_value = GenericValue(tuple, [concrete_values])
                new_composite = Composite(new_value, arg.varname, arg.node)
                processed_args.append((new_composite, ARGS))
            else:
                # We figured out the exact types. Treat them as separate positional
                # args.
                for subval in concrete_values:
                    processed_args.append((Composite(subval), None))
        elif label is KWARGS:
            items = {}
            extra_values = []
            # We union all the kwargs that may be provided by any union member, so that
            # we give an error if
            for subval in flatten_values(arg.value, unwrap_annotated=True):
                result = _preprocess_kwargs_no_mvv(subval, visitor, node)
                if result is None:
                    return None
                new_items, new_value = result
                if new_value is not None:
                    extra_values.append(new_value)
                for key, (required, value) in new_items.items():
                    if key in items:
                        old_required, old_value = items[key]
                        # If the item is not required in any of the dicts, we treat it as not
                        # required at the end.
                        items[key] = old_required and required, unite_values(
                            old_value, value
                        )
                    else:
                        items[key] = required, value
            for key, (required, value) in items.items():
                if required:
                    processed_args.append((Composite(value), key))
                else:
                    processed_args.append((Composite(value), PossibleKwarg(key)))
            if extra_values:
                kwargs_requireds.append(not items)
                new_value = GenericValue(
                    dict, [TypedValue(str), unite_values(*extra_values)]
                )
                # don't preserve the varname because we may have mutilated the dict
                new_composite = Composite(new_value)
                processed_args.append((new_composite, KWARGS))
        else:
            assert False, f"unhandled label {label}"

    # Step 2: enforce invariants about ARGS and KWARGS placement. We dump
    # any single argument that come after *args into *args, and we merge all *args.
    # But for keywords, we first get all the arguments with known keys, and after that unite
    # all the **kwargs into a single argument.
    more_processed_args: List[Composite] = []
    more_processed_kwargs: Dict[str, Tuple[bool, Composite]] = {}
    star_args: Optional[Value] = None
    star_kwargs: Optional[Value] = None

    for arg, label in processed_args:
        if label is None:
            # Should never happen because the parser doesn't let you
            if more_processed_kwargs or star_kwargs is not None:
                visitor.show_error(
                    node,
                    "Positional argument follow keyword arguments",
                    ErrorCode.incompatible_call,
                )
                return None
            if star_args is not None:
                star_args = unite_values(arg.value, star_args)
            else:
                more_processed_args.append(arg)
        elif label is ARGS:
            # This is legal: f(x=3, *args)
            # But this is not: f(**kwargs, **args)
            if star_kwargs is not None:
                visitor.show_error(
                    node, "*args follows **kwargs", ErrorCode.incompatible_call
                )
                return None
            if star_args is not None:
                assert isinstance(arg.value, GenericValue), repr(processed_args)
                star_args = unite_values(arg.value.args[0], star_args)
            else:
                assert isinstance(arg.value, GenericValue), repr(processed_args)
                star_args = arg.value.args[0]
        elif isinstance(label, str):
            if label in more_processed_kwargs:
                visitor.show_error(
                    node,
                    f"Multiple values provided for argument '{label}'",
                    ErrorCode.incompatible_call,
                )
                return None
            more_processed_kwargs[label] = (True, arg)
        elif isinstance(label, PossibleKwarg):
            if label.name in more_processed_kwargs:
                visitor.show_error(
                    node,
                    f"Multiple values provided for argument '{label.name}'",
                    ErrorCode.incompatible_call,
                )
                return None
            more_processed_kwargs[label.name] = (False, arg)
        elif label is KWARGS:
            assert isinstance(arg.value, GenericValue), repr(processed_args)
            new_kwargs = arg.value.args[1]
            if star_kwargs is None:
                star_kwargs = new_kwargs
            else:
                star_kwargs = unite_values(star_kwargs, new_kwargs)
        else:
            assert False, repr(label)

    return ActualArguments(
        more_processed_args,
        star_args,
        more_processed_kwargs,
        star_kwargs,
        kwargs_required=any(kwargs_requireds),
    )


def _preprocess_kwargs_no_mvv(
    value: Value, visitor: "NameCheckVisitor", node: ast.AST
) -> Optional[Tuple[Dict[str, Tuple[bool, Value]], Optional[Value]]]:
    """Preprocess a Value passed as **kwargs.

    Three possible return types:

    - None if there was a blocking error (the passed in type is not a mapping).
    - A pair of two values:
        - An {argument: (required, Value)} dict if we know the precise arguments (e.g.,
            for a TypedDict).
        - A single Value if the argument is a mapping, but we don't know all the precise keys.
            This is None if all the keys are known.

    """
    value = replace_known_sequence_value(value)
    if isinstance(value, TypedDictValue):
        return value.items, None
    elif isinstance(value, DictIncompleteValue):
        return _preprocess_kwargs_kv_pairs(value.kv_pairs, visitor, node)
    else:
        mapping_tv_map = MappingValue.can_assign(value, visitor)
        if isinstance(mapping_tv_map, CanAssignError):
            visitor.show_error(
                node,
                f"{value} is not a mapping",
                ErrorCode.incompatible_call,
                detail=str(mapping_tv_map),
            )
            return None
        key_type = mapping_tv_map.get(K, AnyValue(AnySource.generic_argument))
        value_type = mapping_tv_map.get(V, AnyValue(AnySource.generic_argument))
        return _preprocess_kwargs_kv_pairs(
            [KVPair(key_type, value_type, is_many=True)], visitor, node
        )


def _preprocess_kwargs_kv_pairs(
    items: Sequence[KVPair], visitor: "NameCheckVisitor", node: ast.AST
) -> Optional[Tuple[Dict[str, Tuple[bool, Value]], Optional[Value]]]:
    out_items = {}
    possible_values = []
    covered_keys: Set[Value] = set()
    for pair in reversed(items):
        if not pair.is_many:
            if isinstance(pair.key, AnnotatedValue):
                key = pair.key.value
            else:
                key = pair.key
            if isinstance(key, KnownValue):
                if isinstance(key.val, str):
                    if key in covered_keys:
                        continue
                    out_items[key.val] = (pair.is_required, pair.value)
                    continue
                else:
                    visitor.show_error(
                        node,
                        f"Dict passed as **kwargs contains non-string key {key.val!r}",
                        ErrorCode.incompatible_call,
                    )
                    return None

        possible_keys = []
        has_non_literal = False
        for subkey in flatten_values(pair.key, unwrap_annotated=True):
            if isinstance(subkey, KnownValue):
                if isinstance(subkey.val, str):
                    possible_keys.append(subkey.val)
                else:
                    visitor.show_error(
                        node,
                        "Dict passed as **kwargs contains non-string key"
                        f" {subkey.val!r}",
                        ErrorCode.incompatible_call,
                    )
                    return None
            else:
                can_assign = TypedValue(str).can_assign(subkey, visitor)
                if isinstance(can_assign, CanAssignError):
                    visitor.show_error(
                        node,
                        f"Dict passed as **kwargs contains non-string key {subkey!r}",
                        ErrorCode.incompatible_call,
                        detail=str(can_assign),
                    )
                    return None
                has_non_literal = True
        if possible_keys and not has_non_literal:
            for key in possible_keys:
                out_items[key] = (False, pair.value)
        else:
            possible_values.append(pair.value)
    if possible_values:
        extra_value = unite_values(*possible_values)
    else:
        extra_value = None
    return out_items, extra_value


@dataclass(frozen=True)
class OverloadedSignature:
    """Represent an overloaded function."""

    signatures: Tuple[Signature, ...]

    def __init__(self, sigs: Sequence[Signature]) -> None:
        object.__setattr__(self, "signatures", tuple(sigs))

    def check_call(
        self, args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
    ) -> ImplReturn:
        """Check a call to an overloaded function.

        The way overloads are handled is not well specified in any PEPs. Useful resources
        include:

        - Michael Lee's `specification <https://github.com/python/typing/issues/253#issuecomment-389262904>`_
          of mypy's behavior.
        - Eric Traut's `discussion <https://github.com/microsoft/pyright/issues/2521#issuecomment-956823577>`_
          of pyright's behavior.

        Our behavior is closer to mypy. The general rule is to pick the first overload that matches
        and return an error otherwise, but there are two twists: ``Any`` and unions.

        If an overload matched only due to ``Any``, we continue looking for more overloads. If there
        are other matching overloads, we return ``Any`` (with ``AnySource.multiple_overload_matches``).
        This is different from pyright's behavior: pyright picks the first overload regardless
        of ``Any``, which is unsafe in general. Returning a ``Union`` would be more precise, but
        may lead to false positives according to experience from other type checkers. A match
        "due to ``Any``" is defined as a check that succeeded because ``Any`` was on the right-hand
        side but not the left-hand side of a typecheck.

        This is implemented by setting a flag on the :class:`pyanalyze.value.CanAssignContext` when
        a type check succeeds due to ``Any``. This flag gets propagated to
        :attr:`ImplReturn.used_any_for_match`.

        If an overload does not match, but one of the arguments passed was a ``Union``, we try all
        the components of the ``Union`` separately. If some of them match, we subtract them from the
        ``Union`` and try the remaining overloads with a narrower ``Union``. In this case, we return
        a ``Union`` of the return values of all the matching overloads on success.

        The decomposition happens in the private ``_check_param_type_compatibility`` method of
        :class:`Signature`. When we perform decomposition, this method returns a
        :class:`pyanalyze.value.Value` representing the remaining union members. This value is
        then used to construct a new :class:`ActualArguments` object, which ends up in
        :attr:`ImplReturn.remaining_arguments`.

        An overload that matches without requiring use of ``Any`` or
        union decomposition is called a "clean match".

        """
        actual_args = preprocess_args(args, visitor, node)
        if actual_args is None:
            return ImplReturn(AnyValue(AnySource.error))

        errors_per_overload = []
        any_rets: List[ImplReturn] = []
        union_rets: List[ImplReturn] = []
        union_and_any_rets: List[ImplReturn] = []
        last = len(self.signatures) - 1
        for i, sig in enumerate(self.signatures):
            with visitor.catch_errors() as caught_errors:
                ret = sig.check_call_preprocessed(
                    actual_args,
                    visitor,
                    node,
                    # We set is_overload to False for the last overload
                    # because we can't do union decomposition on the last one:
                    # there's no other overload that could handle the remaining
                    # union members.
                    is_overload=i != last,
                )
            errors_per_overload.append(caught_errors)
            if ret.is_error:
                continue
            elif ret.remaining_arguments is not None:
                if ret.used_any_for_match:
                    # If an overload used both Any and union decomposition,
                    # we treat it differently from either: unlike Any matches
                    # it's not enough to prevent an error later (because it's
                    # not a full match), but unlike union matches it's enough
                    # to trigger an Any return type.
                    union_and_any_rets.append(ret)
                else:
                    union_rets.append(ret)
                actual_args = ret.remaining_arguments
            elif ret.used_any_for_match:
                any_rets.append(ret)
            else:
                # We got a clean match!
                return self._unite_rets(any_rets, union_and_any_rets, union_rets, ret)

        if any_rets:
            # We don't do this if we have union_rets, because if we got here, we
            # didn't get any clean matches. Therefore, we must have some remaining
            # union members we haven't handled.
            return self._unite_rets(any_rets, union_and_any_rets, union_rets)

        # None of the signatures matched
        errors = list(itertools.chain.from_iterable(errors_per_overload))
        codes = set(error["error_code"] for error in errors)
        if len(codes) == 1:
            (error_code,) = codes
        else:
            error_code = ErrorCode.incompatible_call
        detail = self._make_detail(errors_per_overload)
        visitor.show_error(
            node, "Cannot call overloaded function", error_code, detail=str(detail)
        )
        return ImplReturn(AnyValue(AnySource.error))

    def _unite_rets(
        self,
        any_rets: Sequence[ImplReturn],
        union_and_any_rets: Sequence[ImplReturn],
        union_rets: Sequence[ImplReturn],
        clean_ret: Optional[ImplReturn] = None,
    ) -> ImplReturn:
        if any_rets or union_and_any_rets:
            if (
                len(any_rets) == 1
                and not union_rets
                and not union_and_any_rets
                and clean_ret is None
            ):
                return any_rets[0]
            return ImplReturn(AnyValue(AnySource.multiple_overload_matches))
        elif union_rets:
            if clean_ret is not None:
                return ImplReturn.unite_impl_rets([*union_rets, clean_ret])
            return ImplReturn.unite_impl_rets(union_rets)
        assert clean_ret is not None
        return clean_ret

    def _make_detail(
        self, errors_per_overload: Sequence[Sequence[Dict[str, Any]]]
    ) -> CanAssignError:
        details = []
        for sig, errors in zip(self.signatures, errors_per_overload):
            for error in errors:
                inner = CanAssignError(
                    error["e"],
                    [CanAssignError(error["detail"])] if error["detail"] else [],
                )
                details.append(CanAssignError(f"In overload {sig}", [inner]))
        return CanAssignError(children=details)

    def substitute_typevars(self, typevars: TypeVarMap) -> "OverloadedSignature":
        return OverloadedSignature(
            [sig.substitute_typevars(typevars) for sig in self.signatures]
        )

    def bind_self(
        self, *, preserve_impl: bool = False
    ) -> Optional["OverloadedSignature"]:
        bound_sigs = [
            sig.bind_self(preserve_impl=preserve_impl) for sig in self.signatures
        ]
        if all_of_type(bound_sigs, Signature):
            return OverloadedSignature(bound_sigs)
        return None

    def has_return_value(self) -> bool:
        return all(sig.has_return_value() for sig in self.signatures)

    @property
    def return_value(self) -> Value:
        return unite_values(*[sig.return_value for sig in self.signatures])

    def __str__(self) -> str:
        sigs = ", ".join(map(str, self.signatures))
        return f"overloaded ({sigs})"

    def walk_values(self) -> Iterable[Value]:
        for sig in self.signatures:
            yield from sig.walk_values()

    def get_asynq_value(self) -> "OverloadedSignature":
        return OverloadedSignature([sig.get_asynq_value() for sig in self.signatures])

    @property
    def is_asynq(self) -> bool:
        return all(sig.is_asynq for sig in self.signatures)

    def can_assign(
        self, other: "ConcreteSignature", ctx: CanAssignContext
    ) -> CanAssign:
        # A signature can be assigned if it can be assigned to all the component signatures.
        tv_maps = []
        for sig in self.signatures:
            can_assign = sig.can_assign(other, ctx)
            if isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"{other} is incompatible with overload {sig}", [can_assign]
                )
            tv_maps.append(can_assign)
        return unify_typevar_maps(tv_maps)


ConcreteSignature = Union[Signature, OverloadedSignature]


@dataclass(frozen=True)
class BoundMethodSignature:
    """Signature for a method bound to a particular value."""

    signature: ConcreteSignature
    self_composite: Composite
    return_override: Optional[Value] = None

    def check_call(
        self, args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
    ) -> ImplReturn:
        ret = self.signature.check_call(
            [(self.self_composite, None), *args], visitor, node
        )
        if self.return_override is not None and not self.signature.has_return_value():
            return ImplReturn(
                self.return_override, ret.constraint, ret.no_return_unless
            )
        return ret

    def get_signature(
        self, *, preserve_impl: bool = False
    ) -> Optional[ConcreteSignature]:
        return self.signature.bind_self(preserve_impl=preserve_impl)

    def has_return_value(self) -> bool:
        if self.return_override is not None:
            return True
        return self.signature.has_return_value()

    @property
    def return_value(self) -> Value:
        if isinstance(self.signature, Signature) and self.signature.has_return_value():
            return self.signature.return_value
        if self.return_override is not None:
            return self.return_override
        return AnyValue(AnySource.unannotated)

    def substitute_typevars(self, typevars: TypeVarMap) -> "BoundMethodSignature":
        return BoundMethodSignature(
            self.signature.substitute_typevars(typevars),
            self.self_composite.substitute_typevars(typevars),
            self.return_override.substitute_typevars(typevars)
            if self.return_override is not None
            else None,
        )


@dataclass(frozen=True)
class PropertyArgSpec:
    """Pseudo-argspec for properties."""

    obj: object
    return_value: Value = AnyValue(AnySource.unannotated)

    def check_call(
        self, args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
    ) -> ImplReturn:
        raise TypeError("property object is not callable")

    def has_return_value(self) -> bool:
        return not isinstance(self.return_value, AnyValue)

    def substitute_typevars(self, typevars: TypeVarMap) -> "PropertyArgSpec":
        return PropertyArgSpec(
            self.obj, self.return_value.substitute_typevars(typevars)
        )


MaybeSignature = Union[
    None, Signature, BoundMethodSignature, PropertyArgSpec, OverloadedSignature
]


def make_bound_method(
    argspec: MaybeSignature,
    self_composite: Composite,
    return_override: Optional[Value] = None,
) -> Optional[BoundMethodSignature]:
    if argspec is None:
        return None
    if isinstance(argspec, (Signature, OverloadedSignature)):
        return BoundMethodSignature(argspec, self_composite, return_override)
    elif isinstance(argspec, BoundMethodSignature):
        if return_override is None:
            return_override = argspec.return_override
        return BoundMethodSignature(argspec.signature, self_composite, return_override)
    else:
        assert False, f"invalid argspec {argspec}"


T = TypeVar("T")
IterableValue = GenericValue(collections.abc.Iterable, [TypeVarValue(T)])
K = TypeVar("K")
V = TypeVar("V")
MappingValue = GenericValue(collections.abc.Mapping, [TypeVarValue(K), TypeVarValue(V)])


def can_assign_var_positional(
    my_param: SigParameter, args_annotation: Value, idx: int, ctx: CanAssignContext
) -> Union[List[TypeVarMap], CanAssignError]:
    tv_maps = []
    my_annotation = my_param.get_annotation()
    if isinstance(args_annotation, SequenceIncompleteValue):
        length = len(args_annotation.members)
        if idx >= length:
            return CanAssignError(
                f"parameter {my_param.name!r} is not accepted; {args_annotation} only"
                f" accepts {length} values"
            )
        their_annotation = args_annotation.members[idx]
        tv_map = their_annotation.can_assign(my_annotation, ctx)
        if isinstance(tv_map, CanAssignError):
            return CanAssignError(
                f"type of parameter {my_param.name!r} is incompatible: *args[{idx}]"
                " type is incompatible",
                [tv_map],
            )
        tv_maps.append(tv_map)
    else:
        tv_map = IterableValue.can_assign(args_annotation, ctx)
        if isinstance(tv_map, CanAssignError):
            return CanAssignError(
                f"{args_annotation} is not an iterable type", [tv_map]
            )
        iterable_arg = tv_map.get(T, AnyValue(AnySource.generic_argument))
        tv_map = iterable_arg.can_assign(my_annotation, ctx)
        if isinstance(tv_map, CanAssignError):
            return CanAssignError(
                f"type of parameter {my_param.name!r} is incompatible: "
                "*args type is incompatible",
                [tv_map],
            )
        tv_maps.append(tv_map)
    return tv_maps


def can_assign_var_keyword(
    my_param: SigParameter, kwargs_annotation: Value, ctx: CanAssignContext
) -> Union[List[TypeVarMap], CanAssignError]:
    my_annotation = my_param.get_annotation()
    tv_maps = []
    if isinstance(kwargs_annotation, TypedDictValue):
        if my_param.name not in kwargs_annotation.items:
            return CanAssignError(
                f"parameter {my_param.name!r} is not accepted by {kwargs_annotation}"
            )
        their_annotation = kwargs_annotation.items[my_param.name][1]
        tv_map = their_annotation.can_assign(my_annotation, ctx)
        if isinstance(tv_map, CanAssignError):
            return CanAssignError(
                f"type of parameter {my_param.name!r} is incompatible:"
                f" *kwargs[{my_param.name!r}] type is incompatible",
                [tv_map],
            )
        tv_maps.append(tv_map)
    else:
        mapping_tv_map = MappingValue.can_assign(kwargs_annotation, ctx)
        if isinstance(mapping_tv_map, CanAssignError):
            return CanAssignError(
                f"{kwargs_annotation} is not a mapping type", [mapping_tv_map]
            )
        key_arg = mapping_tv_map.get(K, AnyValue(AnySource.generic_argument))
        tv_map = key_arg.can_assign(KnownValue(my_param.name), ctx)
        if isinstance(tv_map, CanAssignError):
            return CanAssignError(
                f"parameter {my_param.name!r} is not accepted by **kwargs type",
                [tv_map],
            )
        tv_maps.append(tv_map)
        value_arg = mapping_tv_map.get(V, AnyValue(AnySource.generic_argument))
        tv_map = value_arg.can_assign(my_annotation, ctx)
        if isinstance(tv_map, CanAssignError):
            return CanAssignError(
                f"type of parameter {my_param.name!r} is incompatible: **kwargs type"
                " is incompatible",
                [tv_map],
            )
        tv_maps.append(tv_map)
    return tv_maps
