"""

The :class:`Signature` object and associated functionality. This
provides a way to represent rich callable objects and type check
calls.

"""

import ast
import collections.abc
import enum
import inspect
import itertools
from dataclasses import dataclass, field, replace
from types import FunctionType, MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Container,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import asynq
import qcore
from qcore.helpers import safe_str
from typing_extensions import Literal, Protocol, Self, assert_never

from pyanalyze.predicates import IsAssignablePredicate

from .error_code import Error, ErrorCode
from .node_visitor import Replacement
from .options import IntegerOption
from .safe import safe_getattr
from .stacked_scopes import (
    NULL_CONSTRAINT,
    AbstractConstraint,
    AndConstraint,
    Composite,
    Constraint,
    ConstraintType,
    OrConstraint,
    VarnameWithOrigin,
)
from .type_evaluation import (
    ARGS,
    DEFAULT,
    KWARGS,
    UNKNOWN,
    EvalContext,
    Evaluator,
    Position,
)
from .typevar import resolve_bounds_map
from .value import (
    NO_RETURN_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    AsyncTaskIncompleteValue,
    BoundsMap,
    CallableValue,
    CallValue,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    ConstraintExtension,
    DictIncompleteValue,
    GenericValue,
    HasAttrExtension,
    HasAttrGuardExtension,
    KnownValue,
    KVPair,
    LowerBound,
    MultiValuedValue,
    NoReturnConstraintExtension,
    NoReturnGuardExtension,
    ParameterTypeGuardExtension,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    SelfT,
    SequenceValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeGuardExtension,
    TypeIsExtension,
    TypeVarLike,
    TypeVarMap,
    TypeVarValue,
    Value,
    annotate_value,
    can_assign_and_used_any,
    concrete_values_from_iterable,
    extract_typevars,
    flatten_values,
    get_tv_map,
    is_iterable,
    replace_known_sequence_value,
    stringify_object,
    unannotate,
    unannotate_value,
    unify_bounds_maps,
    unite_values,
)

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor

EMPTY = inspect.Parameter.empty
UNANNOTATED = AnyValue(AnySource.unannotated)
ELLIPSIS = qcore.MarkerObject("ellipsis")
ELLIPSIS_COMPOSITE = Composite(AnyValue(AnySource.ellipsis_callable))

# TODO turn on
USE_CHECK_CALL_FOR_CAN_ASSIGN = False


class MaximumPositionalArgs(IntegerOption):
    """If calls have more than this many positional arguments, attempt to
    turn them into keyword arguments."""

    default_value = 10
    name = "maximum_positional_args"


class InvalidSignature(Exception):
    """Raised when an invalid signature is encountered."""


@dataclass
class PossibleArg:
    # Label used for arguments that may not be present at runtiime.
    name: Optional[str]


@dataclass
class PosOrKeyword:
    name: str
    is_required: bool


# Representation of a single argument to a call. Second member is
# None for positional args, str for keyword args,
# ARGS for *args, KWARGS for **kwargs, PossibleArg for args that may
# be missing, TypeVarValue for a ParamSpec.
Argument = Tuple[
    Composite,
    Union[
        None,
        str,
        PossibleArg,
        Literal[ARGS, KWARGS, ELLIPSIS],
        TypeVarValue,
        PosOrKeyword,
    ],
]

# Arguments bound to a call
BoundArgs = Dict[str, Tuple[Position, Composite]]


class CheckCallContext(Protocol):
    @property
    def visitor(self) -> Optional["NameCheckVisitor"]:
        raise NotImplementedError

    def on_error(
        self,
        __message: str,
        *,
        code: Error = ...,
        node: Optional[ast.AST] = ...,
        detail: Optional[str] = ...,
        replacement: Optional[Replacement] = ...,
    ) -> object:
        raise NotImplementedError

    @property
    def can_assign_ctx(self) -> CanAssignContext:
        raise NotImplementedError


@dataclass
class _CanAssignBasedContext:
    can_assign_ctx: CanAssignContext
    visitor: Optional["NameCheckVisitor"] = None
    errors: List[str] = field(default_factory=list)

    def on_error(
        self,
        message: str,
        *,
        code: Error = ErrorCode.incompatible_call,
        node: Optional[ast.AST] = None,
        detail: Optional[str] = ...,
        replacement: Optional[Replacement] = ...,
    ) -> object:
        self.errors.append(message)
        return None


@dataclass
class _VisitorBasedContext:
    visitor: "NameCheckVisitor"
    node: Optional[ast.AST]

    @property
    def can_assign_ctx(self) -> CanAssignContext:
        return self.visitor

    def on_error(
        self,
        message: str,
        *,
        code: Error = ErrorCode.incompatible_call,
        node: Optional[ast.AST] = None,
        detail: Optional[str] = ...,
        replacement: Optional[Replacement] = None,
    ) -> None:
        if node is None:
            node = self.node
        if node is None:
            return
        self.visitor.show_error(
            node, message, code, detail=detail, replacement=replacement
        )


@dataclass
class ActualArguments:
    """Represents the actual arguments to a call.

    Before creating this class, we decompose ``*args`` and ``**kwargs`` arguments
    of known composition into additional positional and keyword arguments, and we
    merge multiple ``*args`` or ``**kwargs``.

    Creating the ``ActualArguments`` for a call is independent of the signature
    of the callee.

    """

    positionals: List[Tuple[bool, Composite]]
    star_args: Optional[Value]  # represents the type of the elements of *args
    keywords: Dict[str, Tuple[bool, Composite]]
    star_kwargs: Optional[Value]  # represents the type of the elements of **kwargs
    kwargs_required: bool
    pos_or_keyword_params: Container[Union[int, str]]
    ellipsis: bool = False
    param_spec: Optional[TypeVarValue] = None


class CallReturn(NamedTuple):
    """Return value of a preprocessed call.

    This returns data that is useful for overload resolution.

    """

    return_value: Value
    """The return value of the function."""
    sig: "Signature"
    """Signature that was used for this call."""
    is_error: bool = False
    """Whether there was an error in this call. Used only for overload resolutioon."""
    used_any_for_match: bool = False
    """Whether Any was used for this match. Used only for overload resolution."""
    remaining_arguments: Optional[ActualArguments] = None
    """Arguments that still need to be processed. Used only for overload resolution."""


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

    @classmethod
    def unite_impl_rets(cls, rets: Sequence["ImplReturn"]) -> "ImplReturn":
        if not rets:
            return ImplReturn(NO_RETURN_VALUE)
        return ImplReturn(
            unite_values(*[r.return_value for r in rets]),
            OrConstraint.make([r.constraint for r in rets]),
            OrConstraint.make([r.no_return_unless for r in rets]),
        )


@dataclass
class CallContext:
    """The context passed to an :term:`impl` function."""

    vars: Dict[str, Value]
    """Dictionary of variable names passed to the function."""
    visitor: "NameCheckVisitor"
    """Using the visitor can allow various kinds of advanced logic
    in impl functions."""
    composites: Dict[str, Composite]
    node: Optional[ast.AST]
    """AST node corresponding to the function call. Useful for
    showing errors."""

    def ast_for_arg(self, arg: str) -> Optional[ast.AST]:
        composite = self.composite_for_arg(arg)
        if composite is not None:
            return composite.node
        return None

    def varname_for_arg(self, arg: str) -> Optional[VarnameWithOrigin]:
        """Return a :term:`varname` corresponding to the given function argument.

        This is useful for creating a :class:`pyanalyze.stacked_scopes.Constraint`
        referencing the argument.

        """
        composite = self.composite_for_arg(arg)
        if composite is not None:
            return composite.varname
        return None

    def composite_for_arg(self, arg: str) -> Optional[Composite]:
        composite = self.composites.get(arg)
        if isinstance(composite, Composite):
            return composite
        return None

    def show_error(
        self,
        message: str,
        error_code: Error = ErrorCode.incompatible_call,
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


class ParameterKind(enum.Enum):
    """Kinds of parameters."""

    # Values must match inspect._ParameterKind
    POSITIONAL_ONLY = 0
    POSITIONAL_OR_KEYWORD = 1
    VAR_POSITIONAL = 2
    KEYWORD_ONLY = 3
    VAR_KEYWORD = 4
    PARAM_SPEC = 5
    ELLIPSIS = 6  # Callable[..., T]
    """Special kind for `Callable[..., T]`. Such callables are compatible
    with any other callable. There can only be one ELLIPSIS parameter
    and it must be the last one."""

    def allow_unpack(self) -> bool:
        return self is ParameterKind.VAR_KEYWORD or self is ParameterKind.VAR_POSITIONAL


KIND_TO_ALLOWED_PREVIOUS = {
    ParameterKind.POSITIONAL_ONLY: {ParameterKind.POSITIONAL_ONLY},
    ParameterKind.POSITIONAL_OR_KEYWORD: {
        ParameterKind.POSITIONAL_ONLY,
        ParameterKind.POSITIONAL_OR_KEYWORD,
    },
    ParameterKind.VAR_POSITIONAL: {
        ParameterKind.POSITIONAL_OR_KEYWORD,
        ParameterKind.POSITIONAL_ONLY,
    },
    ParameterKind.KEYWORD_ONLY: {
        ParameterKind.POSITIONAL_ONLY,
        ParameterKind.POSITIONAL_OR_KEYWORD,
        ParameterKind.VAR_POSITIONAL,
        ParameterKind.KEYWORD_ONLY,
    },
    ParameterKind.VAR_KEYWORD: {
        ParameterKind.POSITIONAL_ONLY,
        ParameterKind.POSITIONAL_OR_KEYWORD,
        ParameterKind.VAR_POSITIONAL,
        ParameterKind.KEYWORD_ONLY,
    },
    ParameterKind.PARAM_SPEC: {
        ParameterKind.POSITIONAL_ONLY,
        ParameterKind.POSITIONAL_OR_KEYWORD,
    },
    ParameterKind.ELLIPSIS: {ParameterKind.POSITIONAL_ONLY},
}
CAN_HAVE_DEFAULT = {
    ParameterKind.POSITIONAL_ONLY,
    ParameterKind.POSITIONAL_OR_KEYWORD,
    ParameterKind.KEYWORD_ONLY,
}


@dataclass(frozen=True)
class SigParameter:
    """Represents a single parameter to a callable."""

    name: str
    """Name of the parameter."""
    kind: ParameterKind = ParameterKind.POSITIONAL_OR_KEYWORD
    """How the parameter can be passed."""
    default: Optional[Value] = None
    """The default for the parameter, or None if there is no default."""
    annotation: Value = AnyValue(AnySource.unannotated)
    """Type annotation for the parameter."""

    # For compatibility
    empty: ClassVar[Literal[EMPTY]] = EMPTY
    POSITIONAL_ONLY: ClassVar[Literal[ParameterKind.POSITIONAL_ONLY]] = (
        ParameterKind.POSITIONAL_ONLY
    )
    POSITIONAL_OR_KEYWORD: ClassVar[Literal[ParameterKind.POSITIONAL_OR_KEYWORD]] = (
        ParameterKind.POSITIONAL_OR_KEYWORD
    )
    VAR_POSITIONAL: ClassVar[Literal[ParameterKind.VAR_POSITIONAL]] = (
        ParameterKind.VAR_POSITIONAL
    )
    KEYWORD_ONLY: ClassVar[Literal[ParameterKind.KEYWORD_ONLY]] = (
        ParameterKind.KEYWORD_ONLY
    )
    VAR_KEYWORD: ClassVar[Literal[ParameterKind.VAR_KEYWORD]] = (
        ParameterKind.VAR_KEYWORD
    )

    def __post_init__(self) -> None:
        # backward compatibility
        if self.default is EMPTY:  # static analysis: ignore[unsafe_comparison]
            object.__setattr__(self, "default", None)

    def substitute_typevars(self, typevars: TypeVarMap) -> "SigParameter":
        return SigParameter(
            name=self.name,
            kind=self.kind,
            default=self.default,
            annotation=self.annotation.substitute_typevars(typevars),
        )

    def get_annotation(self) -> Value:
        return self.annotation

    def is_unnamed(self) -> bool:
        return self.name.startswith("@")

    def __str__(self) -> str:
        # Adapted from Parameter.__str__
        kind = self.kind
        if self.is_unnamed():
            if self.default is None:
                formatted = str(self.annotation)
            else:
                formatted = f"{self.annotation} = {self.default}"
        else:
            formatted = self.name

            if self.annotation != UNANNOTATED:
                formatted = f"{formatted}: {self.annotation}"

            if self.default is not None:
                if self.annotation != UNANNOTATED:
                    formatted = f"{formatted} = {self.default}"
                else:
                    formatted = f"{formatted}={self.default}"

        if kind is ParameterKind.VAR_POSITIONAL:
            formatted = "*" + formatted
        elif kind is ParameterKind.VAR_KEYWORD or kind is ParameterKind.PARAM_SPEC:
            formatted = "**" + formatted

        return formatted

    def to_argument(self) -> Argument:
        val = Composite(self.annotation)
        if self.kind is ParameterKind.ELLIPSIS:
            return val, ELLIPSIS
        elif self.kind is ParameterKind.PARAM_SPEC:
            assert isinstance(self.annotation, TypeVarValue)
            return val, self.annotation
        elif self.kind is ParameterKind.VAR_KEYWORD:
            return val, KWARGS
        elif self.kind is ParameterKind.VAR_POSITIONAL:
            return val, ARGS
        elif self.kind is ParameterKind.POSITIONAL_ONLY:
            return val, PossibleArg(None) if self.default is not None else None
        elif self.kind is ParameterKind.KEYWORD_ONLY:
            return (
                val,
                PossibleArg(self.name) if self.default is not None else self.name,
            )
        elif self.kind is ParameterKind.POSITIONAL_OR_KEYWORD:
            return val, PosOrKeyword(self.name, self.default is None)
        else:
            assert False, self.kind


@dataclass(frozen=True)
class Signature:
    """Represents the signature of a Python callable.

    This is used to type check function calls and it powers the
    :class:`pyanalyze.value.CallableValue` class.

    """

    _return_key: ClassVar[str] = "%return"

    parameters: Dict[str, SigParameter]
    """An ordered mapping of the signature's parameters."""
    return_value: Value
    """What the callable returns."""
    impl: Optional[Impl] = field(default=None, compare=False)
    """:term:`impl` function for this signature."""
    callable: Optional[object] = field(default=None, compare=False)
    """The callable that this signature represents."""
    is_asynq: bool = False
    """Whether this signature represents an asynq function."""
    has_return_annotation: bool = True
    allow_call: bool = False
    """Whether type checking can call the actual function to retrieve a precise return value."""
    evaluator: Optional[Evaluator] = None
    """Type evaluator for this function."""
    deprecated: Optional[str] = None
    """Deprecation message for this callable."""
    typevars_of_params: Dict[str, List[TypeVarLike]] = field(
        init=False, default_factory=dict, repr=False, compare=False, hash=False
    )
    all_typevars: Set[TypeVarLike] = field(
        init=False, default_factory=set, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        for param_name, param in self.parameters.items():
            typevars = list(extract_typevars(param.annotation))
            if typevars:
                self.typevars_of_params[param_name] = typevars
        return_typevars = list(extract_typevars(self.return_value))
        if return_typevars:
            self.typevars_of_params[self._return_key] = return_typevars
        self.all_typevars.update(
            {
                typevar
                for tv_list in self.typevars_of_params.values()
                for typevar in tv_list
            }
        )
        self.validate()

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.parameters.items()),
                self.return_value,
                self.impl,
                self.callable,
                self.is_asynq,
                self.has_return_annotation,
                self.allow_call,
                self.evaluator,
            )
        )

    def validate(self) -> None:
        seen_kinds = set()
        seen_with_default = set()
        for name, param in self.parameters.items():
            if name != param.name:
                raise InvalidSignature(f"names {name} and {param.name} do not match")
            disallowed_previous = seen_kinds - KIND_TO_ALLOWED_PREVIOUS[param.kind]
            if disallowed_previous:
                disallowed_text = ", ".join(kind.name for kind in disallowed_previous)
                raise InvalidSignature(
                    f"param {param} of kind {param.kind.name} may not follow param of"
                    f" kind {disallowed_text} {self.parameters}"
                )
            if param.default is not None and param.kind not in CAN_HAVE_DEFAULT:
                raise InvalidSignature(
                    f"param {param} of kind {param.kind.name} may not have a default"
                )
            if param.default is None:
                if param.kind is ParameterKind.POSITIONAL_ONLY:
                    if ParameterKind.POSITIONAL_ONLY in seen_with_default:
                        raise InvalidSignature(
                            f"param {param} has no default but follows a param with a"
                            " default"
                        )
                elif param.kind is ParameterKind.POSITIONAL_OR_KEYWORD:
                    if seen_with_default & {
                        ParameterKind.POSITIONAL_ONLY,
                        ParameterKind.POSITIONAL_OR_KEYWORD,
                    }:
                        raise InvalidSignature(
                            f"param {param} has no default but follows a param with a"
                            " default"
                        )

            seen_kinds.add(param.kind)
            if param.default is not None:
                seen_with_default.add(param.kind)

    def _check_param_type_compatibility(
        self,
        param: SigParameter,
        composite: Composite,
        ctx: CheckCallContext,
        typevar_map: Optional[TypeVarMap] = None,
        is_overload: bool = False,
    ) -> Tuple[Optional[BoundsMap], bool, Optional[Value]]:
        """Check type compatibility for a single parameter.

        Returns a three-tuple:
        - A BoundsMap if the assignment succeeded, or None if there was an error.
        - A bool indicating whether Any was used to succeed in the assignment.
        - A Value or None, used for union decomposition with overloads.

        """
        if param.annotation != UNANNOTATED:
            if typevar_map:
                param_typ = param.annotation.substitute_typevars(typevar_map)
            else:
                param_typ = param.annotation
            bounds_map, used_any = can_assign_and_used_any(
                param_typ, composite.value, ctx.can_assign_ctx
            )
            if composite.value is param.default:
                used_any = False
            if isinstance(bounds_map, CanAssignError):
                if composite.value is param.default:
                    bounds_map = {}
                else:
                    if is_overload:
                        triple = decompose_union(
                            param_typ, composite.value, ctx.can_assign_ctx
                        )
                        if triple is not None:
                            return triple
                    ctx.on_error(
                        f"Incompatible argument type for {param.name}: expected"
                        f" {param_typ} but got {composite.value}",
                        code=bounds_map.get_error_code()
                        or ErrorCode.incompatible_argument,
                        node=composite.node if composite.node is not None else None,
                        detail=str(bounds_map),
                    )
                    return None, False, None
            return bounds_map, used_any, None
        return {}, False, None

    def _get_positional_parameter(self, index: int) -> Optional[SigParameter]:
        for i, param in enumerate(self.parameters.values()):
            if param.kind in (
                ParameterKind.VAR_KEYWORD,
                ParameterKind.VAR_POSITIONAL,
                ParameterKind.KEYWORD_ONLY,
            ):
                return None
            if i == index:
                return param
        return None

    def _apply_annotated_constraints(
        self,
        raw_return: Union[Value, ImplReturn],
        composites: Dict[str, Composite],
        ctx: CheckCallContext,
    ) -> Value:
        if isinstance(raw_return, Value):
            ret = ImplReturn(raw_return)
        else:
            ret = raw_return
        constraints = []
        if ret.constraint is not NULL_CONSTRAINT:
            constraints.append(ret.constraint)
        return_value = ret.return_value
        no_return_unless = ret.no_return_unless
        if isinstance(return_value, AnnotatedValue):
            return_value, ptg = unannotate_value(
                return_value, ParameterTypeGuardExtension
            )
            for guard in ptg:
                if guard.varname in composites:
                    composite = composites[guard.varname]
                    if composite.varname is not None:
                        constraint = Constraint(
                            composite.varname,
                            ConstraintType.is_value_object,
                            True,
                            guard.guarded_type,
                        )
                        constraints.append(constraint)

            return_value, tg = unannotate_value(return_value, TypeGuardExtension)
            for guard in tg:
                varname = self._get_typeguard_varname(composites)
                if varname is not None:
                    constraint = Constraint(
                        varname,
                        ConstraintType.is_value_object,
                        True,
                        guard.guarded_type,
                    )
                    constraints.append(constraint)

            return_value, ti = unannotate_value(return_value, TypeIsExtension)
            for guard in ti:
                varname = self._get_typeguard_varname(composites)
                if varname is not None and ctx.visitor is not None:
                    predicate = IsAssignablePredicate(
                        guard.guarded_type, ctx.visitor, positive_only=False
                    )
                    constraint = Constraint(
                        varname, ConstraintType.predicate, True, predicate
                    )
                    constraints.append(constraint)

            return_value, hag = unannotate_value(return_value, HasAttrGuardExtension)
            for guard in hag:
                if guard.varname in composites:
                    composite = composites[guard.varname]
                    if composite.varname is not None:
                        constraint = Constraint(
                            composite.varname,
                            ConstraintType.add_annotation,
                            True,
                            HasAttrExtension(
                                guard.attribute_name, guard.attribute_type
                            ),
                        )
                        constraints.append(constraint)

            return_value, nrg = unannotate_value(return_value, NoReturnGuardExtension)
            extra_nru = []
            for guard in nrg:
                if guard.varname in composites:
                    composite = composites[guard.varname]
                    if composite.varname is not None:
                        constraint = Constraint(
                            composite.varname,
                            ConstraintType.is_value_object,
                            True,
                            guard.guarded_type,
                        )
                        extra_nru.append(constraint)
            if extra_nru:
                no_return_unless = AndConstraint.make([no_return_unless, *extra_nru])

        constraint = AndConstraint.make(constraints)
        extensions = []
        if constraint is not NULL_CONSTRAINT:
            extensions.append(ConstraintExtension(constraint))
        if no_return_unless is not NULL_CONSTRAINT:
            extensions.append(NoReturnConstraintExtension(no_return_unless))
        return annotate_value(return_value, extensions)

    def _get_typeguard_varname(
        self, composites: Dict[str, Composite]
    ) -> Optional[VarnameWithOrigin]:
        # This might miss some cases where we should use the second argument instead. We'll
        # have to come up with additional heuristics if that comes up.
        if isinstance(self.callable, MethodType) or (
            isinstance(self.callable, FunctionType)
            and self.callable.__name__ != self.callable.__qualname__
        ):
            index = 1
        else:
            index = 0
        param = self._get_positional_parameter(index)
        if param is not None:
            composite = composites[param.name]
            if composite.varname is not None:
                return composite.varname
        return None

    def bind_arguments(
        self, actual_args: ActualArguments, ctx: CheckCallContext
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
        param_spec_consumed = False

        for param in self.parameters.values():
            if param.kind is ParameterKind.POSITIONAL_ONLY:
                if positional_index < len(actual_args.positionals):
                    if positional_index in actual_args.pos_or_keyword_params:
                        self.show_call_error(
                            f"Positional parameter {positional_index} should be"
                            " positional-or-keyword",
                            ctx,
                        )
                        return None
                    definitely_provided, composite = actual_args.positionals[
                        positional_index
                    ]
                    if (
                        not definitely_provided
                        and param.default is None
                        and not actual_args.ellipsis
                    ):
                        self.show_call_error(
                            f"Parameter '{param.name}' may not be provided by this"
                            " call",
                            ctx,
                        )
                        return None
                    bound_args[param.name] = (positional_index, composite)
                    positional_index += 1
                elif actual_args.star_args is not None:
                    if param.default is None:
                        position = ARGS  # either that or the call fails
                    else:
                        position = UNKNOWN  # default or args
                    bound_args[param.name] = position, Composite(actual_args.star_args)
                    star_args_consumed = True
                elif param.default is not None:
                    bound_args[param.name] = DEFAULT, Composite(param.default)
                elif actual_args.ellipsis:
                    bound_args[param.name] = UNKNOWN, ELLIPSIS_COMPOSITE
                else:
                    if param.is_unnamed():
                        message = (
                            "Missing required positional argument at position"
                            f" {int(param.name[1:])}"
                        )
                    else:
                        message = f"Missing required positional argument '{param.name}'"
                    self.show_call_error(message, ctx)
                    return None
            elif param.kind is ParameterKind.POSITIONAL_OR_KEYWORD:
                if positional_index < len(actual_args.positionals):
                    definitely_provided, composite = actual_args.positionals[
                        positional_index
                    ]
                    if (
                        not definitely_provided
                        and param.default is None
                        and not actual_args.ellipsis
                    ):
                        self.show_call_error(
                            f"Parameter '{param.name}' may not be provided by this"
                            " call",
                            ctx,
                        )
                        return None
                    bound_args[param.name] = (positional_index, composite)
                    positional_index += 1
                    if param.name in actual_args.keywords:
                        if param.name in actual_args.pos_or_keyword_params:
                            keywords_consumed.add(param.name)
                        else:
                            self.show_call_error(
                                f"Parameter '{param.name}' provided as both a"
                                " positional and a keyword argument",
                                ctx,
                            )
                            return None
                elif actual_args.star_args is not None:
                    if param.name in actual_args.keywords:
                        self.show_call_error(
                            f"Parameter '{param.name}' may be filled from both"
                            " *args and a keyword argument",
                            ctx,
                        )
                        return None
                    star_args_consumed = True
                    if param.default is None:
                        position = ARGS
                    else:
                        position = UNKNOWN
                    # It may also come from **kwargs
                    if actual_args.star_kwargs is not None:
                        value = unite_values(
                            actual_args.star_args, actual_args.star_kwargs
                        )
                        star_kwargs_consumed = True
                        position = UNKNOWN  # could be either args or kwargs
                    else:
                        value = actual_args.star_args
                    bound_args[param.name] = position, Composite(value)
                elif param.name in actual_args.keywords:
                    definitely_provided, composite = actual_args.keywords[param.name]
                    if (
                        not definitely_provided
                        and param.default is None
                        and not actual_args.ellipsis
                    ):
                        self.show_call_error(
                            f"Parameter '{param.name}' may not be provided by this"
                            " call",
                            ctx,
                        )
                        return None
                    bound_args[param.name] = param.name, composite
                    keywords_consumed.add(param.name)
                elif actual_args.star_kwargs is not None:
                    if param.default is None:
                        position = KWARGS
                    else:
                        position = UNKNOWN
                    bound_args[param.name] = position, Composite(
                        actual_args.star_kwargs
                    )
                    star_kwargs_consumed = True
                elif param.default is not None:
                    bound_args[param.name] = DEFAULT, Composite(param.default)
                elif actual_args.ellipsis:
                    bound_args[param.name] = DEFAULT, ELLIPSIS_COMPOSITE
                else:
                    self.show_call_error(
                        f"Missing required argument '{param.name}'", ctx
                    )
                    return None
            elif param.kind is ParameterKind.KEYWORD_ONLY:
                if param.name in actual_args.keywords:
                    if param.name in actual_args.pos_or_keyword_params:
                        self.show_call_error(
                            f"Keyword parameter {param.name} should be"
                            " positional-or-keyword",
                            ctx,
                        )
                        return None
                    definitely_provided, composite = actual_args.keywords[param.name]
                    if (
                        not definitely_provided
                        and param.default is None
                        and not actual_args.ellipsis
                    ):
                        self.show_call_error(
                            f"Parameter '{param.name}' may not be provided by this"
                            " call",
                            ctx,
                        )
                        return None
                    bound_args[param.name] = param.name, composite
                    keywords_consumed.add(param.name)
                elif actual_args.star_kwargs is not None:
                    if param.default is None:
                        position = KWARGS
                    else:
                        position = UNKNOWN
                    bound_args[param.name] = position, Composite(
                        actual_args.star_kwargs
                    )
                    star_kwargs_consumed = True
                    keywords_consumed.add(param.name)
                elif param.default is not None:
                    bound_args[param.name] = DEFAULT, Composite(param.default)
                elif actual_args.ellipsis:
                    bound_args[param.name] = DEFAULT, ELLIPSIS_COMPOSITE
                else:
                    self.show_call_error(
                        f"Missing required argument '{param.name}'", ctx
                    )
                    return None
            elif param.kind is ParameterKind.VAR_POSITIONAL:
                star_args_consumed = True
                positionals = []
                while positional_index < len(actual_args.positionals):
                    positionals.append(
                        actual_args.positionals[positional_index][1].value
                    )
                    positional_index += 1
                position = ARGS
                if actual_args.ellipsis:
                    star_args_value = GenericValue(
                        tuple, [AnyValue(AnySource.ellipsis_callable)]
                    )
                elif actual_args.star_args is not None:
                    star_args_value = SequenceValue(
                        tuple,
                        [
                            *[(False, pos) for pos in positionals],
                            (True, actual_args.star_args),
                        ],
                    )
                else:
                    star_args_value = SequenceValue(
                        tuple, [(False, pos) for pos in positionals]
                    )
                    if not positionals:
                        # no *args were actually provided
                        position = DEFAULT
                bound_args[param.name] = position, Composite(star_args_value)
            elif param.kind is ParameterKind.VAR_KEYWORD:
                star_kwargs_consumed = True
                items = {}
                for key, (
                    definitely_provided,
                    composite,
                ) in actual_args.keywords.items():
                    if key in keywords_consumed:
                        continue
                    items[key] = TypedDictEntry(
                        composite.value, required=definitely_provided
                    )
                position = KWARGS
                if actual_args.ellipsis:
                    star_kwargs_value = GenericValue(
                        dict, [TypedValue(str), AnyValue(AnySource.ellipsis_callable)]
                    )
                elif actual_args.star_kwargs is not None:
                    value_value = unite_values(
                        *(entry.typ for entry in items.values()),
                        actual_args.star_kwargs,
                    )
                    star_kwargs_value = GenericValue(
                        dict, [TypedValue(str), value_value]
                    )
                else:
                    star_kwargs_value = TypedDictValue(items)
                    if not items:
                        position = DEFAULT
                bound_args[param.name] = position, Composite(star_kwargs_value)
            elif param.kind is ParameterKind.ELLIPSIS:
                # just take it all
                star_args_consumed = True
                star_kwargs_consumed = True
                param_spec_consumed = True
                val = AnyValue(AnySource.ellipsis_callable)
                bound_args[param.name] = UNKNOWN, Composite(val)
            elif param.kind is ParameterKind.PARAM_SPEC:
                if actual_args.param_spec is not None:
                    bound_args[param.name] = KWARGS, Composite(actual_args.param_spec)
                    param_spec_consumed = True
                elif (
                    actual_args.star_args is not None
                    and actual_args.star_kwargs is not None
                    and not star_args_consumed
                    and not star_kwargs_consumed
                    and isinstance(actual_args.star_args, ParamSpecArgsValue)
                    and isinstance(actual_args.star_kwargs, ParamSpecKwargsValue)
                    and actual_args.star_args.param_spec
                    is actual_args.star_kwargs.param_spec
                ):
                    star_kwargs_consumed = True
                    star_args_consumed = True
                    composite = Composite(
                        TypeVarValue(
                            actual_args.star_kwargs.param_spec, is_paramspec=True
                        )
                    )
                    bound_args[param.name] = KWARGS, composite
                else:
                    new_actuals = ActualArguments(
                        positionals=actual_args.positionals[positional_index:],
                        star_args=(
                            actual_args.star_args if not star_args_consumed else None
                        ),
                        keywords={
                            key: value
                            for key, value in actual_args.keywords.items()
                            if key not in keywords_consumed
                        },
                        star_kwargs=(
                            actual_args.star_kwargs
                            if not star_kwargs_consumed
                            else None
                        ),
                        kwargs_required=actual_args.kwargs_required,
                        pos_or_keyword_params=actual_args.pos_or_keyword_params,
                    )
                    star_args_consumed = True
                    star_kwargs_consumed = True
                    val = CallValue(new_actuals)
                    bound_args[param.name] = UNKNOWN, Composite(val)
            else:
                assert False, f"unhandled param {param.kind}"

        if not star_args_consumed and positional_index != len(actual_args.positionals):
            self.show_call_error(
                f"Takes {positional_index} positional arguments but"
                f" {len(actual_args.positionals)} were given",
                ctx,
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
                self.show_call_error(message, ctx)
                return None
        if not star_args_consumed and actual_args.star_args:
            self.show_call_error("*args provided but not used", ctx)
            return None
        if (
            not star_kwargs_consumed
            and actual_args.star_kwargs
            and actual_args.kwargs_required
        ):
            self.show_call_error("**kwargs provided but not used", ctx)
            return None
        if not param_spec_consumed and actual_args.param_spec is not None:
            self.show_call_error("ParamSpec provided but not used", ctx)
        return bound_args

    def show_call_error(
        self,
        message: str,
        ctx: CheckCallContext,
        *,
        node: Optional[ast.AST] = None,
        detail: Optional[str] = None,
    ) -> None:
        if self.callable is not None:
            message = f"In call to {stringify_object(self.callable)}: {message}"
        ctx.on_error(message, node=node, detail=detail)

    def get_default_return(self, source: AnySource = AnySource.error) -> CallReturn:
        return_value = self.return_value
        if self._return_key in self.typevars_of_params:
            typevar_values = {tv: AnyValue(source) for tv in self.all_typevars}
            return_value = return_value.substitute_typevars(typevar_values)
        return CallReturn(return_value, is_error=True, sig=self)

    def check_call(
        self,
        args: Iterable[Argument],
        visitor: "NameCheckVisitor",
        node: Optional[ast.AST],
    ) -> Value:
        """Type check a call to this Signature with the given arguments.

        This may call the :term:`impl` function or the underlying callable,
        but normally just uses :meth:`inspect.Signature.bind`.

        """
        args = list(args)
        ctx = _VisitorBasedContext(visitor, node)
        preprocessed = preprocess_args(args, ctx)
        if preprocessed is None:
            return self.get_default_return().return_value
        return self.check_call_preprocessed(
            preprocessed, ctx, original_args=args, node=node
        ).return_value

    def maybe_show_too_many_pos_args_error(
        self,
        *,
        args: Sequence[Argument],
        bound_args: BoundArgs,
        ctx: CheckCallContext,
        node: ast.Call,
    ) -> None:
        """Show an error if the call to this Signature has too many positional arguments."""
        if ctx.visitor is None:
            return
        if len(node.args) < ctx.visitor.options.get_value_for(MaximumPositionalArgs):
            return
        composite_to_name = {}
        for name, (kind, composite) in bound_args.items():
            if isinstance(kind, int):
                composite_to_name[composite] = name
        node_to_composite = {}
        for unbound_arg, kind in args:
            if kind is None and unbound_arg.node is not None:
                node_to_composite[unbound_arg.node] = unbound_arg

        new_args = []
        new_keywords = []
        for arg in node.args:
            if arg not in node_to_composite:
                return
            composite = node_to_composite[arg]
            if composite not in composite_to_name:
                return
            name = composite_to_name[composite]
            new_keywords.append(ast.keyword(arg=name, value=arg))
        new_keywords += node.keywords
        new_node = ast.Call(func=node.func, args=new_args, keywords=new_keywords)
        ctx.visitor.show_error(
            node,
            f"Too many positional arguments for {stringify_object(self.callable)}",
            error_code=ErrorCode.too_many_positional_args,
            replacement=ctx.visitor.replace_node(node, new_node),
        )

    def check_call_preprocessed(
        self,
        preprocessed: ActualArguments,
        ctx: CheckCallContext,
        *,
        is_overload: bool = False,
        original_args: Optional[Sequence[Argument]] = None,
        node: Optional[ast.AST] = None,
    ) -> CallReturn:
        bound_args = self.bind_arguments(preprocessed, ctx)
        if bound_args is None:
            return self.get_default_return()
        if original_args is not None and isinstance(node, ast.Call):
            self.maybe_show_too_many_pos_args_error(
                args=original_args, bound_args=bound_args, ctx=ctx, node=node
            )
        return self.check_call_with_bound_args(
            preprocessed, bound_args, ctx, is_overload=is_overload
        )

    def check_call_with_bound_args(
        self,
        preprocessed: ActualArguments,
        bound_args: BoundArgs,
        ctx: CheckCallContext,
        *,
        is_overload: bool = False,
    ) -> CallReturn:
        variables = {key: composite.value for key, (_, composite) in bound_args.items()}

        if self.callable is not None and ctx.visitor is not None:
            ctx.visitor.record_call(self.callable, variables)

        return_value = self.return_value
        typevar_values: TypeVarMap = {}
        if self.all_typevars:
            bounds_maps = []
            for param_name in self.typevars_of_params:
                if param_name == self._return_key:
                    continue
                param = self.parameters[param_name]
                bounds_map, _, _ = self._check_param_type_compatibility(
                    param, bound_args[param_name][1], ctx
                )
                if bounds_map is None:
                    return self.get_default_return()
                else:
                    bounds_maps.append(bounds_map)
            typevar_values, errors = resolve_bounds_map(
                unify_bounds_maps(bounds_maps),
                ctx.can_assign_ctx,
                all_typevars=self.all_typevars,
            )
            if errors:
                self.show_call_error(
                    "Cannot resolve type variables",
                    ctx,
                    detail=str(CanAssignError(children=list(errors))),
                )
                return self.get_default_return()
            if self._return_key in self.typevars_of_params:
                return_value = return_value.substitute_typevars(typevar_values)

        had_error = False
        used_any = False
        new_args = None
        for name, (position, composite) in bound_args.items():
            param = self.parameters[name]
            (tv_map, param_used_any, remaining_value) = (
                self._check_param_type_compatibility(
                    param,
                    composite,
                    ctx,
                    typevar_values,
                    # If position is None we can't narrow so don't bother.
                    is_overload=is_overload and position is not None,
                )
            )
            if tv_map is None:
                had_error = True
            if param_used_any:
                used_any = True
            if remaining_value is not None:
                if isinstance(position, int):
                    new_positionals = list(preprocessed.positionals)
                    existing_required, _ = new_positionals[position]
                    new_positionals[position] = existing_required, Composite(
                        remaining_value
                    )
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

        composites = {param: composite for param, (_, composite) in bound_args.items()}
        # don't call the implementation function if we had an error, so that
        # the implementation function doesn't have to worry about basic
        # type checking
        if not had_error:
            # Unfortunately we can't make a CallContext out of a _CanAssignBasedContext
            if self.impl is not None and isinstance(ctx, _VisitorBasedContext):
                call_ctx = CallContext(
                    vars=variables,
                    visitor=ctx.visitor,
                    composites=composites,
                    node=ctx.node,
                )
                return_value = self.impl(call_ctx)
            elif self.evaluator is not None:
                varmap = {
                    param: composite.value
                    for param, (_, composite) in bound_args.items()
                }
                positions = {
                    param: position for param, (position, _) in bound_args.items()
                }
                eval_ctx = EvalContext(
                    varmap, positions, ctx.can_assign_ctx, typevar_values
                )
                return_value, errors = self.evaluator.evaluate(eval_ctx)
                for error in errors:
                    error_node = None
                    if error.argument is not None:
                        composite = bound_args[error.argument][1]
                        error_node = composite.node
                    self.show_call_error(
                        error.message, ctx, node=error_node, detail=error.get_detail()
                    )

        if self.allow_call:
            runtime_return = self._maybe_perform_call(preprocessed, ctx)
            if runtime_return is not None:
                if isinstance(return_value, ImplReturn):
                    return_value = ImplReturn(
                        runtime_return,
                        return_value.constraint,
                        return_value.no_return_unless,
                    )
                else:
                    return_value = runtime_return
        ret = self._apply_annotated_constraints(return_value, composites, ctx)
        return CallReturn(
            ret,
            is_error=had_error,
            used_any_for_match=used_any,
            remaining_arguments=new_args,
            sig=self,
        )

    def _maybe_perform_call(
        self, actual_args: ActualArguments, ctx: CheckCallContext
    ) -> Optional[Value]:
        if self.callable is None or not callable(self.callable):
            return None
        args = []
        kwargs = {}
        for definitely_present, composite in actual_args.positionals:
            if not definitely_present:
                return None
            arg = _extract_known_value(composite.value)
            if arg is None:
                return None
            args.append(arg.val)
        if actual_args.star_args is not None:
            values = concrete_values_from_iterable(
                actual_args.star_args, ctx.can_assign_ctx
            )
            if not isinstance(values, collections.abc.Sequence):
                return None
            for args_val in values:
                arg = _extract_known_value(args_val)
                if arg is None:
                    return None
                args.append(arg.val)
        for kwarg, (required, composite) in actual_args.keywords.items():
            if not required:
                return None
            kwarg_value = _extract_known_value(composite.value)
            if kwarg_value is None:
                return None
            kwargs[kwarg] = kwarg_value.val
        if actual_args.star_kwargs is not None:
            value = replace_known_sequence_value(actual_args.star_kwargs)
            if isinstance(value, DictIncompleteValue):
                for pair in value.kv_pairs:
                    if pair.is_many or not pair.is_required:
                        return None
                    key_val = _extract_known_value(pair.key)
                    value_val = _extract_known_value(pair.value)
                    if (
                        key_val is None
                        or value_val is None
                        or not isinstance(key_val.val, str)
                    ):
                        return None
                    kwargs[key_val.val] = value_val.val
            else:
                return None

        try:
            value = self.callable(*args, **kwargs)
        except Exception as e:
            message = f"Error calling {self}: {safe_str(e)}"
            ctx.on_error(message)
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
        # Callable[..., Any] is compatible with an asynq callable too.
        if (
            self.is_asynq
            and not other.is_asynq
            and not any(
                param.kind is ParameterKind.ELLIPSIS
                for param in other.parameters.values()
            )
        ):
            return CanAssignError("callable is not asynq")
        if USE_CHECK_CALL_FOR_CAN_ASSIGN:
            return self.can_assign_through_check_call(other, ctx)
        their_return = other.return_value
        my_return = self.return_value
        return_tv_map = my_return.can_assign(their_return, ctx)
        if isinstance(return_tv_map, CanAssignError):
            return CanAssignError(
                "return annotation is not compatible", [return_tv_map]
            )
        tv_maps = [return_tv_map]
        their_params = list(other.parameters.values())
        their_args = other.get_param_of_kind(ParameterKind.VAR_POSITIONAL)
        if their_args is not None:
            their_args_index = their_params.index(their_args)
            args_annotation = their_args.get_annotation()
        else:
            their_args_index = -1
            args_annotation = None
        their_kwargs = other.get_param_of_kind(ParameterKind.VAR_KEYWORD)
        if their_kwargs is not None:
            kwargs_annotation = their_kwargs.get_annotation()
        else:
            kwargs_annotation = None
        their_ellipsis = other.get_param_of_kind(ParameterKind.ELLIPSIS)
        if their_ellipsis is not None:
            args_annotation = kwargs_annotation = AnyValue(AnySource.ellipsis_callable)
        consumed_positional = set()
        consumed_required_pos_only = set()
        consumed_keyword = set()
        consumed_paramspec = False
        for i, my_param in enumerate(self.parameters.values()):
            my_annotation = my_param.get_annotation()
            if my_param.kind is ParameterKind.POSITIONAL_ONLY:
                if i < len(their_params) and their_params[i].kind in (
                    ParameterKind.POSITIONAL_ONLY,
                    ParameterKind.POSITIONAL_OR_KEYWORD,
                ):
                    if my_param.default is not None and their_params[i].default is None:
                        return CanAssignError(
                            f"positional-only param {my_param.name!r} has no default"
                        )

                    their_annotation = their_params[i].get_annotation()
                    tv_map = their_annotation.can_assign(my_annotation, ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            "type of positional-only parameter"
                            f" {my_param.name!r} is incompatible",
                            [tv_map],
                        )
                    tv_maps.append(tv_map)
                    consumed_positional.add(their_params[i].name)
                    if their_params[i].default is None:
                        consumed_required_pos_only.add(their_params[i].name)
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
            elif my_param.kind is ParameterKind.POSITIONAL_OR_KEYWORD:
                if (
                    i < len(their_params)
                    and their_params[i].kind is ParameterKind.POSITIONAL_OR_KEYWORD
                ):
                    if my_param.name != their_params[i].name:
                        return CanAssignError(
                            f"param name {their_params[i].name!r} does not match"
                            f" {my_param.name!r}"
                        )
                    if my_param.default is not None and their_params[i].default is None:
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
                    and their_params[i].kind is ParameterKind.POSITIONAL_ONLY
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
            elif my_param.kind is ParameterKind.KEYWORD_ONLY:
                their_param = other.parameters.get(my_param.name)
                if their_param is not None and their_param.kind in (
                    ParameterKind.POSITIONAL_OR_KEYWORD,
                    ParameterKind.KEYWORD_ONLY,
                ):
                    if my_param.default is not None and their_param.default is None:
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
            elif my_param.kind is ParameterKind.VAR_POSITIONAL:
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
                        ParameterKind.POSITIONAL_ONLY,
                        ParameterKind.POSITIONAL_OR_KEYWORD,
                    )
                ]
                for extra_param in extra_positional:
                    tv_map = extra_param.get_annotation().can_assign(my_annotation, ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"type of param {extra_param.name!r} is incompatible"
                            " with *args type",
                            [tv_map],
                        )
                    tv_maps.append(tv_map)
            elif my_param.kind is ParameterKind.VAR_KEYWORD:
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
                    in (ParameterKind.KEYWORD_ONLY, ParameterKind.POSITIONAL_OR_KEYWORD)
                    and param.name not in consumed_required_pos_only
                ]
                for extra_param in extra_keyword:
                    tv_map = extra_param.get_annotation().can_assign(my_annotation, ctx)
                    if isinstance(tv_map, CanAssignError):
                        return CanAssignError(
                            f"type of param {extra_param.name!r} is incompatible"
                            " with **kwargs type",
                            [tv_map],
                        )
                    tv_maps.append(tv_map)
            elif my_param.kind is ParameterKind.PARAM_SPEC:
                remaining = [
                    param
                    for param in other.parameters.values()
                    if param.name not in consumed_positional
                    and param.name not in consumed_keyword
                ]
                new_sig = Signature.make(remaining)
                assert isinstance(my_annotation, TypeVarValue)
                tv_maps.append(
                    {
                        my_annotation.typevar: [
                            LowerBound(my_annotation.typevar, CallableValue(new_sig))
                        ]
                    }
                )
                consumed_paramspec = True
            elif my_param.kind is ParameterKind.ELLIPSIS:
                consumed_paramspec = True
            else:
                assert False, f"unhandled param {my_param}"

        if not consumed_paramspec:
            for param in their_params:
                if (
                    param.kind is ParameterKind.VAR_POSITIONAL
                    or param.kind is ParameterKind.VAR_KEYWORD
                ):
                    continue  # ok if they have extra *args or **kwargs
                elif param.default is not None:
                    continue
                elif param.kind is ParameterKind.POSITIONAL_ONLY:
                    if param.name not in consumed_positional:
                        return CanAssignError(
                            f"takes extra positional-only parameter {param.name!r}"
                        )
                elif param.kind is ParameterKind.POSITIONAL_OR_KEYWORD:
                    if (
                        param.name not in consumed_positional
                        and param.name not in consumed_keyword
                    ):
                        return CanAssignError(f"takes extra parameter {param.name!r}")
                elif param.kind is ParameterKind.KEYWORD_ONLY:
                    if param.name not in consumed_keyword:
                        return CanAssignError(f"takes extra parameter {param.name!r}")
                elif param.kind is ParameterKind.PARAM_SPEC:
                    return CanAssignError(f"takes extra ParamSpec {param!r}")
                elif param.kind is ParameterKind.ELLIPSIS:
                    continue
                else:
                    assert False, f"unhandled param {param}"

        return unify_bounds_maps(tv_maps)

    def can_assign_through_check_call(
        self, other: "Signature", ctx: CanAssignContext
    ) -> CanAssign:
        args = [param.to_argument() for param in self.parameters.values()]
        check_ctx = _CanAssignBasedContext(ctx)
        actual_args = preprocess_args(args, check_ctx)
        if actual_args is None:
            return CanAssignError(
                "Invalid callable", [CanAssignError(e) for e in check_ctx.errors]
            )
        return_value = other.check_call_preprocessed(actual_args, check_ctx)
        if check_ctx.errors:
            return CanAssignError(
                "Incompatible callable", [CanAssignError(e) for e in check_ctx.errors]
            )
        return_tv_map = self.return_value.can_assign(return_value.return_value, ctx)
        if isinstance(return_tv_map, CanAssignError):
            return CanAssignError(
                "Return annotation is not compatible", [return_tv_map]
            )
        return return_tv_map

    def get_param_of_kind(self, kind: ParameterKind) -> Optional[SigParameter]:
        for param in self.parameters.values():
            if param.kind is kind:
                return param
        return None

    def substitute_typevars(self, typevars: TypeVarMap) -> "Signature":
        params = []
        for name, param in self.parameters.items():
            if param.kind is ParameterKind.PARAM_SPEC:
                assert isinstance(param.annotation, TypeVarValue)
                tv = param.annotation.typevar
                if tv in typevars:
                    new_val = typevars[tv].substitute_typevars(typevars)
                    if isinstance(new_val, TypeVarValue):
                        assert new_val.is_paramspec, new_val
                        new_param = SigParameter(
                            param.name, param.kind, annotation=new_val
                        )
                        params.append((name, new_param))
                    elif isinstance(new_val, AnyValue):
                        new_param = SigParameter(param.name, ParameterKind.ELLIPSIS)
                        params.append((param.name, new_param))
                    elif isinstance(new_val, CallValue):
                        new_param = SigParameter(
                            param.name, ParameterKind.PARAM_SPEC, annotation=new_val
                        )
                        params.append((param.name, new_param))
                    else:
                        assert isinstance(new_val, CallableValue), new_val
                        assert isinstance(new_val.signature, Signature), new_val
                        params += list(new_val.signature.parameters.items())
                else:
                    params.append((name, param))
            else:
                params.append((name, param.substitute_typevars(typevars)))
        params_dict = dict(params)
        return_value = self.return_value.substitute_typevars(typevars)
        # Returning the same object helps the local return value check, which relies
        # on identity of signature objects.
        if return_value == self.return_value and params_dict == self.parameters:
            return self
        return Signature(
            params_dict,
            return_value,
            impl=self.impl,
            callable=self.callable,
            is_asynq=self.is_asynq,
            has_return_annotation=self.has_return_annotation,
            allow_call=self.allow_call,
            evaluator=self.evaluator,
        )

    def walk_values(self) -> Iterable[Value]:
        yield from self.return_value.walk_values()
        for param in self.parameters.values():
            yield from param.annotation.walk_values()

    def get_asynq_value(self) -> "Signature":
        """Return the :class:`Signature` for the `.asynq` attribute of an
        :class:`pyanalyze.extensions.AsynqCallable`."""
        if not self.is_asynq:
            raise TypeError("get_asynq_value() is only supported for AsynqCallable")
        return_value = AsyncTaskIncompleteValue(asynq.AsyncTask, self.return_value)
        return Signature.make(
            self.parameters.values(),
            return_value,
            impl=self.impl,
            callable=self.callable,
            has_return_annotation=self.has_return_annotation,
            is_asynq=False,
            allow_call=self.allow_call,
            evaluator=self.evaluator,
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
        is_asynq: bool = False,
        allow_call: bool = False,
        evaluator: Optional[Evaluator] = None,
        deprecated: Optional[str] = None,
    ) -> "Signature":
        """Create a :class:`Signature` object.

        This is more convenient to use than the constructor
        because it abstracts away the creation of the underlying
        :class:`inspect.Signature`.

        """
        if return_annotation is None:
            return_annotation = AnyValue(AnySource.unannotated)
            has_return_annotation = False
        param_dict = {}
        i = 0
        for param in parameters:
            if param.kind is ParameterKind.VAR_POSITIONAL and isinstance(
                param.annotation, SequenceValue
            ):
                simple_members = param.annotation.get_member_sequence()
                if simple_members is None:
                    param_dict[param.name] = param
                    i += 1
                else:
                    for member in simple_members:
                        name = f"@{i}"
                        param_dict[name] = SigParameter(
                            name, ParameterKind.POSITIONAL_ONLY, annotation=member
                        )
                        i += 1
            elif param.kind is ParameterKind.VAR_KEYWORD and isinstance(
                param.annotation, TypedDictValue
            ):
                for name, entry in param.annotation.items.items():
                    param_dict[name] = SigParameter(
                        name,
                        ParameterKind.KEYWORD_ONLY,
                        annotation=entry.typ,
                        default=None if entry.required else AnyValue(AnySource.marker),
                    )
                    i += 1
                if param.annotation.extra_keys is not None:
                    name = f"%kwargs{i}"
                    param_dict[name] = SigParameter(
                        name,
                        ParameterKind.VAR_KEYWORD,
                        annotation=GenericValue(
                            dict, [TypedValue(str), param.annotation.extra_keys]
                        ),
                    )
                    i += 1
            else:
                param_dict[param.name] = param
                i += 1
        if deprecated is None and callable is not None:
            deprecated = safe_getattr(callable, "__deprecated__", None)
        return cls(
            param_dict,
            return_value=return_annotation,
            impl=impl,
            callable=callable,
            has_return_annotation=has_return_annotation,
            is_asynq=is_asynq,
            allow_call=allow_call,
            evaluator=evaluator,
            deprecated=deprecated,
        )

    def __str__(self) -> str:
        param_str = ", ".join(self._render_parameters())
        asynq_str = "@asynq " if self.is_asynq else ""
        rendered = f"{asynq_str}({param_str}) -> {self.return_value}"
        if self.impl:
            rendered += " (with impl)"
        if self.evaluator:
            rendered += " (with evaluator)"
        return rendered

    def _render_parameters(self) -> Iterable[str]:
        # Adapted from inspect.Signature's __str__
        render_pos_only_separator = False
        render_kw_only_separator = True
        for param in self.parameters.values():
            formatted = str(param)

            kind = param.kind

            if kind == ParameterKind.POSITIONAL_ONLY:
                render_pos_only_separator = True
            elif render_pos_only_separator:
                yield "/"
                render_pos_only_separator = False

            if kind == ParameterKind.VAR_POSITIONAL:
                render_kw_only_separator = False
            elif kind == ParameterKind.KEYWORD_ONLY and render_kw_only_separator:
                yield "*"
                render_kw_only_separator = False

            yield formatted

        if render_pos_only_separator:
            yield "/"

    def bind_self(
        self,
        *,
        preserve_impl: bool = False,
        self_annotation_value: Optional[Value] = None,
        self_value: Optional[Value] = None,
        ctx: CanAssignContext,
    ) -> Optional["Signature"]:
        params = list(self.parameters.values())
        if not params:
            return None
        kind = params[0].kind
        if kind in (ParameterKind.ELLIPSIS, ParameterKind.VAR_POSITIONAL):
            new_params = params
            self_tuple_annotation = params[0].annotation
            if (
                isinstance(self_tuple_annotation, GenericValue)
                and self_tuple_annotation.typ is tuple
            ):
                self_annotation = self_tuple_annotation.args[0]
            else:
                self_annotation = AnyValue(AnySource.inference)
        elif kind in (
            ParameterKind.POSITIONAL_ONLY,
            ParameterKind.POSITIONAL_OR_KEYWORD,
        ):
            new_params = params[1:]
            self_annotation = params[0].annotation
        else:
            return None
        if self_annotation_value is not None:
            tv_map = get_tv_map(self_annotation, self_annotation_value, ctx)
            if isinstance(tv_map, CanAssignError):
                return None
        else:
            tv_map = {}
        if self_value is not None:
            tv_map = {**tv_map, SelfT: self_value}
        if tv_map:
            new_params = {
                param.name: param.substitute_typevars(tv_map) for param in new_params
            }
            return_value = self.return_value.substitute_typevars(tv_map)
        else:
            new_params = {param.name: param for param in new_params}
            return_value = self.return_value
        return Signature(
            new_params,
            return_value,
            # We don't carry over the implementation function by default, because it
            # may not work when passed different arguments.
            impl=self.impl if preserve_impl else None,
            callable=self.callable,
            is_asynq=self.is_asynq,
            has_return_annotation=self.has_return_value(),
            allow_call=self.allow_call,
        )

    def has_return_value(self) -> bool:
        return self.has_return_annotation or self.evaluator is not None

    def replace_return_value(self, return_value: Value) -> Self:
        return replace(self, return_value=return_value)


ELLIPSIS_PARAM = SigParameter("...", ParameterKind.ELLIPSIS)
ANY_SIGNATURE = Signature.make([ELLIPSIS_PARAM], AnyValue(AnySource.explicit))
""":class:`Signature` that should be compatible with any other
:class:`Signature`."""


def preprocess_args(
    args: Iterable[Argument], ctx: CheckCallContext
) -> Optional[ActualArguments]:
    """Preprocess the argument list. Produces an ActualArguments object."""

    # Step 1: Split up args and kwargs if possible.
    processed_args: List[Argument] = []
    kwargs_requireds = []
    param_spec = None
    param_spec_star_arg = None
    seen_param_spec_kwargs = False
    for arg, label in args:
        if label is ARGS:
            if isinstance(arg.value, ParamSpecArgsValue):
                if param_spec is not None:
                    ctx.on_error(
                        "Only a single ParamSpec.args can be passed", node=arg.node
                    )
                param_spec = TypeVarValue(arg.value.param_spec)
                param_spec_star_arg = arg
                continue
            concrete_values = concrete_values_from_iterable(
                arg.value, ctx.can_assign_ctx
            )
            if isinstance(concrete_values, CanAssignError):
                ctx.on_error(
                    f"{arg.value} is not iterable",
                    detail=str(concrete_values),
                    node=arg.node,
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
            if isinstance(arg.value, ParamSpecKwargsValue):
                if param_spec is None:
                    ctx.on_error(
                        "ParamSpec.kwargs cannot be passed without ParamSpec.args",
                        node=arg.node,
                    )
                elif param_spec.typevar is not arg.value.param_spec:
                    ctx.on_error(
                        "ParamSpec.args and ParamSpec.kwargs must use the same ParamSpec",
                        node=arg.node,
                    )
                elif seen_param_spec_kwargs:
                    ctx.on_error(
                        "Only a single ParamSpec.kwargs can be passed", node=arg.node
                    )
                seen_param_spec_kwargs = True
                continue
            items = {}
            extra_values = []
            if arg.value is NO_RETURN_VALUE:
                new_value = GenericValue(dict, [TypedValue(str), NO_RETURN_VALUE])
                processed_args.append((Composite(new_value), KWARGS))
                continue

            # We union all the kwargs that may be provided by any union member, so that
            # we give an error if
            for subval in flatten_values(arg.value, unwrap_annotated=True):
                result = _preprocess_kwargs_no_mvv(subval, ctx)
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
                    processed_args.append((Composite(value), PossibleArg(key)))
            if extra_values:
                kwargs_requireds.append(not items)
                new_value = GenericValue(
                    dict, [TypedValue(str), unite_values(*extra_values)]
                )
                # don't preserve the varname because we may have mutilated the dict
                new_composite = Composite(new_value)
                processed_args.append((new_composite, KWARGS))
        else:
            processed_args.append((arg, label))
    if param_spec_star_arg is not None and not seen_param_spec_kwargs:
        ctx.on_error(
            "ParamSpec.args cannot be passed without ParamSpec.kwargs",
            node=param_spec_star_arg.node,
        )

    # Step 2: enforce invariants about ARGS and KWARGS placement. We dump
    # any single arguments that come after *args into *args, and we merge all *args.
    # But for keywords, we first get all the arguments with known keys, and after that unite
    # all the **kwargs into a single argument.
    more_processed_args: List[Tuple[bool, Composite]] = []
    more_processed_kwargs: Dict[str, Tuple[bool, Composite]] = {}
    star_args: Optional[Value] = None
    star_kwargs: Optional[Value] = None
    is_ellipsis: bool = False
    pok_indices = set()
    param_spec = None

    for arg, label in processed_args:
        if label is None or (isinstance(label, PossibleArg) and label.name is None):
            is_required = label is None
            # Should never happen because the parser doesn't let you
            if more_processed_kwargs or star_kwargs is not None:
                ctx.on_error("Positional argument follow keyword arguments")
                return None
            if star_args is not None:
                star_args = unite_values(arg.value, star_args)
            else:
                more_processed_args.append((is_required, arg))
        elif label is ARGS:
            # This is legal: f(x=3, *args)
            # But this is not: f(**kwargs, *args)
            if star_kwargs is not None:
                ctx.on_error("*args follows **kwargs")
                return None
            if star_args is not None:
                assert isinstance(arg.value, GenericValue), repr(processed_args)
                star_args = unite_values(arg.value.args[0], star_args)
            else:
                assert isinstance(arg.value, GenericValue), repr(processed_args)
                star_args = arg.value.args[0]
        elif isinstance(label, (str, PossibleArg)):
            is_required = isinstance(label, str)
            if isinstance(label, PossibleArg):
                assert isinstance(label.name, str), label
                label = label.name
            if label in more_processed_kwargs:
                ctx.on_error(f"Multiple values provided for argument '{label}'")
                return None
            more_processed_kwargs[label] = (is_required, arg)
        elif label is KWARGS:
            assert isinstance(arg.value, GenericValue), repr(processed_args)
            new_kwargs = arg.value.args[1]
            if star_kwargs is None:
                star_kwargs = new_kwargs
            else:
                star_kwargs = unite_values(star_kwargs, new_kwargs)
        elif isinstance(label, PosOrKeyword):
            if label.name in more_processed_kwargs:
                ctx.on_error(f"Multiple values provided for argument '{label}'")
                return None
            pok_indices.add(label.name)
            pok_indices.add(len(more_processed_args))
            more_processed_kwargs[label.name] = (label.is_required, arg)
            more_processed_args.append((label.is_required, arg))
        elif isinstance(label, TypeVarValue):
            if param_spec is not None:
                ctx.on_error("Multiple ParamSpecs passed")
                continue
            param_spec = label
        elif label is ELLIPSIS:
            is_ellipsis = True
        else:
            assert False, repr(label)

    return ActualArguments(
        more_processed_args,
        star_args,
        more_processed_kwargs,
        star_kwargs,
        kwargs_required=any(kwargs_requireds),
        ellipsis=is_ellipsis,
        pos_or_keyword_params=pok_indices,
        param_spec=param_spec,
    )


def _preprocess_kwargs_no_mvv(
    value: Value, ctx: CheckCallContext
) -> Optional[Tuple[Dict[str, Tuple[bool, Value]], Optional[Value]]]:
    """Preprocess a Value passed as **kwargs.

    Two possible return types:

    - None if there was a blocking error (the passed in type is not a mapping).
    - A pair of two values:
        - An {argument: (required, Value)} dict if we know the precise arguments (e.g.,
            for a TypedDict).
        - A single Value if the argument is a mapping, but we don't know all the precise keys.
            This is None if all the keys are known. The Value represents the values in the
            mapping (all the keys must be strings).

    """
    value = replace_known_sequence_value(value)
    if isinstance(value, TypedDictValue):
        return {
            key: (entry.required, entry.typ) for key, entry in value.items.items()
        }, None
    elif isinstance(value, DictIncompleteValue):
        return _preprocess_kwargs_kv_pairs(value.kv_pairs, ctx)
    else:
        mapping_tv_map = get_tv_map(MappingValue, value, ctx.can_assign_ctx)
        if isinstance(mapping_tv_map, CanAssignError):
            ctx.on_error(f"{value} is not a mapping", detail=str(mapping_tv_map))
            return None
        key_type = mapping_tv_map.get(K, AnyValue(AnySource.generic_argument))
        value_type = mapping_tv_map.get(V, AnyValue(AnySource.generic_argument))
        return _preprocess_kwargs_kv_pairs(
            [KVPair(key_type, value_type, is_many=True)], ctx
        )


def _preprocess_kwargs_kv_pairs(
    items: Sequence[KVPair], ctx: CheckCallContext
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
                    ctx.on_error(
                        f"Dict passed as **kwargs contains non-string key {key.val!r}"
                    )
                    return None

        possible_keys = []
        has_non_literal = False
        for subkey in flatten_values(pair.key, unwrap_annotated=True):
            if isinstance(subkey, KnownValue):
                if isinstance(subkey.val, str):
                    possible_keys.append(subkey.val)
                else:
                    ctx.on_error(
                        "Dict passed as **kwargs contains non-string key"
                        f" {subkey.val!r}"
                    )
                    return None
            else:
                can_assign = TypedValue(str).can_assign(subkey, ctx.can_assign_ctx)
                if isinstance(can_assign, CanAssignError):
                    ctx.on_error(
                        f"Dict passed as **kwargs contains non-string key {subkey!r}",
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
        self,
        args: Iterable[Argument],
        visitor: "NameCheckVisitor",
        node: Optional[ast.AST],
    ) -> Value:
        """Check a call to an overloaded function.

        The way overloads are handled is not well specified in any PEPs. Useful resources
        include:

        - Michael Lee's `specification
          <https://github.com/python/typing/issues/253#issuecomment-389262904>`_
          of mypy's behavior.
        - Eric Traut's `discussion
          <https://github.com/microsoft/pyright/issues/2521#issuecomment-956823577>`_
          of pyright's behavior.
        - The `documentation
          <https://github.com/microsoft/pyright/blob/main/docs/type-concepts.md#overloads>`_
          for pyright's behavior.

        Our behavior is closer to mypy. The general rule is to pick the first overload that matches
        and return an error otherwise, but there are two twists: ``Any`` and unions.

        Before we do a full check, we first check only whether the argument names and
        numbers match by calling :func:`Signature.bind_arguments` (a trick we picked up
        from pyright). This makes for better error messages.

        If an overload matched only due to ``Any``, we continue looking for more overloads. If there
        are other matching overloads, we return ``Any`` (with
        ``AnySource.multiple_overload_matches``).
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
        ctx = _VisitorBasedContext(visitor, node)
        actual_args = preprocess_args(args, ctx)
        if actual_args is None:
            return AnyValue(AnySource.error)
        # We first bind the arguments for each overload, to get the obvious errors
        # out of the way first.
        errors_per_overload = []
        bound_args_per_overload = []
        for sig in self.signatures:
            with visitor.catch_errors() as caught_errors:
                bound_args = sig.bind_arguments(actual_args, ctx)
            bound_args_per_overload.append(bound_args)
            errors_per_overload.append(caught_errors)

        if not any(bound_args is not None for bound_args in bound_args_per_overload):
            detail = self._make_detail(errors_per_overload, self.signatures)
            visitor.show_error(
                node,
                "Cannot call overloaded function",
                ErrorCode.incompatible_call,
                detail=str(detail),
            )
            return AnyValue(AnySource.error)

        errors_per_overload = []
        any_rets: List[CallReturn] = []
        union_rets: List[CallReturn] = []
        union_and_any_rets: List[CallReturn] = []
        sigs = [
            sig
            for sig, bound_args in zip(self.signatures, bound_args_per_overload)
            if bound_args is not None
        ]
        last = len(sigs) - 1
        for i, sig in enumerate(sigs):
            with visitor.catch_errors() as caught_errors:
                # We can't use check_call_with_bound_args here because we may
                # rebind the arguments.
                ret = sig.check_call_preprocessed(
                    actual_args,
                    ctx,
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
                return self._unite_rets(
                    any_rets,
                    union_and_any_rets,
                    union_rets,
                    ret,
                    visitor=visitor,
                    node=node,
                )

        if any_rets:
            # We don't do this if we have union_rets, because if we got here, we
            # didn't get any clean matches. Therefore, we must have some remaining
            # union members we haven't handled.
            return self._unite_rets(
                any_rets, union_and_any_rets, union_rets, visitor=visitor, node=node
            )

        # None of the signatures matched
        errors = list(itertools.chain.from_iterable(errors_per_overload))
        codes = {error["error_code"] for error in errors}
        if len(codes) == 1:
            (error_code,) = codes
        else:
            error_code = ErrorCode.incompatible_call
        detail = self._make_detail(errors_per_overload, sigs)
        visitor.show_error(
            node, "Cannot call overloaded function", error_code, detail=str(detail)
        )
        return AnyValue(AnySource.error)

    def _unite_rets(
        self,
        any_rets: Sequence[CallReturn],
        union_and_any_rets: Sequence[CallReturn],
        union_rets: Sequence[CallReturn],
        clean_ret: Optional[CallReturn] = None,
        *,
        visitor: "NameCheckVisitor",
        node: Optional[ast.AST],
    ) -> Value:
        if any_rets or union_and_any_rets:
            deduped = {ret.return_value for ret in any_rets}
            if (
                len(deduped) == 1
                and not union_rets
                and not union_and_any_rets
                and clean_ret is None
            ):
                rets = any_rets
            else:
                return AnyValue(AnySource.multiple_overload_matches)
        elif union_rets:
            if clean_ret is not None:
                rets = [*union_rets, clean_ret]
            else:
                rets = union_rets
        else:
            assert clean_ret is not None
            rets = [clean_ret]
        for ret in rets:
            if ret.sig.deprecated is not None:
                visitor.show_error(
                    node,
                    f"Use of deprecated overload {ret.sig}: {ret.sig.deprecated}",
                    ErrorCode.deprecated,
                )
        return unite_values(*[r.return_value for r in rets])

    def _make_detail(
        self,
        errors_per_overload: Sequence[Sequence[Dict[str, Any]]],
        sigs: Sequence[Signature],
    ) -> CanAssignError:
        details = []
        for sig, errors in zip(sigs, errors_per_overload):
            for error in errors:
                inner = CanAssignError(
                    error["e"],
                    [CanAssignError(error["detail"])] if error["detail"] else [],
                )
                details.append(CanAssignError(f"In overload {sig}", [inner]))
        return CanAssignError(children=details)

    def substitute_typevars(self, typevars: TypeVarMap) -> "OverloadedSignature":
        new_sigs = [sig.substitute_typevars(typevars) for sig in self.signatures]
        if all(sig1 is sig2 for sig1, sig2 in zip(self.signatures, new_sigs)):
            return self
        return OverloadedSignature(new_sigs)

    def bind_self(
        self,
        *,
        preserve_impl: bool = False,
        self_value: Optional[Value] = None,
        self_annotation_value: Optional[Value] = None,
        ctx: CanAssignContext,
    ) -> Optional["ConcreteSignature"]:
        bound_sigs = [
            sig.bind_self(
                preserve_impl=preserve_impl,
                self_value=self_value,
                self_annotation_value=self_annotation_value,
                ctx=ctx,
            )
            for sig in self.signatures
        ]
        bound_sigs = [sig for sig in bound_sigs if isinstance(sig, Signature)]
        if len(bound_sigs) == 1:
            return bound_sigs[0]
        elif bound_sigs:
            return OverloadedSignature(bound_sigs)
        return None

    def has_return_value(self) -> bool:
        return all(sig.has_return_value() for sig in self.signatures)

    def replace_return_value(self, return_value: Value) -> "OverloadedSignature":
        return OverloadedSignature(
            [sig.replace_return_value(return_value) for sig in self.signatures]
        )

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
        bounds_maps = []
        for sig in self.signatures:
            can_assign = sig.can_assign(other, ctx)
            if isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"{other} is incompatible with overload {sig}", [can_assign]
                )
            bounds_maps.append(can_assign)
        return unify_bounds_maps(bounds_maps)


ConcreteSignature = Union[Signature, OverloadedSignature]


@dataclass(frozen=True)
class BoundMethodSignature:
    """Signature for a method bound to a particular value."""

    signature: ConcreteSignature
    self_composite: Composite
    return_override: Optional[Value] = None

    def check_call(
        self,
        args: Iterable[Argument],
        visitor: "NameCheckVisitor",
        node: Optional[ast.AST],
    ) -> Value:
        ret = self.signature.check_call(
            [(self.self_composite, None), *args], visitor, node
        )
        if self.return_override is not None and not self.signature.has_return_value():
            if isinstance(ret, AnnotatedValue):
                return annotate_value(self.return_override, ret.metadata)
            return self.return_override
        return ret

    def get_signature(
        self,
        *,
        preserve_impl: bool = False,
        ctx: CanAssignContext,
        self_annotation_value: Optional[Value] = None,
    ) -> Optional[ConcreteSignature]:
        if self_annotation_value is None:
            self_annotation_value = self.self_composite.value
        return self.signature.bind_self(
            preserve_impl=preserve_impl,
            self_value=self.self_composite.value,
            ctx=ctx,
            self_annotation_value=self_annotation_value,
        )

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
            (
                self.return_override.substitute_typevars(typevars)
                if self.return_override is not None
                else None
            ),
        )

    def __str__(self) -> str:
        return f"{self.signature} bound to {self.self_composite.value}"


MaybeSignature = Union[None, Signature, BoundMethodSignature, OverloadedSignature]


def make_bound_method(
    argspec: MaybeSignature,
    self_composite: Composite,
    return_override: Optional[Value] = None,
    *,
    ctx: CanAssignContext,
) -> Optional[BoundMethodSignature]:
    if argspec is None:
        return None
    if isinstance(argspec, (Signature, OverloadedSignature)):
        return BoundMethodSignature(argspec, self_composite, return_override)
    elif isinstance(argspec, BoundMethodSignature):
        if return_override is None:
            return_override = argspec.return_override
        sig = argspec.get_signature(ctx=ctx)
        if sig is None:
            return None
        return BoundMethodSignature(sig, self_composite, return_override)
    else:
        assert_never(argspec)


K = TypeVar("K")
V = TypeVar("V")
MappingValue = GenericValue(collections.abc.Mapping, [TypeVarValue(K), TypeVarValue(V)])


def can_assign_var_positional(
    my_param: SigParameter, args_annotation: Value, idx: int, ctx: CanAssignContext
) -> Union[List[BoundsMap], CanAssignError]:
    my_annotation = my_param.get_annotation()
    if isinstance(args_annotation, SequenceValue):
        members = args_annotation.get_member_sequence()
        if members is not None:
            length = len(members)
            if idx >= length:
                return CanAssignError(
                    f"parameter {my_param.name!r} is not accepted;"
                    f" {args_annotation} only accepts {length} values"
                )
            their_annotation = members[idx]
            can_assign = their_annotation.can_assign(my_annotation, ctx)
            if isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"type of parameter {my_param.name!r} is incompatible:"
                    f" *args[{idx}] type is incompatible",
                    [can_assign],
                )
            return [can_assign]

    iterable_arg = is_iterable(args_annotation, ctx)
    if isinstance(iterable_arg, CanAssignError):
        return CanAssignError(
            f"{args_annotation} is not an iterable type", [iterable_arg]
        )
    bounds_map = iterable_arg.can_assign(my_annotation, ctx)
    if isinstance(bounds_map, CanAssignError):
        return CanAssignError(
            f"type of parameter {my_param.name!r} is incompatible: "
            "*args type is incompatible",
            [bounds_map],
        )
    return [bounds_map]


def can_assign_var_keyword(
    my_param: SigParameter, kwargs_annotation: Value, ctx: CanAssignContext
) -> Union[List[BoundsMap], CanAssignError]:
    my_annotation = my_param.get_annotation()
    bounds_maps = []
    if isinstance(kwargs_annotation, TypedDictValue):
        if my_param.name not in kwargs_annotation.items:
            return CanAssignError(
                f"parameter {my_param.name!r} is not accepted by {kwargs_annotation}"
            )
        their_annotation = kwargs_annotation.items[my_param.name].typ
        can_assign = their_annotation.can_assign(my_annotation, ctx)
        if isinstance(can_assign, CanAssignError):
            return CanAssignError(
                f"type of parameter {my_param.name!r} is incompatible:"
                f" *kwargs[{my_param.name!r}] type is incompatible",
                [can_assign],
            )
        bounds_maps.append(can_assign)
    else:
        mapping_tv_map = get_tv_map(MappingValue, kwargs_annotation, ctx)
        if isinstance(mapping_tv_map, CanAssignError):
            return CanAssignError(
                f"{kwargs_annotation} is not a mapping type", [mapping_tv_map]
            )
        key_arg = mapping_tv_map.get(K, AnyValue(AnySource.generic_argument))
        can_assign = key_arg.can_assign(KnownValue(my_param.name), ctx)
        if isinstance(can_assign, CanAssignError):
            return CanAssignError(
                f"parameter {my_param.name!r} is not accepted by **kwargs type",
                [can_assign],
            )
        bounds_maps.append(can_assign)
        value_arg = mapping_tv_map.get(V, AnyValue(AnySource.generic_argument))
        can_assign = value_arg.can_assign(my_annotation, ctx)
        if isinstance(can_assign, CanAssignError):
            return CanAssignError(
                f"type of parameter {my_param.name!r} is incompatible: **kwargs"
                " type is incompatible",
                [can_assign],
            )
        bounds_maps.append(can_assign)
    return bounds_maps


def decompose_union(
    expected_type: Value, parent_value: Value, ctx: CanAssignContext
) -> Optional[Tuple[BoundsMap, bool, Value]]:
    value = unannotate(parent_value)
    if isinstance(value, MultiValuedValue):
        bounds_maps = []
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
                bounds_maps.append(can_assign)
        if bounds_maps:
            bounds_map = unify_bounds_maps(bounds_maps)
            assert (
                remaining_values
            ), f"all union members matched between {expected_type} and {parent_value}"
            return bounds_map, union_used_any, unite_values(*remaining_values)
    return None


def check_call_preprocessed(
    sig: ConcreteSignature, args: ActualArguments, ctx: CanAssignContext
) -> CanAssign:
    if isinstance(sig, Signature):
        check_ctx = _CanAssignBasedContext(ctx)
        sig.check_call_preprocessed(args, check_ctx)
        if check_ctx.errors:
            return CanAssignError(
                "Incompatible callable", [CanAssignError(e) for e in check_ctx.errors]
            )
        return {}
    else:
        return CanAssignError("Overloads are not supported")


def _extract_known_value(val: Value) -> Optional[KnownValue]:
    if isinstance(val, AnnotatedValue):
        val = val.value
    if isinstance(val, KnownValue):
        return val
    return None
