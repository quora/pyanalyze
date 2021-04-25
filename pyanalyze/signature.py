"""

Wrappers around Signature objects.

"""

from .error_code import ErrorCode
from .stacked_scopes import (
    AndConstraint,
    Composite,
    Constraint,
    ConstraintType,
    NULL_CONSTRAINT,
    AbstractConstraint,
    PredicateProvider,
    Varname,
)
from .value import (
    AnnotatedValue,
    CanAssignContext,
    HasAttrExtension,
    HasAttrGuardExtension,
    KnownValue,
    ParameterTypeGuardExtension,
    SequenceIncompleteValue,
    DictIncompleteValue,
    TypeGuardExtension,
    UNRESOLVED_VALUE,
    Value,
    TypeVarMap,
    extract_typevars,
    stringify_object,
    unify_typevar_maps,
)

import ast
from dataclasses import dataclass, field
from functools import reduce
from types import MethodType, FunctionType
import inspect
import qcore
from typing import (
    Any,
    Iterable,
    NamedTuple,
    Optional,
    ClassVar,
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

# Implementation functions are passed the following arguments:
# - variables: a dictionary with the function's arguments (keys are variable names and values are
#   Value objects)
# - visitor: the test_scope NameCheckVisitor object, which can be used to check context and emit
#   errors
# - node: the AST node corresponding to the call, which needs to be given in order to show errors.
# They return either a single Value object, indicating what the function returns, or a tuple of two
# or three elements:
# - The return value
# - A Constraint indicating things that are true if the function returns a truthy value,
#   or a PredicateProvider
# - A Constraint indicating things that are true unless the function does not return
ImplementationFnReturn = Union[
    Value,
    Tuple[Value, AbstractConstraint],
    Tuple[Value, AbstractConstraint, AbstractConstraint],
]
# Values should be of type Value, but currently *args/**kwargs are entered
# as tuples and dicts. TODO: fix this.
VarsDict = Dict[str, Any]
ImplementationFn = Callable[
    [VarsDict, "NameCheckVisitor", ast.AST], ImplementationFnReturn
]
EMPTY = inspect.Parameter.empty

ARGS = qcore.MarkerObject("*args")
KWARGS = qcore.MarkerObject("**kwargs")
# Representation of a single argument to a call. Second member is
# None for positional args, str for keyword args, ARGS for *args, KWARGS
# for **kwargs.
Argument = Tuple[Composite, Union[None, str, Literal[ARGS], Literal[KWARGS]]]


class ImplReturn(NamedTuple):
    """Return value of impl functions.

    These functions return either a single Value object, indicating what the
    function returns, or an instance of this class:
    - The return value
    - A Constraint indicating things that are true if the function returns a truthy value,
      or a PredicateProvider
    - A Constraint indicating things that are true unless the function does not return
    """

    return_value: Value
    constraint: Union[AbstractConstraint, PredicateProvider] = NULL_CONSTRAINT
    no_return_unless: AbstractConstraint = NULL_CONSTRAINT


@dataclass
class CallContext:
    """The context passed to an impl function."""

    vars: Dict[str, Value]
    visitor: "NameCheckVisitor"
    bound_args: inspect.BoundArguments
    node: ast.AST

    def ast_for_arg(self, arg: str) -> Optional[ast.AST]:
        composite = self.composite_for_arg(arg)
        if composite is not None:
            return composite.node
        return None

    def varname_for_arg(self, arg: str) -> Optional[Varname]:
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
        arg: Optional[str] = None,
        node: Optional[ast.AST] = None,
    ) -> None:
        node = None
        if arg is not None:
            node = self.ast_for_arg(arg)
        if node is None:
            node = self.node
        self.visitor.show_error(node, message, error_code=error_code)


Impl = Callable[[CallContext], Union[Value, ImplReturn]]


def clean_up_implementation_fn_return(
    return_value: ImplementationFnReturn,
) -> ImplReturn:
    if return_value is None:
        return_value = UNRESOLVED_VALUE
    # Implementation functions may return a pair of (value, constraint)
    # or a three-tuple of (value, constraint, NoReturn unless)
    if isinstance(return_value, tuple):
        if len(return_value) == 2:
            return_value, constraint = return_value
            no_return_unless = NULL_CONSTRAINT
        elif len(return_value) == 3:
            return_value, constraint, no_return_unless = return_value
        else:
            assert (
                False
            ), f"implementation must return a 2- or 3-tuple, not {return_value}"
    else:
        constraint = no_return_unless = NULL_CONSTRAINT
    # this indicates a bug in test_scope, so using assert
    assert isinstance(
        return_value, Value
    ), f"implementation did not return a Value: {return_value}"
    return ImplReturn(return_value, constraint, no_return_unless)


class SigParameter(inspect.Parameter):
    """Wrapper around inspect.Parameter that stores annotations as Value objects."""

    __slots__ = ()

    def __init__(
        self,
        name: str,
        kind: inspect._ParameterKind = inspect.Parameter.POSITIONAL_OR_KEYWORD,
        *,
        default: Union[None, Value, Literal[EMPTY]] = None,
        annotation: Optional[Value] = None,
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


@dataclass
class Signature:
    _return_key: ClassVar[str] = "%return"

    signature: inspect.Signature
    # Deprecated in favor of impl
    implementation: Optional[ImplementationFn] = None
    impl: Optional[Impl] = None
    callable: Optional[object] = None
    is_asynq: bool = False
    has_return_annotation: bool = True
    typevars_of_params: Dict[str, List["TypeVar"]] = field(
        init=False, default_factory=dict, repr=False, compare=False
    )
    all_typevars: Set["TypeVar"] = field(
        init=False, default_factory=set, repr=False, compare=False
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
        self.all_typevars = {
            typevar
            for tv_list in self.typevars_of_params.values()
            for typevar in tv_list
        }

    def _check_param_type_compatibility(
        self,
        param: SigParameter,
        var_value: Value,
        visitor: "NameCheckVisitor",
        node: ast.AST,
        typevar_map: TypeVarMap,
    ) -> bool:
        if param.annotation is not EMPTY and not (
            isinstance(param.default, Composite) and var_value is param.default.value
        ):
            if typevar_map:
                param_typ = param.annotation.substitute_typevars(typevar_map)
            else:
                param_typ = param.annotation
            compatible = param_typ.can_assign(var_value, visitor)
            if compatible is None:
                visitor.show_error(
                    node,
                    f"Incompatible argument type for {param.name}: expected {param_typ} but got {var_value}",
                    ErrorCode.incompatible_argument,
                )
                return False
        return True

    def _translate_bound_arg(self, argument: Any) -> Value:
        if argument is EMPTY:
            return UNRESOLVED_VALUE
        elif isinstance(argument, Composite):
            return argument.value
        elif isinstance(argument, tuple):
            return SequenceIncompleteValue(
                tuple, [composite.value for composite in argument]
            )
        elif isinstance(argument, dict):
            return DictIncompleteValue(
                [
                    (KnownValue(key), composite.value)
                    for key, composite in argument.items()
                ]
            )
        else:
            raise TypeError(repr(argument))

    def _apply_annotated_constraints(
        self,
        raw_return: Union[Value, ImplReturn],
        bound_args: inspect.BoundArguments,
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

    def check_call(
        self, args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
    ) -> ImplReturn:
        """Tries to call this object with the given arguments and keyword arguments.

        Raises a TypeError if something goes wrong.

        This is done by calling the function generated by generate_function(), and then examining
        the local variables to validate types and keyword-only arguments.

        """
        call_args = []
        call_kwargs = {}
        for composite, label in args:
            if label is None:
                call_args.append(composite)
            elif isinstance(label, str):
                call_kwargs[label] = composite
            elif label is ARGS or label is KWARGS:
                # TODO handle these:
                # - type check that they are iterables/mappings
                # - if it's a KnownValue or SequenceIncompleteValue, just add to call_args
                # - else do something smart to still typecheck the call
                return ImplReturn(UNRESOLVED_VALUE)
        try:
            bound_args = self.signature.bind(*call_args, **call_kwargs)
        except TypeError as e:
            if self.callable is not None:
                message = f"In call to {stringify_object(self.callable)}: {e}"
            else:
                message = str(e)
            visitor.show_error(node, message, ErrorCode.incompatible_call)
            return ImplReturn(UNRESOLVED_VALUE)
        bound_args.apply_defaults()
        variables = {
            name: self._translate_bound_arg(value)
            for name, value in bound_args.arguments.items()
        }
        return_value = self.signature.return_annotation
        typevar_values: Dict[TypeVar, Value] = {}
        if self.all_typevars:
            for param_name in self.typevars_of_params:
                if param_name == self._return_key:
                    continue
                var_value = variables[param_name]
                param = self.signature.parameters[param_name]
                if param.annotation is EMPTY:
                    continue
                tv_map = param.annotation.can_assign(var_value, visitor)
                if tv_map:
                    # For now, the first assignment wins.
                    for typevar, value in tv_map.items():
                        typevar_values.setdefault(typevar, value)
            for typevar in self.all_typevars:
                typevar_values.setdefault(typevar, UNRESOLVED_VALUE)
            if self._return_key in self.typevars_of_params:
                return_value = return_value.substitute_typevars(typevar_values)

        had_error = False
        for name, var_value in variables.items():
            param = self.signature.parameters[name]
            if not self._check_param_type_compatibility(
                param, var_value, visitor, node, typevar_values
            ):
                had_error = True

        # don't call the implementation function if we had an error, so that
        # the implementation function doesn't have to worry about basic
        # type checking
        if not had_error and self.impl is not None:
            ctx = CallContext(
                vars=variables, visitor=visitor, bound_args=bound_args, node=node
            )
            return_value = self.impl(ctx)
            return self._apply_annotated_constraints(return_value, bound_args)

        if not had_error and self.implementation is not None:
            return_value = self.implementation(variables, visitor, node)
            return_value = clean_up_implementation_fn_return(return_value)
            return self._apply_annotated_constraints(return_value, bound_args)
        elif return_value is EMPTY:
            return ImplReturn(UNRESOLVED_VALUE)
        else:
            return self._apply_annotated_constraints(return_value, bound_args)

    def can_assign(
        self, other: "Signature", ctx: CanAssignContext
    ) -> Union[str, TypeVarMap]:
        """Equivalent of Value.can_assign.

        If the other signature is incompatible with this signature, return a string
        explaining the discrepancy.

        """
        if self.is_asynq and not other.is_asynq:
            return "callable is not asynq"
        their_return = other.signature.return_annotation
        my_return = self.signature.return_annotation
        return_tv_map = my_return.can_assign(their_return, ctx)
        if return_tv_map is None:
            return (
                f"return annotation {their_return} is not compatible with {my_return}"
            )
        tv_maps = [return_tv_map]
        their_params = list(other.signature.parameters.values())
        their_args = other.get_param_of_kind(SigParameter.VAR_POSITIONAL)
        their_kwargs = other.get_param_of_kind(SigParameter.VAR_KEYWORD)
        consumed_positional = set()
        consumed_keyword = set()
        for i, my_param in enumerate(self.signature.parameters.values()):
            if my_param.kind is SigParameter.POSITIONAL_ONLY:
                if i < len(their_params) and their_params[i].kind in (
                    SigParameter.POSITIONAL_ONLY,
                    SigParameter.POSITIONAL_OR_KEYWORD,
                ):
                    tv_map = their_params[i].annotation.can_assign(
                        my_param.annotation, ctx
                    )
                    if tv_map is None:
                        return (
                            f"type of positional-only parameter {my_param.name!r} is incompatible: "
                            f"{their_params[i].annotation} is incompatible with {my_param.annotation}"
                        )
                    tv_maps.append(tv_map)
                    consumed_positional.add(their_params[i].name)
                elif their_args is not None:
                    tv_map = their_args.annotation.can_assign(my_param.annotation, ctx)
                    if tv_map is None:
                        return (
                            f"type of positional-only parameter {my_param.name!r} is incompatible: "
                            f"*args type {their_args.annotation} is incompatible with {my_param.annotation}"
                        )
                    tv_maps.append(tv_map)
                else:
                    return f"positional-only parameter {i} is not accepted"
            elif my_param.kind is SigParameter.POSITIONAL_OR_KEYWORD:
                if (
                    i < len(their_params)
                    and their_params[i].kind is SigParameter.POSITIONAL_OR_KEYWORD
                ):
                    if my_param.name != their_params[i].name:
                        return f"param name {their_params[i].name!r} does not match {my_param.name!r}"
                    tv_map = their_params[i].annotation.can_assign(
                        my_param.annotation, ctx
                    )
                    if tv_map is None:
                        return (
                            f"type of parameter {my_param.name!r} is incompatible: "
                            f"{their_params[i].annotation} is incompatible with {my_param.annotation}"
                        )
                    tv_maps.append(tv_map)
                    consumed_positional.add(their_params[i])
                    consumed_keyword.add(their_params[i])
                elif (
                    i < len(their_params)
                    and their_params[i].kind is SigParameter.POSITIONAL_ONLY
                ):
                    return f"parameter {my_param.name!r} is not accepted as a keyword argument"
                elif their_args is not None and their_kwargs is not None:
                    tv_map = their_args.annotation.can_assign(my_param.annotation, ctx)
                    if tv_map is None:
                        return (
                            f"type of parameter {my_param.name!r} is incompatible: "
                            f"*args type {their_args.annotation} is incompatible with {my_param.annotation}"
                        )
                    tv_maps.append(tv_map)
                    tv_map = their_kwargs.annotation.can_assign(
                        my_param.annotation, ctx
                    )
                    if tv_map is None:
                        return (
                            f"type of parameter {my_param.name!r} is incompatible: "
                            f"**kwargs type {their_kwargs.annotation} is incompatible with {my_param.annotation}"
                        )
                    tv_maps.append(tv_map)
                else:
                    return f"parameter {my_param.name!r} is not accepted"
            elif my_param.kind is SigParameter.KEYWORD_ONLY:
                their_param = other.signature.parameters.get(my_param.name)
                if their_param is not None and their_param.kind in (
                    SigParameter.POSITIONAL_OR_KEYWORD,
                    SigParameter.KEYWORD_ONLY,
                ):
                    tv_map = their_param.annotation.can_assign(my_param.annotation, ctx)
                    if tv_map is None:
                        return (
                            f"type of parameter {my_param.name!r} is incompatible: "
                            f"{their_param.annotation} is incompatible with {my_param.annotation}"
                        )
                    tv_maps.append(tv_map)
                    consumed_keyword.add(their_param.name)
                elif their_kwargs is not None:
                    tv_map = their_kwargs.annotation.can_assign(
                        my_param.annotation, ctx
                    )
                    if tv_map is None:
                        return (
                            f"type of parameter {my_param.name!r} is incompatible: "
                            f"**kwargs type {their_kwargs.annotation} is incompatible with {my_param.annotation}"
                        )
                    tv_maps.append(tv_map)
                else:
                    return f"parameter {my_param.name!r} is not accepted"
            elif my_param.kind is SigParameter.VAR_POSITIONAL:
                if their_args is None:
                    return "*args are not accepted"
                tv_map = their_args.annotation.can_assign(my_param.annotation, ctx)
                if tv_map is None:
                    return (
                        f"type of *args is incompatible: "
                        f"{their_args.annotation} is incompatible with {my_param.annotation}"
                    )
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
                    tv_map = extra_param.annotation.can_assign(my_param.annotation, ctx)
                    if tv_map is None:
                        return (
                            f"type of param {extra_param.name!r} is incompatible with "
                            f"*args type {their_args.annotation}"
                        )
                    tv_maps.append(tv_map)
            elif my_param.kind is SigParameter.VAR_KEYWORD:
                if their_kwargs is None:
                    return "**kwargs are not accepted"
                tv_map = their_kwargs.annotation.can_assign(my_param.annotation, ctx)
                if tv_map is None:
                    return (
                        f"type of **kwargs is incompatible: "
                        f"{their_kwargs.annotation} is incompatible with {my_param.annotation}"
                    )
                tv_maps.append(tv_map)
                extra_keyword = [
                    param
                    for param in their_params
                    if param.name not in consumed_keyword
                    and param.kind
                    in (SigParameter.KEYWORD_ONLY, SigParameter.POSITIONAL_OR_KEYWORD)
                ]
                for extra_param in extra_keyword:
                    tv_map = extra_param.annotation.can_assign(my_param.annotation, ctx)
                    if tv_map is None:
                        return (
                            f"type of param {extra_param.name!r} is incompatible with "
                            f"**kwargs type {their_kwargs.annotation}"
                        )
                    tv_maps.append(tv_map)

        final_tv_map = unify_typevar_maps(tv_maps)
        if final_tv_map is None:
            return "callables are incompatible"
        return final_tv_map

    def get_param_of_kind(self, kind: inspect._ParameterKind) -> Optional[SigParameter]:
        for param in self.signature.parameters.values():
            if param.kind is kind:
                return param
        return None

    @classmethod
    def make(
        cls,
        parameters: Iterable[SigParameter],
        return_annotation: Optional[Value] = None,
        *,
        impl: Optional[Impl] = None,
        implementation: Optional[ImplementationFn] = None,
        callable: Optional[object] = None,
        has_return_annotation: bool = True,
    ) -> "Signature":
        if return_annotation is None:
            return_annotation = UNRESOLVED_VALUE
            has_return_annotation = False
        return cls(
            signature=inspect.Signature(
                parameters, return_annotation=return_annotation
            ),
            implementation=implementation,
            impl=impl,
            callable=callable,
            has_return_annotation=has_return_annotation,
        )

    # TODO: do we need these?
    def has_return_value(self) -> bool:
        return self.has_return_annotation

    @property
    def return_value(self) -> Value:
        return self.signature.return_annotation


@dataclass
class BoundMethodSignature:
    signature: Signature
    self_value: Value

    def check_call(
        self, args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
    ) -> Tuple[Value, AbstractConstraint, AbstractConstraint]:
        return self.signature.check_call(
            # TODO get a composite
            [(Composite(self.self_value, None, None), None), *args],
            visitor,
            node,
        )

    def has_return_value(self) -> bool:
        return self.signature.has_return_value()

    @property
    def return_value(self) -> Value:
        return self.signature.return_value


@dataclass
class PropertyArgSpec:
    """Pseudo-argspec for properties."""

    obj: object
    return_value: Value = UNRESOLVED_VALUE

    def check_call(
        self, args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
    ) -> Tuple[Value, AbstractConstraint, AbstractConstraint]:
        raise TypeError("property object is not callable")

    def has_return_value(self) -> bool:
        return self.return_value is not UNRESOLVED_VALUE


MaybeSignature = Union[None, Signature, BoundMethodSignature, PropertyArgSpec]


def make_bound_method(
    argspec: MaybeSignature, self_value: Value
) -> Optional[BoundMethodSignature]:
    if argspec is None:
        return None
    if isinstance(argspec, Signature):
        return BoundMethodSignature(argspec, self_value)
    elif isinstance(argspec, BoundMethodSignature):
        return BoundMethodSignature(argspec.signature, self_value)
    else:
        assert False, f"invalid argspec {argspec}"
