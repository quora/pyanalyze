"""

Wrappers around Signature objects.

"""

from functools import reduce
from .error_code import ErrorCode
from .stacked_scopes import (
    AndConstraint,
    Composite,
    Constraint,
    ConstraintType,
    NULL_CONSTRAINT,
    AbstractConstraint,
)
from .value import (
    AnnotatedValue,
    KnownValue,
    ParameterTypeGuardExtension,
    SequenceIncompleteValue,
    DictIncompleteValue,
    UNRESOLVED_VALUE,
    Value,
    TypeVarMap,
    extract_typevars,
    stringify_object,
)

import ast
from dataclasses import dataclass, field
import inspect
import qcore
import logging
from typing import (
    Any,
    Iterable,
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
Logger = Callable[[int, str, object], object]
EMPTY = inspect.Parameter.empty

ARGS = qcore.MarkerObject("*args")
KWARGS = qcore.MarkerObject("**kwargs")
# Representation of a single argument to a call. Second member is
# None for positional args, str for keyword args, ARGS for *args, KWARGS
# for **kwargs.
Argument = Tuple[Composite, Union[None, str, Literal[ARGS], Literal[KWARGS]]]


def clean_up_implementation_fn_return(
    return_value: ImplementationFnReturn,
) -> Tuple[Value, AbstractConstraint, AbstractConstraint]:
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
    return return_value, constraint, no_return_unless


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
            default_composite = Composite(default, None)
        else:
            default_composite = default
        if annotation is None:
            annotation = EMPTY
        super().__init__(name, kind, default=default_composite, annotation=annotation)


@dataclass
class Signature:
    _return_key: ClassVar[str] = "%return"

    signature: inspect.Signature
    implementation: Optional[ImplementationFn] = None
    callable: Optional[object] = None
    has_return_annotation: bool = True
    logger: Optional[Logger] = field(repr=False, default=None, compare=False)
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

    def log(self, level: int, label: str, value: object) -> None:
        if self.logger is not None:
            self.logger(level, label, value)

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
        self, return_value: Value, bound_args: inspect.BoundArguments
    ) -> Tuple[Value, AbstractConstraint, AbstractConstraint]:
        constraints = []
        if isinstance(return_value, AnnotatedValue):
            for guard in return_value.get_metadata_of_type(ParameterTypeGuardExtension):
                print("GUARD", guard)
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
        if constraints:
            constraint = reduce(AndConstraint, constraints)
        else:
            constraint = NULL_CONSTRAINT
        return return_value, constraint, NULL_CONSTRAINT

    def check_call(
        self, args: Iterable[Argument], visitor: "NameCheckVisitor", node: ast.AST
    ) -> Tuple[Value, AbstractConstraint, AbstractConstraint]:
        """Tries to call this object with the given arguments and keyword arguments.

        Raises a TypeError if something goes wrong.

        This is done by calling the function generated by generate_function(), and then examining
        the local variables to validate types and keyword-only arguments.

        """
        print("SIG", self)
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
                self.log(logging.DEBUG, "Ignoring call with *args/**kwargs", composite)
                return UNRESOLVED_VALUE, NULL_CONSTRAINT, NULL_CONSTRAINT
        try:
            bound_args = self.signature.bind(*call_args, **call_kwargs)
        except TypeError as e:
            if self.callable is not None:
                message = f"In call to {stringify_object(self.callable)}: {e}"
            else:
                message = str(e)
            visitor.show_error(node, message, ErrorCode.incompatible_call)
            return UNRESOLVED_VALUE, NULL_CONSTRAINT, NULL_CONSTRAINT
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
        if not had_error and self.implementation is not None:
            return_value = self.implementation(variables, visitor, node)
            return clean_up_implementation_fn_return(return_value)
        elif return_value is EMPTY:
            return UNRESOLVED_VALUE, NULL_CONSTRAINT, NULL_CONSTRAINT
        else:
            return self._apply_annotated_constraints(return_value, bound_args)

    @classmethod
    def make(
        cls,
        parameters: Iterable[SigParameter],
        return_annotation: Optional[Value] = None,
        *,
        implementation: Optional[ImplementationFn] = None,
        callable: Optional[object] = None,
        logger: Optional[Logger] = None,
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
            [(Composite(self.self_value, None), None), *args],
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
