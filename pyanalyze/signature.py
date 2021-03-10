"""

Wrappers around Signature objects.

"""

from .error_code import ErrorCode
from .stacked_scopes import NULL_CONSTRAINT, AbstractConstraint
from .value import (
    TypedValue,
    KnownValue,
    UNRESOLVED_VALUE,
    Value,
    TypeVarMap,
    extract_typevars,
)

import ast
import builtins
from dataclasses import dataclass, field, InitVar
import qcore
import logging
import re
from typing import (
    Any,
    Sequence,
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
# - A Constraint indicating things that are true if the function returns a truthy value
# - A Constraint indicating things that are true unless the function does not return
ImplementationFnReturn = Union[
    Value,
    Tuple[Value, AbstractConstraint],
    Tuple[Value, AbstractConstraint, AbstractConstraint],
]
VarsDict = Dict[str, Value]
ImplementationFn = Callable[
    [VarsDict, "NameCheckVisitor", ast.AST], ImplementationFnReturn
]

NON_IDENTIFIER_CHARS = re.compile(r"[^a-zA-Z_\d]")


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


@dataclass
class Parameter:
    """Class representing a function parameter.

    default_value is Parameter.no_default_value when there is no default value.

    """

    no_default_value: ClassVar[object] = object()

    name: str
    default_value: object = no_default_value
    typ: Optional[Value] = None

    def __post_init__(self) -> None:
        assert self.typ is None or isinstance(self.typ, Value), repr(self)


@dataclass
class BoundMethodArgSpecWrapper:
    """Wrapper around ExtendedArgSpec to support bound methods.

    Adds the object that the method is bound to as an argument.

    """

    argspec: "ExtendedArgSpec" = field(init=False)
    argspec_arg: InitVar[Union["ExtendedArgSpec", "BoundMethodArgSpecWrapper"]]
    self_value: Value

    def __post_init__(
        self, argspec_arg: Union["ExtendedArgSpec", "BoundMethodArgSpecWrapper"]
    ) -> None:
        if isinstance(argspec_arg, BoundMethodArgSpecWrapper):
            argspec_arg = argspec_arg.argspec
        assert isinstance(
            argspec_arg, ExtendedArgSpec
        ), f"invalid argspec {argspec_arg!r}"
        self.argspec = argspec_arg

    def check_call(
        self,
        args: Iterable[Value],
        keywords: Iterable[Tuple[str, Value]],
        visitor: "NameCheckVisitor",
        node: ast.AST,
    ) -> Tuple[Value, AbstractConstraint, AbstractConstraint]:
        return self.argspec.check_call(
            [self.self_value, *args], keywords, visitor, node
        )

    def has_return_value(self) -> bool:
        return self.argspec.has_return_value()

    @property
    def return_value(self) -> Value:
        return self.argspec.return_value


@dataclass
class PropertyArgSpec:
    """Pseudo-argspec for properties."""

    obj: object
    return_value: Value = UNRESOLVED_VALUE

    def check_call(
        self,
        args: Iterable[Value],
        keywords: Iterable[Tuple[str, Value]],
        visitor: "NameCheckVisitor",
        node: ast.AST,
    ) -> Tuple[Value, AbstractConstraint, AbstractConstraint]:
        raise TypeError("property object is not callable")

    def has_return_value(self) -> bool:
        return self.return_value is not UNRESOLVED_VALUE


Logger = Callable[[int, str, object], object]


@dataclass
class ExtendedArgSpec:
    """A richer version of the standard inspect.ArgSpec object.

    This stores:
    - arguments (list of Parameters): all standard arguments taken by the object.
    - starargs (string or None): name of the *args argument, if present.
    - kwargs (string or None): name of the **kwargs argument, if present.
    - kwonly_args (list of Parameters): list of keyword-only arguments taken by the object. Python 2
      itself does not support this, but some of our objects simulate keyword-only arguments by
      working with **kwargs.
    - return_value (Value): what the object returns
    - name (string): name of the object. This is used only for error output.
    - implementation (function): a function that implements the object. This gets passed a
      dictionary corresponding to the function's scope, a NameCheckVisitor object, and the node
      where the function was called. This can be used for more complicated argument checking and
      for computing the return value from the arguments.

    """

    _default_value: ClassVar[object] = object()
    _kwonly_args_name: ClassVar[str] = "__kwargs"
    _return_key: ClassVar[str] = "%return"
    _excluded_attributes = {"logger"}

    arguments: Sequence[Parameter]
    starargs: Optional[str] = None
    kwargs: Optional[str] = None
    kwonly_args: Sequence[Parameter] = field(default_factory=list)
    return_value: Value = UNRESOLVED_VALUE
    name: Optional[str] = None
    implementation: Optional[ImplementationFn] = None
    logger: Optional[Logger] = field(repr=False, default=None, compare=False)
    params_of_names: Dict[str, Parameter] = field(init=False, repr=False)
    _has_return_value: bool = field(init=False, repr=False)
    typevars_of_params: Dict[str, List["TypeVar"]] = field(
        init=False, default_factory=dict, repr=False
    )
    all_typevars: Set["TypeVar"] = field(init=False, default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._has_return_value = self.return_value is not UNRESOLVED_VALUE
        self.params_of_names = {}
        for param in self.arguments:
            self.params_of_names[param.name] = param
        for param in self.kwonly_args:
            self.params_of_names[param.name] = param
        if self.starargs is not None:
            self.params_of_names[self.starargs] = Parameter(
                self.starargs, typ=TypedValue(tuple)
            )
        if self.kwargs is not None:
            self.params_of_names[self.kwargs] = Parameter(
                self.kwargs, typ=TypedValue(dict)
            )
        for param_name, param in self.params_of_names.items():
            if param.typ is None:
                continue
            typevars = list(extract_typevars(param.typ))
            if typevars:
                self.typevars_of_params[param_name] = typevars
        return_typevars = list(extract_typevars(self.return_value))
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

    @qcore.caching.cached_per_instance()
    def generate_function(self) -> Callable[..., Any]:
        """Generates a function with this argspec.

        This is done by exec-ing code that corresponds to this argspec. The function will return
        its locals(). Keyword-only arguments are not checked.

        """
        argument_strings = []
        scope = {
            "_default_value": self._default_value,
            # so that we can call locals() from inside the function even if a function argument is
            # called locals
            # if you want to call a function argument __builtin__, go do it somewhere else please
            "__builtin__": builtins,
        }

        def add_arg(arg):
            if arg.default_value is Parameter.no_default_value:
                argument_strings.append(arg.name)
            else:
                default_obj_name = "__default_" + arg.name
                scope[default_obj_name] = KnownValue(arg.default_value)
                argument_strings.append("%s=%s" % (arg.name, default_obj_name))

        for arg in self.arguments:
            add_arg(arg)

        if self.starargs is not None:
            argument_strings.append("*%s" % self.starargs)

        if self.kwonly_args:
            if self.starargs is None:
                argument_strings.append("*")
            for arg in self.kwonly_args:
                add_arg(arg)
        if self.kwargs is not None:
            argument_strings.append("**%s" % self.kwargs)

        if self.name is None:
            name = "test_function"
        else:
            name = str(self.name)
        # for lambdas name is "<lambda>"
        name = NON_IDENTIFIER_CHARS.sub("_", name)

        code_str = """
def %(name)s(%(arguments)s):
    return __builtin__.locals()
""" % {
            "arguments": ", ".join(argument_strings),
            "name": name,
        }

        self.log(logging.DEBUG, "Code to execute", code_str)
        exec (code_str, scope)
        return scope[name]

    def _check_param_type_compatibility(
        self,
        param: Parameter,
        var_value: Value,
        visitor: "NameCheckVisitor",
        node: ast.AST,
        typevar_map: TypeVarMap,
    ) -> None:
        if param.typ is not None and var_value != KnownValue(param.default_value):
            if typevar_map:
                param_typ = param.typ.substitute_typevars(typevar_map)
            else:
                param_typ = param.typ
            compatible = param_typ.can_assign(var_value, visitor)
            if compatible is None:
                visitor.show_error(
                    node,
                    "Incompatible argument type for %s: expected %s but got %s"
                    % (param.name, param_typ, var_value),
                    ErrorCode.incompatible_argument,
                )

    def check_call(
        self,
        args: Iterable[Value],
        keywords: Iterable[Tuple[str, Value]],
        visitor: "NameCheckVisitor",
        node: ast.AST,
    ) -> Tuple[Value, AbstractConstraint, AbstractConstraint]:
        """Tries to call this object with the given arguments and keyword arguments.

        Raises a TypeError if something goes wrong.

        This is done by calling the function generated by generate_function(), and then examining
        the local variables to validate types and keyword-only arguments.

        """
        fn = self.generate_function()
        try:
            variables = fn(*args, **dict(keywords))
        except TypeError as e:
            visitor.show_error(node, repr(e), ErrorCode.incompatible_call)
            return UNRESOLVED_VALUE, NULL_CONSTRAINT, NULL_CONSTRAINT
        return_value = self.return_value
        typevar_values: Dict[TypeVar, Value] = {}
        if self.all_typevars:
            for param_name in self.typevars_of_params:
                if param_name == self._return_key:
                    continue
                var_value = variables[param_name]
                if var_value is self._default_value:
                    continue
                param = self.params_of_names[param_name]
                if param.typ is None:
                    continue
                tv_map = param.typ.can_assign(var_value, visitor)
                if tv_map:
                    # For now, the first assignment wins.
                    for typevar, value in tv_map.items():
                        typevar_values.setdefault(typevar, value)
            for typevar in self.all_typevars:
                typevar_values.setdefault(typevar, UNRESOLVED_VALUE)
            if self._return_key in self.typevars_of_params:
                return_value = return_value.substitute_typevars(typevar_values)

        non_param_names = {self.starargs, self.kwargs, self._kwonly_args_name}
        for name, var_value in variables.items():
            if var_value is not self._default_value and name not in non_param_names:
                param = self.params_of_names[name]
                self._check_param_type_compatibility(
                    param, var_value, visitor, node, typevar_values
                )

        if self.implementation is not None:
            return_value = self.implementation(variables, visitor, node)
            return clean_up_implementation_fn_return(return_value)
        else:
            return return_value, NULL_CONSTRAINT, NULL_CONSTRAINT

    def has_return_value(self) -> bool:
        # We can't check self.return_value directly here because that may have
        # been wrapped in an Awaitable.
        return self._has_return_value


MaybeArgspec = Union[None, ExtendedArgSpec, PropertyArgSpec, BoundMethodArgSpecWrapper]
