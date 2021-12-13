"""

Code for understanding type annotations.

This file contains functions that turn various representations of
Python type annotations into :class:`pyanalyze.value.Value` objects.

There are three major functions:

- :func:`type_from_runtime` takes a runtime Python object, for example
  ``type_from_value(int)`` -> ``TypedValue(int)``.
- :func:`type_from_value` takes an existing :class:`pyanalyze.value.Value`
  object. For example, evaluating the expression ``int`` will produce
  ``KnownValue(int)``, and calling :func:`type_from_value` on that value
  will produce ``TypedValue(int)``.
- :func:`type_from_ast` takes an AST node and evaluates it into a type.

These functions all rely on each other. For example, when a forward
reference is found in a runtime annotation, the code parses it and calls
:func:`type_from_ast` to evaluate it.

These functions all use :class:`Context` objects to resolve names and
show errors.

"""
from dataclasses import dataclass, InitVar, field
import typing
import typing_inspect
import qcore
import ast
from typed_ast import ast3
import builtins
from collections.abc import Callable, Iterable
from typed_ast import ast3
from typing import (
    Any,
    Container,
    NamedTuple,
    cast,
    TypeVar,
    ContextManager,
    Mapping,
    NewType,
    Sequence,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from .error_code import ErrorCode
from .extensions import (
    AsynqCallable,
    CustomCheck,
    ExternalType,
    HasAttrGuard,
    ParameterTypeGuard,
    TypeGuard,
)
from .find_unused import used
from .signature import SigParameter, Signature
from .safe import is_typing_name, is_instance_of_typing_name
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CustomCheckExtension,
    Extension,
    HasAttrGuardExtension,
    KnownValue,
    MultiValuedValue,
    NO_RETURN_VALUE,
    ParameterTypeGuardExtension,
    TypeGuardExtension,
    TypedValue,
    SequenceIncompleteValue,
    annotate_value,
    unite_values,
    Value,
    GenericValue,
    SubclassValue,
    TypedDictValue,
    NewTypeValue,
    TypeVarValue,
)

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor

try:
    from typing import get_origin, get_args  # Python 3.9
    from types import GenericAlias
except ImportError:
    GenericAlias = None

    def get_origin(obj: object) -> Any:
        return None

    def get_args(obj: object) -> Tuple[Any, ...]:
        return ()


@dataclass
class Context:
    """A context for evaluating annotations.

    The base implementation does very little. Subclass this to do something more useful.

    """

    should_suppress_undefined_names: bool = field(default=False, init=False)
    """While this is True, no errors are shown for undefined names."""

    def suppress_undefined_names(self) -> ContextManager[None]:
        """Temporarily suppress errors about undefined names."""
        return qcore.override(self, "should_suppress_undefined_names", True)

    def show_error(
        self, message: str, error_code: ErrorCode = ErrorCode.invalid_annotation
    ) -> None:
        """Show an error found while evaluating an annotation."""
        pass

    def get_name(self, node: ast.Name) -> Value:
        """Return the :class:`Value <pyanalyze.value.Value>` corresponding to a name."""
        return AnyValue(AnySource.inference)

    def handle_undefined_name(self, name: str) -> Value:
        if self.should_suppress_undefined_names:
            return AnyValue(AnySource.inference)
        self.show_error(
            f"Undefined name {name!r} used in annotation", ErrorCode.undefined_name
        )
        return AnyValue(AnySource.error)

    def get_name_from_globals(self, name: str, globals: Mapping[str, Any]) -> Value:
        if name in globals:
            return KnownValue(globals[name])
        elif hasattr(builtins, name):
            return KnownValue(getattr(builtins, name))
        return self.handle_undefined_name(name)


@used  # part of an API
def type_from_ast(
    ast_node: ast.AST,
    visitor: Optional["NameCheckVisitor"] = None,
    ctx: Optional[Context] = None,
) -> Value:
    """Given an AST node representing an annotation, return a
    :class:`Value <pyanalyze.value.Value>`.

    :param ast_node: AST node to evaluate.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    """
    if ctx is None:
        ctx = _DefaultContext(visitor, ast_node)
    return _type_from_ast(ast_node, ctx)


def type_from_runtime(
    val: object,
    visitor: Optional["NameCheckVisitor"] = None,
    node: Optional[ast.AST] = None,
    globals: Optional[Mapping[str, object]] = None,
    ctx: Optional[Context] = None,
) -> Value:
    """Given a runtime annotation object, return a
    :class:`Value <pyanalyze.value.Value>`.

    :param val: Object to evaluate. This will usually come from an
                ``__annotations__`` dictionary.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param node: AST node that the annotation derives from. This is
                 used for showing errors. Ignored if `ctx` is given.

    :param globals: Dictionary of global variables that can be used
                    to resolve names. Ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    """

    if ctx is None:
        ctx = _DefaultContext(visitor, node, globals)
    return _type_from_runtime(val, ctx)


def type_from_value(
    value: Value,
    visitor: Optional["NameCheckVisitor"] = None,
    node: Optional[ast.AST] = None,
    ctx: Optional[Context] = None,
    is_typeddict: bool = False,
) -> Value:
    """Given a :class:`Value <pyanalyze.value.Value` representing an annotation,
    return a :class:`Value <pyanalyze.value.Value>` representing the type.

    The input value represents an expression, the output value represents
    a type. For example, the :term:`impl` of ``typing.cast(typ, val)``
    calls :func:`type_from_value` on the value it receives for its
    `typ` argument and returns the result.

    :param value: :class:`Value <pyanalyze.value.Value` to evaluate.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param node: AST node that the annotation derives from. This is
                 used for showing errors. Ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    :param is_typeddict: Whether we are at the top level of a `TypedDict`
                         definition.

    """
    if ctx is None:
        ctx = _DefaultContext(visitor, node)
    return _type_from_value(value, ctx, is_typeddict=is_typeddict)


def value_from_ast(ast_node: ast.AST, ctx: Context) -> Value:
    val = _Visitor(ctx).visit(ast_node)
    if val is None:
        ctx.show_error("Invalid type annotation")
        return AnyValue(AnySource.error)
    return val


def _type_from_ast(node: ast.AST, ctx: Context, is_typeddict: bool = False) -> Value:
    val = value_from_ast(node, ctx)
    return _type_from_value(val, ctx, is_typeddict=is_typeddict)


def _type_from_runtime(val: Any, ctx: Context, is_typeddict: bool = False) -> Value:
    if isinstance(val, str):
        return _eval_forward_ref(val, ctx, is_typeddict=is_typeddict)
    elif isinstance(val, tuple):
        # This happens under some Python versions for types
        # nested in tuples, e.g. on 3.6:
        # > typing_inspect.get_args(Union[Set[int], List[str]])
        # ((typing.Set, int), (typing.List, str))
        if not val:
            # from Tuple[()]
            return KnownValue(())
        origin = val[0]
        if len(val) == 2:
            args = (val[1],)
        else:
            args = val[1:]
        return _value_of_origin_args(origin, args, val, ctx)
    elif GenericAlias is not None and isinstance(val, GenericAlias):
        origin = get_origin(val)
        args = get_args(val)
        if origin is tuple and not args:
            return SequenceIncompleteValue(tuple, [])
        return _value_of_origin_args(origin, args, val, ctx)
    elif typing_inspect.is_literal_type(val):
        args = typing_inspect.get_args(val)
        if len(args) == 0:
            return KnownValue(args[0])
        else:
            return unite_values(*[KnownValue(arg) for arg in args])
    elif typing_inspect.is_union_type(val):
        args = typing_inspect.get_args(val)
        return unite_values(*[_type_from_runtime(arg, ctx) for arg in args])
    elif typing_inspect.is_tuple_type(val):
        args = typing_inspect.get_args(val)
        if not args:
            return TypedValue(tuple)
        elif len(args) == 2 and args[1] is Ellipsis:
            return GenericValue(tuple, [_type_from_runtime(args[0], ctx)])
        elif len(args) == 1 and args[0] == ():
            return SequenceIncompleteValue(tuple, [])  # empty tuple
        else:
            args_vals = [_type_from_runtime(arg, ctx) for arg in args]
            return SequenceIncompleteValue(tuple, args_vals)
    elif is_instance_of_typing_name(val, "_TypedDictMeta"):
        required_keys = getattr(val, "__required_keys__", None)
        # 3.8's typing.TypedDict doesn't have __required_keys__. With
        # inheritance, this makes it apparently impossible to figure out which
        # keys are required at runtime.
        total = getattr(val, "__total__", True)
        return TypedDictValue(
            {
                key: _get_typeddict_value(value, ctx, key, required_keys, total)
                for key, value in val.__annotations__.items()
            }
        )
    elif val is InitVar:
        # On 3.6 and 3.7, InitVar[T] just returns InitVar at runtime, so we can't
        # get the actual type out.
        return AnyValue(AnySource.inference)
    elif isinstance(val, InitVar):
        # val.type exists only on 3.8+, but on earlier versions
        # InitVar instances aren't being created
        # static analysis: ignore[undefined_attribute]
        return type_from_runtime(val.type)
    elif is_instance_of_typing_name(val, "AnnotatedMeta"):
        # Annotated in 3.6's typing_extensions
        origin, metadata = val.__args__
        return _make_annotated(
            _type_from_runtime(origin, ctx), [KnownValue(v) for v in metadata], ctx
        )
    elif is_instance_of_typing_name(val, "_AnnotatedAlias"):
        # Annotated in typing and newer typing_extensions
        return _make_annotated(
            _type_from_runtime(val.__origin__, ctx),
            [KnownValue(v) for v in val.__metadata__],
            ctx,
        )
    elif typing_inspect.is_generic_type(val):
        origin = typing_inspect.get_origin(val)
        args = typing_inspect.get_args(val)
        if getattr(val, "_special", False):
            args = []  # distinguish List from List[T] on 3.7 and 3.8
        return _value_of_origin_args(origin, args, val, ctx, is_typeddict=is_typeddict)
    elif typing_inspect.is_callable_type(val):
        args = typing_inspect.get_args(val)
        return _value_of_origin_args(Callable, args, val, ctx)
    elif val is AsynqCallable:
        return CallableValue(Signature.make([], is_ellipsis_args=True, is_asynq=True))
    elif isinstance(val, type):
        return _maybe_typed_value(val)
    elif val is None:
        return KnownValue(None)
    elif is_typing_name(val, "NoReturn"):
        return NO_RETURN_VALUE
    elif val is typing.Any:
        return AnyValue(AnySource.explicit)
    elif hasattr(val, "__supertype__"):
        if isinstance(val.__supertype__, type):
            # NewType
            return NewTypeValue(val)
        elif typing_inspect.is_tuple_type(val.__supertype__):
            # TODO figure out how to make NewTypes over tuples work
            return AnyValue(AnySource.inference)
        else:
            ctx.show_error(f"Invalid NewType {val}")
            return AnyValue(AnySource.error)
    elif typing_inspect.is_typevar(val):
        tv = cast(TypeVar, val)
        if tv.__bound__ is not None:
            bound = _type_from_runtime(tv.__bound__, ctx)
        else:
            bound = None
        if tv.__constraints__:
            constraints = tuple(
                _type_from_runtime(constraint, ctx) for constraint in tv.__constraints__
            )
        else:
            constraints = ()
        return TypeVarValue(tv, bound=bound, constraints=constraints)
    elif is_typing_name(val, "Final") or is_typing_name(val, "ClassVar"):
        return AnyValue(AnySource.incomplete_annotation)
    elif typing_inspect.is_classvar(val) or typing_inspect.is_final_type(val):
        if hasattr(val, "__type__"):
            # 3.6
            typ = val.__type__
        else:
            # 3.7+
            typ = val.__args__[0]
        return _type_from_runtime(typ, ctx)
    elif is_instance_of_typing_name(val, "_ForwardRef") or is_instance_of_typing_name(
        val, "ForwardRef"
    ):
        # This has issues because the forward ref may be defined in a different file, in
        # which case we don't know which names are valid in it.
        with ctx.suppress_undefined_names():
            try:
                code = ast.parse(val.__forward_arg__)
            except SyntaxError:
                ctx.show_error(
                    f"Syntax error in forward reference: {val.__forward_arg__}"
                )
                return AnyValue(AnySource.error)
            return _type_from_ast(code.body[0], ctx, is_typeddict=is_typeddict)
    elif val is Ellipsis:
        # valid in Callable[..., ]
        return AnyValue(AnySource.explicit)
    elif is_instance_of_typing_name(val, "_TypeAlias"):
        # typing.Pattern and Match, which are not normal generic types for some reason
        return GenericValue(val.impl_type, [_type_from_runtime(val.type_var, ctx)])
    elif isinstance(val, TypeGuard):
        return AnnotatedValue(
            TypedValue(bool),
            [TypeGuardExtension(_type_from_runtime(val.guarded_type, ctx))],
        )
    elif is_instance_of_typing_name(val, "_TypeGuard"):
        # 3.6 only
        return AnnotatedValue(
            TypedValue(bool),
            [TypeGuardExtension(_type_from_runtime(val.__type__, ctx))],
        )
    elif isinstance(val, AsynqCallable):
        arg_types = val.args
        return_type = val.return_type
        if arg_types is Ellipsis:
            return CallableValue(
                Signature.make(
                    [],
                    _type_from_runtime(return_type, ctx),
                    is_ellipsis_args=True,
                    is_asynq=True,
                )
            )
        if not isinstance(arg_types, tuple):
            ctx.show_error("Invalid arguments to AsynqCallable")
            return AnyValue(AnySource.error)
        params = [
            SigParameter(
                f"__arg{i}",
                kind=SigParameter.POSITIONAL_ONLY,
                annotation=_type_from_runtime(arg, ctx),
            )
            for i, arg in enumerate(arg_types)
        ]
        sig = Signature.make(
            params, _type_from_runtime(return_type, ctx), is_asynq=True
        )
        return CallableValue(sig)
    elif isinstance(val, ExternalType):
        try:
            typ = qcore.helpers.object_from_string(val.type_path)
        except Exception:
            ctx.show_error(f"Cannot resolve type {val.type_path!r}")
            return AnyValue(AnySource.error)
        return _type_from_runtime(typ, ctx)
    # Python 3.6 only (on later versions Required/NotRequired match
    # is_generic_type).
    elif is_instance_of_typing_name(val, "_MaybeRequired"):
        required = is_instance_of_typing_name(val, "_Required")
        if is_typeddict:
            return _Pep655Value(required, _type_from_runtime(val.__type__, ctx))
        else:
            cls = "Required" if required else "NotRequired"
            ctx.show_error(f"{cls}[] used in unsupported context")
            return AnyValue(AnySource.error)
    else:
        origin = get_origin(val)
        if isinstance(origin, type):
            return _maybe_typed_value(origin)
        elif val is NamedTuple:
            return TypedValue(tuple)
        ctx.show_error(f"Invalid type annotation {val}")
        return AnyValue(AnySource.error)


def _get_typeddict_value(
    value: Value,
    ctx: Context,
    key: str,
    required_keys: Optional[Container[str]],
    total: bool,
) -> Tuple[bool, Value]:
    val = _type_from_runtime(value, ctx, is_typeddict=True)
    if isinstance(val, _Pep655Value):
        return (val.required, val.value)
    if required_keys is None:
        required = total
    else:
        required = key in required_keys
    return required, val


def _eval_forward_ref(val: str, ctx: Context, is_typeddict: bool = False) -> Value:
    try:
        tree = ast.parse(val, mode="eval")
    except SyntaxError:
        ctx.show_error(f"Syntax error in type annotation: {val}")
        return AnyValue(AnySource.error)
    else:
        return _type_from_ast(tree.body, ctx, is_typeddict=is_typeddict)


def _type_from_value(value: Value, ctx: Context, is_typeddict: bool = False) -> Value:
    if isinstance(value, KnownValue):
        return _type_from_runtime(value.val, ctx, is_typeddict=is_typeddict)
    elif isinstance(value, TypeVarValue):
        return value
    elif isinstance(value, MultiValuedValue):
        return unite_values(*[_type_from_value(val, ctx) for val in value.vals])
    elif isinstance(value, AnnotatedValue):
        return _type_from_value(value.value, ctx)
    elif isinstance(value, _SubscriptedValue):
        return _type_from_subscripted_value(
            value.root, value.members, ctx, is_typeddict=is_typeddict
        )
    elif isinstance(value, AnyValue):
        return value
    elif isinstance(value, TypedValue) and isinstance(value.typ, str):
        # Synthetic type
        return value
    else:
        ctx.show_error(f"Unrecognized annotation {value}")
        return AnyValue(AnySource.error)


def _type_from_subscripted_value(
    root: Optional[Value],
    members: Sequence[Value],
    ctx: Context,
    is_typeddict: bool = False,
) -> Value:
    if isinstance(root, GenericValue):
        if len(root.args) == len(members):
            return GenericValue(
                root.typ, [_type_from_value(member, ctx) for member in members]
            )
    if isinstance(root, _SubscriptedValue):
        root_type = _type_from_value(root, ctx)
        return _type_from_subscripted_value(root_type, members, ctx)
    elif isinstance(root, MultiValuedValue):
        return unite_values(
            *[
                _type_from_subscripted_value(subval, members, ctx, is_typeddict)
                for subval in root.vals
            ]
        )
    if isinstance(root, TypedValue) and isinstance(root.typ, str):
        return GenericValue(root.typ, [_type_from_value(elt, ctx) for elt in members])

    if not isinstance(root, KnownValue):
        ctx.show_error(f"Cannot resolve subscripted annotation: {root}")
        return AnyValue(AnySource.error)
    root = root.val
    if root is typing.Union:
        return unite_values(*[_type_from_value(elt, ctx) for elt in members])
    elif is_typing_name(root, "Literal"):
        # Note that in Python 3.8, the way typing's internal cache works means that
        # Literal[1] and Literal[True] are cached to the same value, so if you use
        # both, you'll get whichever one was used first in later calls. There's nothing
        # we can do about that.
        if all(isinstance(elt, KnownValue) for elt in members):
            return unite_values(*members)
        else:
            ctx.show_error(f"Arguments to Literal[] must be literals, not {members}")
            return AnyValue(AnySource.error)
    elif root is typing.Tuple or root is tuple:
        if len(members) == 2 and members[1] == KnownValue(Ellipsis):
            return GenericValue(tuple, [_type_from_value(members[0], ctx)])
        elif len(members) == 1 and members[0] == KnownValue(()):
            return SequenceIncompleteValue(tuple, [])
        else:
            return SequenceIncompleteValue(
                tuple, [_type_from_value(arg, ctx) for arg in members]
            )
    elif root is typing.Optional:
        if len(members) != 1:
            ctx.show_error("Optional[] takes only one argument")
            return AnyValue(AnySource.error)
        return unite_values(KnownValue(None), _type_from_value(members[0], ctx))
    elif root is typing.Type or root is type:
        if len(members) != 1:
            ctx.show_error("Type[] takes only one argument")
            return AnyValue(AnySource.error)
        argument = _type_from_value(members[0], ctx)
        return SubclassValue.make(argument)
    elif is_typing_name(root, "Annotated"):
        origin, *metadata = members
        return _make_annotated(_type_from_value(origin, ctx), metadata, ctx)
    elif is_typing_name(root, "TypeGuard"):
        if len(members) != 1:
            ctx.show_error("TypeGuard requires a single argument")
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeGuardExtension(_type_from_value(members[0], ctx))]
        )
    elif is_typing_name(root, "Required"):
        if not is_typeddict:
            ctx.show_error("Required[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(members) != 1:
            ctx.show_error("Required[] requires a single argument")
            return AnyValue(AnySource.error)
        return _Pep655Value(True, _type_from_value(members[0], ctx))
    elif is_typing_name(root, "NotRequired"):
        if not is_typeddict:
            ctx.show_error("NotRequired[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(members) != 1:
            ctx.show_error("NotRequired[] requires a single argument")
            return AnyValue(AnySource.error)
        return _Pep655Value(False, _type_from_value(members[0], ctx))
    elif root is Callable or root is typing.Callable:
        if len(members) == 2:
            args, return_value = members
            return _make_callable_from_value(args, return_value, ctx)
        ctx.show_error("Callable requires exactly two arguments")
        return AnyValue(AnySource.error)
    elif root is AsynqCallable:
        if len(members) == 2:
            args, return_value = members
            return _make_callable_from_value(args, return_value, ctx, is_asynq=True)
        ctx.show_error("AsynqCallable requires exactly two arguments")
        return AnyValue(AnySource.error)
    elif typing_inspect.is_generic_type(root):
        origin = typing_inspect.get_origin(root)
        if origin is None:
            # On Python 3.9 at least, get_origin() of a class that inherits
            # from Generic[T] is None.
            origin = root
        if getattr(origin, "__extra__", None) is not None:
            origin = origin.__extra__
        return GenericValue(origin, [_type_from_value(elt, ctx) for elt in members])
    elif isinstance(root, type):
        return GenericValue(root, [_type_from_value(elt, ctx) for elt in members])
    else:
        # In Python 3.9, generics are implemented differently and typing.get_origin
        # can help.
        origin = get_origin(root)
        if isinstance(origin, type):
            return GenericValue(origin, [_type_from_value(elt, ctx) for elt in members])
        ctx.show_error(f"Unrecognized subscripted annotation: {root}")
        return AnyValue(AnySource.error)


class _DefaultContext(Context):
    def __init__(
        self,
        visitor: "NameCheckVisitor",
        node: Optional[ast.AST],
        globals: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        self.visitor = visitor
        self.node = node
        self.globals = globals

    def show_error(
        self, message: str, error_code: ErrorCode = ErrorCode.invalid_annotation
    ) -> None:
        if self.visitor is not None and self.node is not None:
            self.visitor.show_error(self.node, message, error_code)

    def get_name(self, node: ast.Name) -> Value:
        if self.visitor is not None:
            return self.visitor.resolve_name(
                node,
                error_node=self.node,
                suppress_errors=self.should_suppress_undefined_names,
            )
        elif self.globals is not None:
            if node.id in self.globals:
                return KnownValue(self.globals[node.id])
            elif hasattr(builtins, node.id):
                return KnownValue(getattr(builtins, node.id))
        if self.should_suppress_undefined_names:
            return AnyValue(AnySource.inference)
        self.show_error(
            f"Undefined name {node.id!r} used in annotation", ErrorCode.undefined_name
        )
        return AnyValue(AnySource.error)


@dataclass(frozen=True)
class _SubscriptedValue(Value):
    root: Optional[Value]
    members: Tuple[Value, ...]


@dataclass
class _Pep655Value(Value):
    required: bool
    value: Value


class _Visitor(ast.NodeVisitor):
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def generic_visit(self, node: ast.AST) -> None:
        raise NotImplementedError(f"no visitor implemented for {node!r}")

    def visit_Name(self, node: ast.Name) -> Value:
        return self.ctx.get_name(node)

    def visit_Subscript(self, node: ast.Subscript) -> Value:
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if isinstance(index, SequenceIncompleteValue):
            members = index.members
        else:
            members = (index,)
        return _SubscriptedValue(value, members)

    def visit_Attribute(self, node: ast.Attribute) -> Optional[Value]:
        root_value = self.visit(node.value)
        if isinstance(root_value, KnownValue):
            try:
                return KnownValue(getattr(root_value.val, node.attr))
            except AttributeError:
                self.ctx.show_error(
                    f"{root_value.val!r} has no attribute {node.attr!r}"
                )
                return AnyValue(AnySource.error)
        elif not isinstance(root_value, AnyValue):
            self.ctx.show_error(f"Cannot resolve annotation {root_value}")
        return AnyValue(AnySource.error)

    def visit_Tuple(self, node: ast.Tuple) -> Value:
        elts = [self.visit(elt) for elt in node.elts]
        return SequenceIncompleteValue(tuple, elts)

    def visit_List(self, node: ast.List) -> Value:
        elts = [self.visit(elt) for elt in node.elts]
        return SequenceIncompleteValue(list, elts)

    def visit_Index(self, node: ast.Index) -> Value:
        # class is unused in 3.9
        return self.visit(node.value)  # static analysis: ignore[undefined_attribute]

    def visit_Ellipsis(self, node: ast.Ellipsis) -> Value:
        return KnownValue(Ellipsis)

    def visit_Constant(self, node: ast.Constant) -> Value:
        return KnownValue(node.value)

    def visit_NameConstant(self, node: ast.NameConstant) -> Value:
        return KnownValue(node.value)

    def visit_Num(self, node: ast.Num) -> Value:
        return KnownValue(node.n)

    def visit_Str(self, node: ast.Str) -> Value:
        return KnownValue(node.s)

    def visit_Bytes(self, node: ast.Bytes) -> Value:
        return KnownValue(node.s)

    def visit_Expr(self, node: ast.Expr) -> Value:
        return self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> Optional[Value]:
        if isinstance(node.op, (ast.BitOr, ast3.BitOr)):
            return _SubscriptedValue(
                KnownValue(Union), (self.visit(node.left), self.visit(node.right))
            )
        else:
            return None

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Optional[Value]:
        # Only int and float negation on literals are supported.
        if isinstance(node.op, (ast.USub, ast3.USub)):
            operand = self.visit(node.operand)
            if isinstance(operand, KnownValue) and isinstance(
                operand.val, (int, float)
            ):
                return KnownValue(-operand.val)
        return None

    def visit_Call(self, node: ast.Call) -> Optional[Value]:
        func = self.visit(node.func)
        if not isinstance(func, KnownValue):
            return None
        if func.val == NewType:
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            args = []
            kwargs = {}
            for arg_value in arg_values:
                if isinstance(arg_value, KnownValue):
                    args.append(arg_value.val)
                else:
                    return None
            for name, kwarg_value in kwarg_values:
                if name is None:
                    if isinstance(kwarg_value, KnownValue) and isinstance(
                        kwarg_value.val, dict
                    ):
                        kwargs.update(kwarg_value.val)
                    else:
                        return None
                else:
                    if isinstance(kwarg_value, KnownValue):
                        kwargs[name] = kwarg_value.val
                    else:
                        print(kwarg_value)
                        return None
            return KnownValue(func.val(*args, **kwargs))
        elif func.val == TypeVar:
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            if not arg_values:
                self.ctx.show_error("TypeVar() requires at least one argument")
                return None
            name_val = arg_values[0]
            if not isinstance(name_val, KnownValue):
                self.ctx.show_error("TypeVar name must be a literal")
                return None
            constraints = []
            for arg_value in arg_values[1:]:
                constraints.append(_type_from_value(arg_value, self.ctx))
            bound = None
            for name, kwarg_value in kwarg_values:
                if name in ("covariant", "contravariant"):
                    continue
                elif name == "bound":
                    bound = _type_from_value(kwarg_value, self.ctx)
                else:
                    self.ctx.show_error(f"Unrecognized TypeVar kwarg {name}")
                    return None
            tv = TypeVar(name_val.val)
            return TypeVarValue(tv, bound, tuple(constraints))
        elif isinstance(func.val, type):
            if func.val is object:
                return AnyValue(AnySource.inference)
            return TypedValue(func.val)
        else:
            return None


def _value_of_origin_args(
    origin: object,
    args: Sequence[object],
    val: object,
    ctx: Context,
    is_typeddict: bool = False,
) -> Value:
    if origin is typing.Type or origin is type:
        if not args:
            return TypedValue(type)
        return SubclassValue.make(_type_from_runtime(args[0], ctx))
    elif origin is typing.Tuple or origin is tuple:
        if not args:
            return TypedValue(tuple)
        elif len(args) == 2 and args[1] is Ellipsis:
            return GenericValue(tuple, [_type_from_runtime(args[0], ctx)])
        elif len(args) == 1 and args[0] == ():
            return SequenceIncompleteValue(tuple, [])
        else:
            args_vals = [_type_from_runtime(arg, ctx) for arg in args]
            return SequenceIncompleteValue(tuple, args_vals)
    elif origin is typing.Union:
        return unite_values(*[_type_from_runtime(arg, ctx) for arg in args])
    elif origin is Callable or origin is typing.Callable:
        if len(args) == 2 and args[0] is Ellipsis:
            return CallableValue(
                Signature.make(
                    [], _type_from_runtime(args[1], ctx), is_ellipsis_args=True
                )
            )
        elif len(args) == 0:
            return TypedValue(Callable)
        *arg_types, return_type = args
        if len(arg_types) == 1 and isinstance(arg_types[0], list):
            arg_types = arg_types[0]
        params = [
            SigParameter(
                f"__arg{i}",
                kind=SigParameter.POSITIONAL_ONLY,
                annotation=_type_from_runtime(arg, ctx, is_typeddict=True),
            )
            for i, arg in enumerate(arg_types)
        ]
        sig = Signature.make(params, _type_from_runtime(return_type, ctx))
        return CallableValue(sig)
    elif is_typing_name(origin, "Annotated"):
        origin, metadata = args
        # This should never happen
        if not isinstance(metadata, Iterable):
            ctx.show_error("Unexpected format in Annotated")
            return AnyValue(AnySource.error)
        return _make_annotated(
            _type_from_runtime(origin, ctx),
            [KnownValue(data) for data in metadata],
            ctx,
        )
    elif isinstance(origin, type):
        # turn typing.List into list in some Python versions
        # compare https://github.com/ilevkivskyi/typing_inspect/issues/36
        extra_origin = getattr(origin, "__extra__", None)
        if extra_origin is not None:
            origin = extra_origin
        if args:
            args_vals = [_type_from_runtime(val, ctx) for val in args]
            if all(isinstance(val, AnyValue) for val in args_vals):
                return _maybe_typed_value(origin)
            return GenericValue(origin, args_vals)
        else:
            return _maybe_typed_value(origin)
    elif is_typing_name(origin, "TypeGuard"):
        if len(args) != 1:
            ctx.show_error("TypeGuard requires a single argument")
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeGuardExtension(_type_from_runtime(args[0], ctx))]
        )
    elif is_typing_name(origin, "Final"):
        if len(args) != 1:
            ctx.show_error("Final requires a single argument")
            return AnyValue(AnySource.error)
        # TODO(#160): properly support Final
        return _type_from_runtime(args[0], ctx)
    elif is_typing_name(origin, "ClassVar"):
        if len(args) != 1:
            ctx.show_error("ClassVar requires a single argument")
            return AnyValue(AnySource.error)
        return _type_from_runtime(args[0], ctx)
    elif is_typing_name(origin, "Required"):
        if not is_typeddict:
            ctx.show_error("Required[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(args) != 1:
            ctx.show_error("Required[] requires a single argument")
            return AnyValue(AnySource.error)
        return _Pep655Value(True, _type_from_runtime(args[0], ctx))
    elif is_typing_name(origin, "NotRequired"):
        if not is_typeddict:
            ctx.show_error("NotRequired[] used in unsupported context")
            return AnyValue(AnySource.error)
        if len(args) != 1:
            ctx.show_error("NotRequired[] requires a single argument")
            return AnyValue(AnySource.error)
        return _Pep655Value(False, _type_from_runtime(args[0], ctx))
    elif origin is None and isinstance(val, type):
        # This happens for SupportsInt in 3.7.
        return _maybe_typed_value(val)
    else:
        ctx.show_error(
            f"Unrecognized annotation {origin}[{', '.join(map(repr, args))}]"
        )
        return AnyValue(AnySource.error)


def _maybe_typed_value(val: type) -> Value:
    if val is type(None):
        return KnownValue(None)
    return TypedValue(val)


def _make_callable_from_value(
    args: Value, return_value: Value, ctx: Context, is_asynq: bool = False
) -> Value:
    return_annotation = _type_from_value(return_value, ctx)
    if args == KnownValue(Ellipsis):
        return CallableValue(
            Signature.make(
                [],
                return_annotation=return_annotation,
                is_ellipsis_args=True,
                is_asynq=is_asynq,
            )
        )
    elif isinstance(args, SequenceIncompleteValue):
        params = [
            SigParameter(
                f"__arg{i}",
                kind=SigParameter.POSITIONAL_ONLY,
                annotation=_type_from_value(arg, ctx),
            )
            for i, arg in enumerate(args.members)
        ]
        sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        return CallableValue(sig)
    else:
        return AnyValue(AnySource.inference)


def _make_annotated(origin: Value, metadata: Sequence[Value], ctx: Context) -> Value:
    metadata = [_value_from_metadata(entry, ctx) for entry in metadata]
    return annotate_value(origin, metadata)


def _value_from_metadata(entry: Value, ctx: Context) -> Union[Value, Extension]:
    if isinstance(entry, KnownValue):
        if isinstance(entry.val, ParameterTypeGuard):
            return ParameterTypeGuardExtension(
                entry.val.varname, _type_from_runtime(entry.val.guarded_type, ctx)
            )
        elif isinstance(entry.val, HasAttrGuard):
            return HasAttrGuardExtension(
                entry.val.varname,
                _type_from_runtime(entry.val.attribute_name, ctx),
                _type_from_runtime(entry.val.attribute_type, ctx),
            )
        elif isinstance(entry.val, CustomCheck):
            return CustomCheckExtension(entry.val)
    return entry
