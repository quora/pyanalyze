"""

Code for understanding type annotations.

"""
import mypy_extensions
import typing_extensions
import typing
import typing_inspect
import qcore
import ast
from six.moves import builtins

try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable

from .error_code import ErrorCode
from .value import (
    KnownValue,
    NO_RETURN_VALUE,
    UNRESOLVED_VALUE,
    TypedValue,
    SequenceIncompleteValue,
    unite_values,
    Value,
    GenericValue,
    SubclassValue,
    TypedDictValue,
    NewTypeValue,
)

try:
    from typing import get_origin  # Python 3.9
except ImportError:

    def get_origin(obj):
        return None


def type_from_ast(ast_node, visitor=None):
    """Given an AST node representing an annotation, return a Value."""
    ctx = _Context(visitor, ast_node)
    return _type_from_ast(ast_node, ctx)


def type_from_runtime(val, visitor=None, node=None, globals=None):
    """Given a runtime annotation object, return a Value."""
    ctx = _Context(visitor, node, globals)
    return _type_from_runtime(val, ctx)


def type_from_value(value, visitor=None, node=None):
    """Given a Value from resolving an annotation, return the type."""
    ctx = _Context(visitor, node)
    return _type_from_value(value, ctx)


def _type_from_ast(node, ctx):
    val = _Visitor(ctx).visit(node)
    if val is None:
        # TODO show an error here
        return UNRESOLVED_VALUE
    return _type_from_value(val, ctx)


def _type_from_runtime(val, ctx):
    if isinstance(val, str):
        return _eval_forward_ref(val, ctx)
    elif isinstance(val, tuple):
        # This happens under some Python versions for types
        # nested in tuples, e.g. on 3.6:
        # > typing_inspect.get_args(Union[Set[int], List[str]])
        # ((typing.Set, int), (typing.List, str))
        origin = val[0]
        if len(val) == 2:
            args = (val[1],)
        else:
            args = val[1:]
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
        else:
            args_vals = [_type_from_runtime(arg, ctx) for arg in args]
            return SequenceIncompleteValue(tuple, args_vals)
    elif is_instance_of_typing_name(val, "_TypedDictMeta"):
        return TypedDictValue(
            {
                key: _type_from_runtime(value, ctx)
                for key, value in val.__annotations__.items()
            }
        )
    elif typing_inspect.is_callable_type(val):
        return TypedValue(Callable)
    elif typing_inspect.is_generic_type(val):
        origin = typing_inspect.get_origin(val)
        args = typing_inspect.get_args(val)
        return _value_of_origin_args(origin, args, val, ctx)
    elif isinstance(val, type):
        if val is type(None):
            return KnownValue(None)
        return TypedValue(val)
    elif val is None:
        return KnownValue(None)
    elif is_typing_name(val, "NoReturn"):
        return NO_RETURN_VALUE
    elif val is typing.Any:
        return UNRESOLVED_VALUE
    elif hasattr(val, "__supertype__"):
        if isinstance(val.__supertype__, type):
            # NewType
            return NewTypeValue(val)
        elif typing_inspect.is_tuple_type(val.__supertype__):
            # TODO figure out how to make NewTypes over tuples work
            return UNRESOLVED_VALUE
        else:
            ctx.show_error("Invalid NewType %s" % (val,))
            return UNRESOLVED_VALUE
    elif typing_inspect.is_typevar(val):
        # TypeVar; not supported yet
        return UNRESOLVED_VALUE
    elif typing_inspect.is_classvar(val):
        return UNRESOLVED_VALUE
    elif is_instance_of_typing_name(val, "_ForwardRef") or is_instance_of_typing_name(
        val, "ForwardRef"
    ):
        # This has issues because the forward ref may be defined in a different file, in
        # which case we don't know which names are valid in it.
        with qcore.override(ctx, "suppress_undefined_name", True):
            return UNRESOLVED_VALUE
    elif val is Ellipsis:
        # valid in Callable[..., ]
        return UNRESOLVED_VALUE
    elif is_instance_of_typing_name(val, "_TypeAlias"):
        # typing.Pattern and Match, which are not normal generic types for some reason
        return GenericValue(val.impl_type, [_type_from_runtime(val.type_var, ctx)])
    else:
        origin = get_origin(val)
        if origin is not None:
            return TypedValue(origin)
        ctx.show_error("Invalid type annotation %s" % (val,))
        return UNRESOLVED_VALUE


def _eval_forward_ref(val, ctx):
    try:
        tree = ast.parse(val, mode="eval")
    except SyntaxError:
        ctx.show_error("Syntax error in type annotation: %s" % (val,))
        return UNRESOLVED_VALUE
    else:
        return _type_from_ast(tree.body, ctx)


def _type_from_value(value, ctx):
    if isinstance(value, KnownValue):
        return _type_from_runtime(value.val, ctx)
    elif isinstance(value, _SubscriptedValue):
        if not isinstance(value.root, KnownValue):
            ctx.show_error("Cannot resolve subscripted annotation: %s" % (value.root,))
            return UNRESOLVED_VALUE
        root = value.root.val
        if root is typing.Union:
            return unite_values(*[_type_from_value(elt, ctx) for elt in value.members])
        elif is_typing_name(root, "Literal"):
            if all(isinstance(elt, KnownValue) for elt in value.members):
                return unite_values(value.members)
            else:
                ctx.show_error(
                    "Arguments to Literal[] must be literals, not %s" % (value.members,)
                )
                return UNRESOLVED_VALUE
        elif root is typing.Optional:
            if len(value.members) != 1:
                ctx.show_error("Optional[] takes only one argument")
                return UNRESOLVED_VALUE
            return unite_values(
                KnownValue(None), _type_from_value(value.members[0], ctx)
            )
        elif root is typing.Type:
            if len(value.members) != 1:
                ctx.show_error("Type[] takes only one argument")
                return UNRESOLVED_VALUE
            argument = _type_from_value(value.members[0], ctx)
            if isinstance(argument, TypedValue) and isinstance(argument.typ, type):
                return SubclassValue(argument.typ)
            return TypedValue(type)
        elif typing_inspect.is_generic_type(root):
            origin = typing_inspect.get_origin(root)
            if getattr(origin, "__extra__", None) is not None:
                origin = origin.__extra__
            return GenericValue(
                origin, [_type_from_value(elt, ctx) for elt in value.members]
            )
        elif isinstance(root, type):
            return GenericValue(
                root, [_type_from_value(elt, ctx) for elt in value.members]
            )
        else:
            # In Python 3.9, generics are implemented differently and typing.get_origin
            # can help.
            origin = get_origin(root)
            if origin is not None:
                return GenericValue(
                    origin, [_type_from_value(elt, ctx) for elt in value.members]
                )
            ctx.show_error("Unrecognized subscripted annotation: %s" % (root,))
            return UNRESOLVED_VALUE
    else:
        return UNRESOLVED_VALUE


class _Context(object):
    def __init__(self, visitor, node, globals=None):
        self.visitor = visitor
        self.node = node
        self.globals = globals
        self.suppress_undefined_name = False

    def show_error(self, message, error_code=ErrorCode.invalid_annotation):
        if self.visitor is not None:
            self.visitor.show_error(self.node, message, error_code)

    def get_name(self, node):
        if self.visitor is not None:
            return self.visitor.resolve_name(
                node, error_node=self.node, suppress_errors=self.suppress_undefined_name
            )
        elif self.globals is not None:
            if node.id in self.globals:
                return KnownValue(self.globals[node.id])
            elif hasattr(builtins, node.id):
                return KnownValue(getattr(builtins, node.id))
            else:
                return UNRESOLVED_VALUE
        else:
            return UNRESOLVED_VALUE


class _SubscriptedValue(Value):
    def __init__(self, root, members):
        self.root = root
        self.members = members


class _Visitor(ast.NodeVisitor):
    def __init__(self, ctx):
        self.ctx = ctx

    def visit_Name(self, node):
        return self.ctx.get_name(node)

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if isinstance(index, SequenceIncompleteValue):
            members = index.members
        else:
            members = [index]
        return _SubscriptedValue(value, members)

    def visit_Attribute(self, node):
        root_value = self.visit(node.value)
        if isinstance(root_value, KnownValue):
            try:
                return KnownValue(getattr(root_value.val, node.attr))
            except AttributeError:
                return None
        return None

    def visit_Tuple(self, node):
        elts = [self.visit(elt) for elt in node.elts]
        return SequenceIncompleteValue(tuple, elts)

    def visit_List(self, node):
        elts = [self.visit(elt) for elt in node.elts]
        return SequenceIncompleteValue(list, elts)

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Ellipsis(self, node):
        return KnownValue(Ellipsis)


def is_typing_name(obj, name):
    for mod in (typing, typing_extensions, mypy_extensions):
        try:
            typing_obj = getattr(mod, name)
        except AttributeError:
            continue
        else:
            if obj is typing_obj:
                return True
    return False


def is_instance_of_typing_name(obj, name):
    for mod in (typing, typing_extensions, mypy_extensions):
        try:
            typing_obj = getattr(mod, name)
        except AttributeError:
            continue
        else:
            if isinstance(obj, typing_obj):
                return True
    return False


def _value_of_origin_args(origin, args, val, ctx):
    if origin is typing.Type or origin is type:
        if isinstance(args[0], type):
            return SubclassValue(args[0])
        elif args[0] is typing.Any:
            return TypedValue(type)
        else:
            # Perhaps a forward reference
            return UNRESOLVED_VALUE
    elif origin is typing.Tuple:
        if not args:
            return TypedValue(tuple)
        elif len(args) == 2 and args[1] is Ellipsis:
            return GenericValue(tuple, [_type_from_runtime(args[0], ctx)])
        else:
            args_vals = [_type_from_runtime(arg, ctx) for arg in args]
            return SequenceIncompleteValue(tuple, args_vals)
    elif isinstance(origin, type):
        # turn typing.List into list in some Python versions
        # compare https://github.com/ilevkivskyi/typing_inspect/issues/36
        if getattr(origin, "__extra__", None) is not None:
            origin = origin.__extra__
        if args:
            args_vals = [_type_from_runtime(val, ctx) for val in args]
            if all(val is UNRESOLVED_VALUE for val in args_vals):
                return TypedValue(origin)
            return GenericValue(origin, args_vals)
        else:
            return TypedValue(origin)
    elif origin is None and isinstance(val, type):
        # This happens for SupportsInt in 3.7.
        return TypedValue(val)
    else:
        return UNRESOLVED_VALUE
