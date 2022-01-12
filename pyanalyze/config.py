"""

Module-specific configurations for test_scope.

This is in the process of being replaced with options.py.

"""
import abc
import asynq
import enum
import inspect
import qcore
from unittest import mock
import asyncio
from types import ModuleType
from typing import Any, Callable, Dict, Sequence, Set, TYPE_CHECKING, Tuple, Union

from . import value
from .extensions import CustomCheck

if TYPE_CHECKING:
    from .arg_spec import ArgSpecCache
    from .signature import ConcreteSignature
    from .reexport import ImplicitReexportTracker


class Config(object):
    """Base class for configurations."""

    #
    # Used in several parts of test_scope
    #

    # Deprecated
    # default module to run on, if any
    DEFAULT_BASE_MODULE = None

    # Replaced with options.Paths
    # file paths to run on by default
    DEFAULT_DIRS = ()

    # arg_spec.UnwrapClass
    def unwrap_cls(self, cls: type) -> type:
        """Does any application-specific unwrapping logic for wrapper classes."""
        return cls

    # arg_spec.ConstructorHooks
    def get_constructor(
        self, cls: type
    ) -> Union[None, "ConcreteSignature", inspect.Signature, Callable[..., Any]]:
        """Return a constructor signature for this class.

        May return either a function that pyanalyze will use the signature of, an inspect
        Signature object, or a pyanalyze Signature object. The function or signature
        should take a self parameter.

        """
        return None

    #
    # Used by name_check_visitor.py
    #

    # Replaced with Options for each error code.
    # Sets of errors that are enabled or disabled. By default,
    # all errors are enabled except those in error_code.DISABLED_BY_DEFAULT.
    ENABLED_ERRORS = set()
    DISABLED_ERRORS = set()

    # Replaced with options.EnforceNoUnused
    # If true, an error is raised when pyanalyze finds any unused objects.
    ENFORCE_NO_UNUSED_OBJECTS = False

    # Replaced with options.ForLoopAlwaysEntered
    # If True, we assume that for loops are always entered at least once,
    # which affects the potentially_undefined_name check. This will miss
    # some bugs but also remove some annoying false positives.
    FOR_LOOP_ALWAYS_ENTERED = False

    # Replaced with options.IgnoreNoneAttributes
    # If True, we ignore None when type checking attribute access on a Union
    # type.
    IGNORE_NONE_ATTRIBUTES = False

    # Replaced with options.IgnoredPaths
    # Attribute accesses on these do not result in errors
    IGNORED_PATHS = []

    # options.ExtraBuiltins
    # Even if these variables are undefined, no errors are shown
    IGNORED_VARIABLES = {"__IPYTHON__"}  # special global defined in IPython

    # Replaced with options.IgnoredEndOfReference
    # When these attributes are accessed but they don't exist, the error is ignored
    IGNORED_END_OF_REFERENCE = {
        # these are created by the mock module
        "call_count",
        "assert_has_calls",
        "reset_mock",
        "called",
        "assert_called_once",
        "assert_called_once_with",
        "assert_called_with",
        "count",
        "assert_any_call",
        "assert_not_called",
    }

    # options.UnimportableModules
    # Do not attempt to import these modules if they are imported within a function
    UNIMPORTABLE_MODULES = set()

    # options.IgnoredCallees
    # Calls to these aren't checked for argument validity
    IGNORED_CALLEES = [
        # not a set because trying to include a.mocklib.call in a set complains about a dict not
        # being hashable
        # getargspec gets confused about this subclass of tuple that overrides __new__ and __call__
        mock.call,
        mock.MagicMock,
        mock.Mock,
    ]

    # Use file-specific overrides instead
    # In these files, we do not check for duplicate dictionary keys
    IGNORED_FILES_FOR_DUPLICATE_DICT_KEYS = set()

    # Use file-specific overrides instead
    # In these files, strings with non-ASCII characters do not require explicit prefixes
    IGNORED_FILES_FOR_EXPLICIT_STRING_LITERALS = set()

    # arg_spec.ClassesSafeToInstantiate
    # test_scope will instantiate instances of these classes if it can infer the value of all of
    # their arguments. This is useful mostly for classes that are commonly instantiated with static
    # arguments.
    CLASSES_SAFE_TO_INSTANTIATE = (
        CustomCheck,
        value.Value,
        value.Extension,
        value.KVPair,
        asynq.ConstFuture,
        range,
        tuple,
    )

    # arg_spec.FunctionsSafeToCall
    # Similarly, these functions will be called
    FUNCTIONS_SAFE_TO_CALL = (
        # we can't have len here because of code like l = []; l.append('foo'); 3 / len(l)
        # which would incorrrectly be flagged as division by zero because test_scope doesn't know
        # what append does
        sorted,
        asynq.asynq,
        value.make_weak,
    )

    # name_check_visitor.SafeDecoratorsForNestedFunctions
    # These decorators can safely be applied to nested functions. If True, they take arguments.
    SAFE_DECORATORS_FOR_NESTED_FUNCTIONS = {
        asynq.asynq: True,
        classmethod: False,
        staticmethod: False,
    }
    if asyncio is not None:
        SAFE_DECORATORS_FOR_NESTED_FUNCTIONS[asyncio.coroutine] = False

    # attributes.TreatClassAttributeAsAny
    def should_ignore_class_attribute(self, cls_val: object) -> bool:
        return cls_val is None or cls_val is NotImplemented

    # Set of dunder methods (e.g., '{"__lshift__"}') that pyanalyze is not allowed to call on
    # objects.
    # name_check_visitor.DisallowCallsToDunders
    DISALLOW_CALLS_TO_DUNDERS: Set[str] = set()

    # Decorators that are equivalent to asynq.asynq and asynq.async_proxy.
    # name_check_visitor.AsynqDecorators
    ASYNQ_DECORATORS: Set[object] = {asynq.asynq}
    # name_check_visitor.AsyncProxyDecorators
    ASYNC_PROXY_DECORATORS: Set[object] = {asynq.async_proxy}

    # name_check_visitor.ComprehensionLengthInferenceLimit
    # If we iterate over something longer than this, we don't try to infer precise
    # types for comprehensions. Increasing this can hurt performance.
    COMPREHENSION_LENGTH_INFERENCE_LIMIT: int = 25
    # name_check_visitor.UnionSimplificationLimit
    # We may simplify unions with more than this many values.
    UNION_SIMPLIFICATION_LIMIT: int = 100

    #
    # Used for VariableNameValue
    #

    # List of VariableNameValue instances that create pseudo-types associated with certain variable
    # names
    VARIABLE_NAME_VALUES: Sequence[value.VariableNameValue] = []

    @qcore.caching.cached_per_instance()
    def varname_value_map(self) -> Dict[str, value.VariableNameValue]:
        """Returns a map of variable name to applicable VariableNameValue object."""
        ret = {}
        for val in self.VARIABLE_NAME_VALUES:
            for varname in val.varnames:
                ret[varname] = val
        return ret

    #
    # Used by find_unused.py
    #

    # unused
    def registered_values(self) -> Set[object]:
        """Returns a set of objects that are registered by various decorators.

        These are excluded from the find_unused check.

        """
        return set()

    # shared_options.IgnoreUnused
    def should_ignore_unused(
        self, module: ModuleType, attribute: str, object: object
    ) -> bool:
        """If this returns True, we will exclude this object from the unused object check.

        The arguments are the module the object was found in, the attribute used to
        access it, and the object itself.

        """
        return False

    #
    # Used by asynq_checker.py
    #

    # Normally, async calls to async functions are only enforced in functions that are already
    # async. In subclasses of classes listed here, all async functions must be called async.
    # asynq_checker.ClassesCheckedForAsynq
    BASE_CLASSES_CHECKED_FOR_ASYNQ: Set[type] = set()

    # Async batching in these component methods isn't checked even when they exist on a class in
    # BASE_CLASSES_CHECKED_FOR_ASYNQ
    # asynq_checker.MethodsNotCheckedForAsynq
    METHODS_NOT_CHECKED_FOR_ASYNQ: Set[str] = set()

    # We ignore _async methdos in these modules.
    # asynq_checker.NonAsynqModules
    NON_ASYNQ_MODULES: Set[str] = {"multiprocessing"}

    #
    # Used by ClassAttributeChecker
    #

    # When these attributes are unused, they are not listed as such by the unused attribute finder
    # name_check_visitor.IgnoredUnusedAttributes
    IGNORED_UNUSED_ATTRS: Set[str] = {
        # ABCs
        "_abc_cache",
        "_abc_negative_cache",
        "__abstractmethods__",
        "_abc_negative_cache_version",
        "_abc_registry",
        # Python core
        "__module__",
        "__doc__",
        "__init__",
        "__dict__",
        "__weakref__",
        "__enter__",
        "__exit__",
        "__metaclass__",
    }

    # List of pairs of (class, set of attribute names). When these attribute names are seen as
    # unused on a child or base class of the class, they are not listed.
    # name_check_visitor.IgnoredUnusedClassAttributes
    IGNORED_UNUSED_ATTRS_BY_CLASS: Sequence[Tuple[type, Set[str]]] = []

    # Used in the check for object attributes that are accessed but not set. In general, the check
    # will only alert about attributes that don't exist when it has visited all the base classes of
    # the class with the possibly missing attribute. However, these classes are never going to be
    # visited (since they're builtin), but they don't set any attributes that we rely on.
    # name_check_visitor.IgnoredTypesForAttributeChecking
    IGNORED_TYPES_FOR_ATTRIBUTE_CHECKING: Set[type] = {object, abc.ABC}

    #
    # Used by arg_spec.py
    #
    # arg_spec.KnownSignatures
    def get_known_argspecs(
        self, arg_spec_cache: "ArgSpecCache"
    ) -> Dict[object, "ConcreteSignature"]:
        """Initialize any hardcoded argspecs."""
        return {}

    # name_check_visitor.AllowDuplicateValues
    def should_check_class_for_duplicate_values(self, cls: type) -> bool:
        """Whether we should produce an error if this class contains duplicate values.

        Used for the duplicate_enum check.

        """
        return issubclass(cls, enum.Enum)

    # checker.AdditionalBaseProviders
    def get_additional_bases(self, typ: Union[type, super]) -> Set[type]:
        """Return additional classes that should be considered bae classes of typ."""
        return set()

    #
    # Used by reexport.py
    #
    # reexport.ReexportConfig
    def configure_reexports(self, tracker: "ImplicitReexportTracker") -> None:
        """Override this to set some names as explicitly re-exported."""
        pass
