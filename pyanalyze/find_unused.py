from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

"""

Implementation of unused object detection.

"""

import asynq
from collections import defaultdict
import enum
import inspect
import qcore
import __future__


_used_objects = set()


def used(obj):
    """Decorator indicating that an object is being used.

    This stops the UnusedObjectFinder from marking it as unused.

    """
    _used_objects.add(obj)
    return obj


# so it doesn't itself get marked as unused
used(used)


class _UsageKind(enum.IntEnum):
    unused = 1
    used_in_test = 2
    used = 3

    @classmethod
    def classify(cls, module_name):
        if "." not in module_name:
            return cls.used
        own_name = module_name.rsplit(".", maxsplit=1)[1]
        if own_name.startswith("test"):
            return cls.used_in_test
        else:
            return cls.used

    @classmethod
    def aggregate(cls, usages):
        return max(usages, default=cls.unused)

    @classmethod
    def aggregate_modules(cls, module_names):
        return cls.aggregate(cls.classify(module_name) for module_name in module_names)


class UnusedObjectFinder(object):
    """Context to find unused objects.

    This records all accesses for Python functions and classes and prints out all existing
    objects that are completely unused.

    """

    def __init__(
        self,
        config,
        enabled=False,
        print_output=True,
        include_modules=False,
        print_all=False,
    ):
        self.config = config
        self.enabled = enabled
        self.include_modules = include_modules
        self.print_output = print_output
        self.print_all = print_all
        self.usages = defaultdict(lambda: defaultdict(set))
        self.import_stars = defaultdict(set)
        self.module_to_import_stars = defaultdict(set)
        self.visited_modules = []
        self._recursive_stack = set()

    def __enter__(self):
        if self.enabled:
            return self
        else:
            return None

    def __exit__(self, exc_typ, exc_val, exc_tb):
        if not self.enabled or not self.print_output:
            return

        for module in sorted(self.visited_modules, key=lambda mod: mod.__name__):
            self._print_unused_from_module(module)

    def record(self, owner, attr, using_module):
        if not self.enabled:
            return
        try:
            self.usages[owner][attr].add(using_module)
        except Exception:
            pass

    def record_import_star(self, module, using_module):
        self.import_stars[module].add(using_module)
        self.module_to_import_stars[using_module].add(module)

    def record_module_visited(self, module):
        self.visited_modules.append(module)

    def _print_unused_from_module(self, module):
        is_test_module = module.__name__.split(".")[-1].startswith("test")
        for attr, value in module.__dict__.items():
            usages = self.usages[module][attr]
            if self.print_all:
                print("%s.%s: %d (%s)" % (module.__name__, attr, len(usages), usages))
                continue
            # Ignore attributes injected by Python
            if attr.startswith("__") and attr.endswith("__"):
                continue
            # Ignore tests
            if is_test_module and attr.startswith(("test", "Test")):
                continue
            own_usage = _UsageKind.aggregate_modules(usages)
            star_usage = self._has_import_star_usage(module, attr)
            usage = _UsageKind.aggregate([own_usage, star_usage])
            if usage is _UsageKind.used:
                continue
            if not self._should_record_as_unused(module, attr, value):
                continue
            if any(
                hasattr(import_starred, attr)
                for import_starred in self.module_to_import_stars[module]
            ):
                continue
            if usage is _UsageKind.used_in_test:
                if not is_test_module:
                    print("%s.%s: used only in tests" % (module.__name__, attr))
            else:
                print("%s.%s: unused" % (module.__name__, attr))

    def _has_import_star_usage(self, module, attr):
        with qcore.override(self, "_recursive_stack", set()):
            return self._has_import_star_usage_inner(module, attr)

    def _has_import_star_usage_inner(self, module, attr):
        if module in self._recursive_stack:
            return _UsageKind.unused
        self._recursive_stack.add(module)
        usage = _UsageKind.aggregate_modules(self.usages[module][attr])
        if usage is _UsageKind.used:
            return _UsageKind.used
        import_stars = self.import_stars[module]
        recursive_usage = _UsageKind.aggregate(
            self._has_import_star_usage_inner(importing_module, attr)
            for importing_module in import_stars
        )
        return _UsageKind.aggregate([usage, recursive_usage])

    def _should_record_as_unused(self, module, attr, value):
        if self.config.should_ignore_unused(module, attr, value):
            return False
        # TODO: remove most of the below and just rely on @used and
        # should_ignore_unused()
        # also include async functions, but don't call is_async_fn on modules because it can fail
        if inspect.isfunction(value) or (
            not inspect.ismodule(value) and asynq.is_async_fn(value)
        ):
            registered = self.config.registered_values()
            try:
                if value in registered:
                    return False
            except TypeError:
                return False  # mock.call can get here
        elif inspect.isclass(value):
            # can't reliably detect usage of classes with a metaclass
            metaclass = type(value)
            try:
                is_allowed = metaclass in self.config.ALLOWED_METACLASSES
            except TypeError:
                # apparently mock objects have a dictionary as their metaclass
                is_allowed = False
            if not is_allowed:
                return False
            # controllers are called directly by Pylons
            if any(issubclass(value, cls) for cls in self.config.USED_BASE_CLASSES):
                return False
            if value in self.config.registered_values():
                return False
        elif inspect.ismodule(value):
            # test modules will usually show up as unused
            if value.__name__.split(".")[-1].startswith("test"):
                return False
        try:
            # __future__ imports are usually unused
            return value not in _used_objects and not isinstance(
                value, __future__._Feature
            )
        except Exception:
            return True
