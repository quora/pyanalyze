from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

"""

Implementation of unused object detection.

"""

import attr
from collections import defaultdict
import enum
import inspect
import qcore
from types import ModuleType
from typing import Any
import __future__

from .analysis_lib import safe_in


_used_objects = set()
_test_helper_objects = set()


def used(obj):
    """Decorator indicating that an object is being used.

    This stops the UnusedObjectFinder from marking it as unused.

    """
    _used_objects.add(obj)
    return obj


def test_helper(obj):
    """Decorator indicating that an object is intended as a helper for tests.

    If the object is used only in tests, this stops the UnusedObjectFinder from
    marking it as unused.

    """
    _test_helper_objects.add(obj)
    return obj


# so it doesn't itself get marked as unused
used(used)
used(test_helper)


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


@attr.s
class UnusedObject(object):
    module = attr.ib(type=ModuleType)
    attribute = attr.ib(type=str)
    value = attr.ib(type=Any)
    message = attr.ib(type=str)

    def __str__(self):
        return "%s.%s: %s" % (self.module.__name__, self.attribute, self.message)


class UnusedObjectFinder(object):
    """Context to find unused objects.

    This records all accesses for Python functions and classes and prints out all existing
    objects that are completely unused.

    """

    def __init__(self, config, enabled=False, print_output=True, print_all=False):
        self.config = config
        self.enabled = enabled
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

        for unused_object in self.get_unused_objects():
            print(unused_object)

    def record(self, owner, attr, using_module):
        if not self.enabled:
            return
        try:
            self.usages[owner][attr].add(using_module)
        except Exception:
            pass

    def record_import_star(self, imported_module, importing_module):
        self.import_stars[imported_module].add(importing_module)
        self.module_to_import_stars[importing_module].add(imported_module)

    def record_module_visited(self, module):
        self.visited_modules.append(module)

    def get_unused_objects(self):
        for module in sorted(self.visited_modules, key=lambda mod: mod.__name__):
            for obj in self._get_unused_from_module(module):
                yield obj

    def _get_unused_from_module(self, module):
        is_test_module = any(
            part.startswith("test") for part in module.__name__.split(".")
        )
        for attr, value in module.__dict__.items():
            usages = self.usages[module][attr]
            if self.print_all:
                message = "%d (%s)" % (len(usages), usages)
                yield UnusedObject(module, attr, value, message)
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
                if not is_test_module and not safe_in(value, _test_helper_objects):
                    yield UnusedObject(module, attr, value, "used only in tests")
            else:
                yield UnusedObject(module, attr, value, "unused")

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
        if inspect.ismodule(value):
            # test modules will usually show up as unused
            if value.__name__.split(".")[-1].startswith("test"):
                return False
            # if it was ever import *ed from, don't treat it as unused
            if value in self.import_stars:
                return False
        if safe_in(value, _used_objects):
            return False
        try:
            # __future__ imports are usually unused
            return not isinstance(value, __future__._Feature)
        except Exception:
            return True
