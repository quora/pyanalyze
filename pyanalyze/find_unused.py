from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

"""

Implementation of unused object detection.

"""

import asynq
import collections
import inspect
import six


class UnusedObjectFinder(object):
    """Context to find unused objects.

    This records all accesses for Python functions and classes and prints out all existing
    objects that are completely unused.

    """

    def __init__(self, config, enabled=False, print_output=True, include_modules=False):
        self.config = config
        self.enabled = enabled
        self.include_modules = include_modules
        self.print_output = print_output
        self.usages = collections.Counter()

    def __enter__(self):
        if self.enabled:
            return self
        else:
            return None

    def __exit__(self, exc_typ, exc_val, exc_tb):
        if not self.enabled or not self.print_output:
            return

        def sort_fn(obj):
            if hasattr(obj, "__module__"):
                return (obj.__module__, obj.__name__)
            else:
                return obj.__name__

        for obj in sorted(self.get_unused_objects(), key=sort_fn):
            obj = self.config.unwrap_cls(obj)
            if obj.__name__.startswith(("Test", "test")):
                continue
            if hasattr(obj, "__module__"):
                print("%s.%s: %s" % (obj.__module__, obj.__name__, obj))
            else:
                print("%s: %s" % (obj.__name__, obj))

    def get_unused_objects(self):
        if self.config.DEFAULT_BASE_MODULE is None:
            return set()
        existing_objects = self._get_recordable_objects_in_module(
            self.config.DEFAULT_BASE_MODULE, include_modules=self.include_modules
        )
        return {obj for obj in existing_objects if self.usages[obj] == 0}

    def record(self, value):
        if self.enabled and self._should_record_usage(
            value, include_modules=self.include_modules
        ):
            self.usages[value] += 1

    def _get_recordable_objects_in_module(self, module, include_modules=False):
        ret = set()
        for value in six.itervalues(module.__dict__):
            if self._is_internal_to_module(value, module):
                # recurse into submodules
                if inspect.ismodule(value):
                    ret |= self._get_recordable_objects_in_module(
                        value, include_modules=include_modules
                    )

                # record recordable objects
                if self._should_record_usage(value, include_modules=include_modules):
                    ret.add(value)
        return ret

    def _is_internal_to_module(self, value, enclosing_module):
        if inspect.ismodule(value):
            return value.__name__.startswith(enclosing_module.__name__ + ".")
        elif hasattr(value, "__module__"):
            return value.__module__ == enclosing_module.__name__
        else:
            # can't tell, so don't bother
            return False

    def _should_record_usage(self, value, include_modules=False):
        # also include async functions, but don't call is_async_fn on modules because it can fail
        if inspect.isfunction(value) or (
            not inspect.ismodule(value) and asynq.is_async_fn(value)
        ):
            registered = self.config.registered_values()
            try:
                return value not in registered
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
            return value not in self.config.registered_values()
        elif include_modules and inspect.ismodule(value):
            return True
        else:
            return False
