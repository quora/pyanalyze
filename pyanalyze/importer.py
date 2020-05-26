from __future__ import absolute_import, division, print_function, unicode_literals

"""

Module responsible for importing files.

"""

import logging
import nose.config
import nose.importer
import os.path
import sys

_importer = nose.importer.Importer(config=nose.config.Config(addPaths=False))

# don't print useless stuff
nose.importer.log.setLevel(logging.INFO)


def load_module_from_file(filename, excluded_paths=frozenset()):
    # Attempt to get the location of the module relative to sys.path so we can import it
    # somewhat properly
    abspath = os.path.abspath(filename)
    longest_match = -1
    module_path = None
    for path in sys.path:
        if not path.endswith("/"):
            path += "/"
        # hack: when you run test_scope in a/, a/ is part of the path, but some submodules of a/
        # don't react kindly to being imported as global modules instead of submodules of a/,
        # so exclude the directory
        if (
            path not in excluded_paths
            and abspath.startswith(path)
            and len(path) > longest_match
        ):
            new_module_path = abspath[len(path) :]
            # hack: some directories that are on the path are also themselves importable
            # packages (e.g. a/standalone)
            if new_module_path == "__init__.py":
                continue
            else:
                module_path = new_module_path
            longest_match = len(path)

    if module_path is None:
        return None, False

    parts = os.path.splitext(module_path)[0].split(os.sep)
    # the importable module path does not include __init__
    if parts[-1] == "__init__":
        parts = parts[:-1]
    module_path = ".".join(parts).lstrip(".")
    try:
        module = _importer.importFromPath(abspath, module_path)
    except ImportError as e:
        if is_ignorable_importerror(e):
            return None, True
        else:
            raise
    else:
        return module, module.__file__.endswith(".so")


def is_ignorable_importerror(e):
    """Returns whether the given exception indicates that the module can be ignored.

    This is used to ignore files compiled for the wrong Python version or platform.

    """
    if not isinstance(e, ImportError):
        return False
    message = e.args[0]

    # Python 3 pyanalyze trying to import a Python 2 module
    if message.endswith("undefined symbol: _Py_ZeroStruct"):
        return True
    return False
