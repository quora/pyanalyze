"""

Module responsible for importing files.

"""

import importlib.util
from pathlib import Path
import sys
from typing import Optional, cast, Tuple
from types import ModuleType


def load_module_from_file(
    filename: str, verbose: bool = True
) -> Tuple[Optional[ModuleType], bool]:
    # Attempt to get the location of the module relative to sys.path so we can import it
    # somewhat properly
    abspath = Path(filename).resolve()
    module_path = None
    for sys_path_entry in sys.path:
        if not sys_path_entry:
            continue
        import_path = Path(sys_path_entry)
        try:
            relative_path = abspath.relative_to(import_path)
        except ValueError:
            continue

        parts = [*relative_path.parts[:-1], relative_path.stem]
        if not all(part.isidentifier() for part in parts):
            continue

        new_module_path = ".".join(parts)
        if module_path is None or len(new_module_path) > len(module_path):
            module_path = new_module_path

    if module_path is None:
        return None, False

    if module_path in sys.modules:
        existing = cast(ModuleType, sys.modules[module_path])
        is_compiled = getattr(existing, "__file__", None) != str(abspath)
        if verbose:
            print(
                f"found {abspath} already present as {module_path} (is_compiled: {is_compiled})"
            )
        return existing, is_compiled
    if verbose:
        print(f"importing {abspath} as {module_path}")
    spec = importlib.util.spec_from_file_location(module_path, abspath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    return module, False
