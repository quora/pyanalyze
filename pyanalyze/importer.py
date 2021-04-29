"""

Module responsible for importing files.

"""

from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from typing import Optional, cast, Tuple
from types import ModuleType


@lru_cache()
def directory_has_init(path: Path) -> bool:
    return (path / "__init__.py").exists()


def load_module_from_file(
    filename: str, verbose: bool = False
) -> Tuple[Optional[ModuleType], bool]:
    # Attempt to get the location of the module relative to sys.path so we can import it
    # somewhat properly
    abspath = Path(filename).resolve()
    candidate_paths = []
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
        if parts[-1] == "__init__":
            parts = parts[:-1]

        candidate_paths.append((import_path, ".".join(parts)))

    # First attempt to import only through paths that have __init__.py at every level
    # to avoid importing through unnecessary namespace packages.
    for restrict_init in (True, False):
        for import_path, module_path in candidate_paths:  # use sys.path order
            if module_path in sys.modules:
                existing = cast(ModuleType, sys.modules[module_path])
                is_compiled = getattr(existing, "__file__", None) != str(abspath)
                if verbose:
                    print(
                        f"found {abspath} already present as {module_path}"
                        f" (is_compiled: {is_compiled})"
                    )
                return existing, is_compiled
            if restrict_init:
                missing_init = False
                for parent in abspath.parents:
                    if parent == import_path:
                        break
                    if not directory_has_init(parent):
                        missing_init = True
                        break
                if missing_init:
                    if verbose:
                        print(f"skipping {import_path} because of missing __init__.py")
                    continue

            if verbose:
                print(f"importing {abspath} as {module_path}")

            if "." in module_path:
                parent_module_path, child_name = module_path.rsplit(".", maxsplit=1)
                try:
                    parent_module = importlib.import_module(parent_module_path)
                except ImportError:
                    continue
            else:
                parent_module = child_name = None

            spec = importlib.util.spec_from_file_location(module_path, abspath)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            spec.loader.exec_module(module)
            if parent_module is not None and child_name is not None:
                setattr(parent_module, child_name, module)
            return module, False

    return None, False
