# static analysis: ignore
import sys
from pathlib import Path

import pytest

from .importer import load_module_from_file


def test_module_imported_by_its_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "importer_test_package"
    package.mkdir()
    (package / "__init__.py").write_text("from . import child\n")
    (package / "child.py").write_text(
        "with open(__file__ + '.log', 'a') as f:\n    f.write('x')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module, is_compiled = load_module_from_file(str(package / "child.py"))
        assert module is sys.modules["importer_test_package.child"]
        assert not is_compiled
        # The child module was executed only once, by its package
        assert (package / "child.py.log").read_text() == "x"
    finally:
        sys.modules.pop("importer_test_package.child", None)
        sys.modules.pop("importer_test_package", None)
