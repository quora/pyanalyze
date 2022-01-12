"""

A tool for walking over stubs and checking that pyanalyze
can handle them.

"""
import ast
from ast_decompiler import decompile
from dataclasses import dataclass
import enum
from pathlib import Path
import textwrap
from pyanalyze.signature import ConcreteSignature
import typeshed_client
from typing import Collection, Container, Iterable, Optional, Sequence, Union

from .config import Config
from .value import AnySource, AnyValue, TypedValue, Value
from .checker import Checker
from .name_check_visitor import NameCheckVisitor

_checker = Checker(Config())
CTX = NameCheckVisitor("", "", ast.parse(""), checker=_checker)


class ErrorCode(enum.Enum):
    unresolved_import = 1
    unresolved_function = 2
    unresolved_object = 3
    signature_failed = 4
    unresolved_type_in_signature = 5
    unused_allowlist_entry = 6
    unresolved_bases = 7
    unresolved_module = 8


DISABLED_BY_DEFAULT = {
    # False positives with imports that only exist on another OS
    ErrorCode.unresolved_import,
    # Happens with functions that only exist on another OS
    ErrorCode.unresolved_function,
}


def _try_decompile(node: ast.AST) -> str:
    try:
        return decompile(node)
    except Exception as e:
        return f"could not decompile {ast.dump(node)} due to {e}\n"


@dataclass
class Error:
    code: ErrorCode
    message: str
    fully_qualified_name: str
    node: Union[
        ast.AST, typeshed_client.OverloadedName, typeshed_client.ImportedName, None
    ] = None

    def display(self) -> str:
        heading = f"{self.fully_qualified_name}: {self.message} ({self.code.name})\n"
        if isinstance(self.node, ast.AST):
            decompiled = _try_decompile(self.node)
            heading += textwrap.indent(decompiled, "  ")
        elif isinstance(self.node, typeshed_client.OverloadedName):
            lines = [
                textwrap.indent(_try_decompile(node), "  ")
                for node in self.node.definitions
            ]
            heading += "".join(lines)
        elif isinstance(self.node, typeshed_client.ImportedName):
            heading += (
                f"  imported from: {'.'.join(self.node.module_name)} with name"
                f" {self.node.name}"
            )
        return heading


def stubwalk(
    typeshed_path: Optional[Path] = None,
    search_path: Sequence[Path] = (),
    allowlist: Collection[str] = (),
    disabled_codes: Container[ErrorCode] = DISABLED_BY_DEFAULT,
    verbose: bool = True,
) -> Sequence[Error]:
    search_context = CTX.arg_spec_cache.ts_finder.resolver.ctx
    if typeshed_path is not None:
        search_context = search_context._replace(typeshed=typeshed_path)
    search_context = search_context._replace(search_path=search_path)
    final_errors = []
    used_allowlist_entries = set()
    for error in _stubwalk(search_context):
        if verbose:
            print(error.display(), end="")
        if error.code in disabled_codes:
            continue
        if error.fully_qualified_name in allowlist:
            used_allowlist_entries.add(error.fully_qualified_name)
            continue
        final_errors.append(error)
    if ErrorCode.unused_allowlist_entry not in disabled_codes:
        for unused_allowlist in set(allowlist) - used_allowlist_entries:
            final_errors.append(
                Error(
                    ErrorCode.unused_allowlist_entry,
                    "Unused allowlist entry",
                    unused_allowlist,
                )
            )
    return final_errors


def _stubwalk(search_context: typeshed_client.SearchContext) -> Iterable[Error]:
    finder = CTX.arg_spec_cache.ts_finder
    resolver = finder.resolver
    for module_name, _ in sorted(typeshed_client.get_all_stub_files(search_context)):
        if module_name in ("this", "antigravity"):
            continue  # please stop opening my browser
        names = typeshed_client.get_stub_names(module_name, search_context=resolver.ctx)
        if names is None:
            yield Error(
                ErrorCode.unresolved_module,
                f"Failed to find stub for module {module_name}",
                module_name,
            )
            continue
        for name, info in names.items():
            is_function = isinstance(
                info.ast,
                (ast.FunctionDef, ast.AsyncFunctionDef, typeshed_client.OverloadedName),
            )
            fq_name = f"{module_name}.{name}"
            if is_function:
                sig = finder.get_argspec_for_fully_qualified_name(fq_name, None)
                if sig is None:
                    yield Error(
                        ErrorCode.signature_failed,
                        "Cannot get signature for function",
                        fq_name,
                        info.ast,
                    )
                else:
                    yield from _error_on_nested_any(sig, "Signature", fq_name, info)
            if isinstance(info.ast, ast.ClassDef):
                bases = finder.get_bases_for_fq_name(fq_name)
                if bases is None:
                    yield Error(
                        ErrorCode.unresolved_bases,
                        "Cannot resolve bases",
                        fq_name,
                        info.ast,
                    )
                else:
                    for base in bases:
                        if not isinstance(base, TypedValue):
                            yield Error(
                                ErrorCode.unresolved_bases,
                                "Cannot resolve one of the bases",
                                fq_name,
                                info.ast,
                            )
                        else:
                            yield from _error_on_nested_any(base, "Base", fq_name, info)
                # TODO:
                # - Loop over all attributes and assert their values don't contain Any
                # - Loop over all methods and check their signatures
            val = finder.resolve_name(module_name, name)
            if val == AnyValue(AnySource.inference):
                if is_function:
                    yield Error(
                        ErrorCode.unresolved_function,
                        "Cannot resolve function",
                        fq_name,
                        info.ast,
                    )
                elif isinstance(info.ast, typeshed_client.ImportedName):
                    yield Error(
                        ErrorCode.unresolved_import,
                        "Cannot resolve imported name",
                        fq_name,
                        info.ast,
                    )
                else:
                    yield Error(
                        ErrorCode.unresolved_object,
                        "Cannot resolve name",
                        fq_name,
                        info.ast,
                    )


def _error_on_nested_any(
    sig_or_val: Union[ConcreteSignature, Value],
    label: str,
    fq_name: str,
    info: typeshed_client.NameInfo,
) -> Iterable[Error]:
    for val in sig_or_val.walk_values():
        if val == AnyValue(AnySource.inference):
            yield Error(
                ErrorCode.unresolved_type_in_signature,
                f"{label} {sig_or_val} contains unresolved type",
                fq_name,
                info.ast,
            )
