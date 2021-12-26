"""

Functionality for dealing with implicit reexports.

"""
from ast import AST
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pyanalyze.options import Options, PyObjectSequenceOption

from .node_visitor import Failure
from .config import Config
from .error_code import ErrorCode

_ReexportConfigProvider = Callable[["ImplicitReexportTracker"], None]


class ReexportConfig(PyObjectSequenceOption[_ReexportConfigProvider]):
    """Callbacks that can configure the :class:`ImplicitReexportTracker`,
    usually by setting names as explicitly exported."""

    name = "reexport_config"

    @classmethod
    def get_value_from_fallback(
        cls, fallback: Config
    ) -> Sequence[_ReexportConfigProvider]:
        return (fallback.configure_reexports,)


class ErrorContext:
    all_failures: List[Failure]

    def show_error(
        self,
        node: AST,
        message: str,
        error_code: Enum,
        *,
        detail: Optional[str] = None,
        save: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Failure]:
        raise NotImplementedError


@dataclass
class ImplicitReexportTracker:
    options: InitVar[Options]
    completed_modules: Set[str] = field(default_factory=set)
    module_to_reexports: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    used_reexports: Dict[str, List[Tuple[str, AST, ErrorContext]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self, options: Options) -> None:
        for func in options.get_value_for(ReexportConfig):
            func(self)

    def record_exported_attribute(self, module: str, attr: str) -> None:
        self.module_to_reexports[module].add(attr)

    def record_module_completed(self, module: str) -> None:
        self.completed_modules.add(module)
        reexports = self.module_to_reexports[module]
        for attr, node, ctx in self.used_reexports[module]:
            if attr not in reexports:
                self.show_error(module, attr, node, ctx)

    def record_attribute_accessed(
        self, module: str, attr: str, node: AST, ctx: ErrorContext
    ) -> None:
        if module in self.completed_modules:
            if attr not in self.module_to_reexports[module]:
                self.show_error(module, attr, node, ctx)
        else:
            self.used_reexports[module].append((attr, node, ctx))

    def show_error(self, module: str, attr: str, node: AST, ctx: ErrorContext) -> None:
        failure = ctx.show_error(
            node,
            f"Attribute '{attr}' is not exported by module '{module}'",
            ErrorCode.implicit_reexport,
        )
        if failure is not None:
            ctx.all_failures.append(failure)
