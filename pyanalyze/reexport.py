"""

Functionality for dealing with implicit reexports.

"""
from ast import AST
from collections import defaultdict
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .node_visitor import Failure
from .config import Config
from .error_code import ErrorCode


class ErrorContext:
    all_failures: List[Failure]

    def show_error(
        self, node: AST, message: str, error_code: Enum
    ) -> Optional[Failure]:
        raise NotImplementedError


@dataclass
class ImplicitReexportTracker:
    config: InitVar[Config]
    completed_modules: Set[str] = field(default_factory=set)
    module_to_reexports: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    used_reexports: Dict[str, List[Tuple[str, AST, ErrorContext]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self, config: Config) -> None:
        config.configure_reexports(self)

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
