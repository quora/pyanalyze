# Has to exist for stubs to be able to import
# from it, because typeshed_client doesn't let
# stubs import from non-stub files.

from typing import Any, Callable, Optional, List

def reveal_type(value: object) -> None: ...
def get_overloads(fully_qualified_name: str) -> List[Callable[..., Any]]: ...
def get_type_evaluation(fully_qualified_name: str) -> Optional[Callable[..., Any]]: ...
def overload(func: Callable[..., Any]) -> Callable[..., Any]: ...
def evaluated(func: Callable[..., Any]) -> Callable[..., Any]: ...
def is_provided(arg: Any) -> bool: ...
def is_positional(arg: Any) -> bool: ...
def is_keyword(arg: Any) -> bool: ...
def is_of_type(arg: Any, type: Any, *, exclude_any: bool = ...) -> bool: ...
def show_error(message: str, *, argument: Optional[Any] = ...) -> bool: ...
def __getattr__(self, __arg: str) -> Any: ...
