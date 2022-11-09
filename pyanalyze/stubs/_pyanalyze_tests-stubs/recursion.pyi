from typing import AnyStr, ContextManager, Dict, Union
from typing_extensions import TypeAlias

class _ScandirIterator(ContextManager[_ScandirIterator[AnyStr]]):
    def close(self) -> None: ...

StrJson: TypeAlias = Union[str, Dict[str, StrJson]]
