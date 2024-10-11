from typing import AnyStr, ContextManager

from typing_extensions import TypeAlias

class _ScandirIterator(ContextManager[_ScandirIterator[AnyStr]]):
    def close(self) -> None: ...

StrJson: TypeAlias = str | dict[str, StrJson]
