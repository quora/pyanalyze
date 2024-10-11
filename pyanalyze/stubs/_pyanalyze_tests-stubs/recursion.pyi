from contextlib import AbstractContextManager
from typing import AnyStr

from typing_extensions import TypeAlias

class _ScandirIterator(AbstractContextManager[_ScandirIterator[AnyStr]]):
    def close(self) -> None: ...

StrJson: TypeAlias = str | dict[str, StrJson]
