from typing import AnyStr, ContextManager

class _ScandirIterator(ContextManager[_ScandirIterator[AnyStr]]):
    def close(self) -> None: ...
