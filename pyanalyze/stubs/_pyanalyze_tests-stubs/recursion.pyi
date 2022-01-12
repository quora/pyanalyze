from typing import ContextManager, AnyStr

class _ScandirIterator(ContextManager[_ScandirIterator[AnyStr]]):
    def close(self) -> None: ...
