from typing import Any

class StubCallable:
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...
