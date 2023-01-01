from pyanalyze.extensions import deprecated
from typing import overload

@overload
@deprecated("int support is deprecated")
def deprecated_overload(x: int) -> int: ...
@overload
def deprecated_overload(x: str) -> str: ...
@deprecated("no functioning capybaras")
def deprecated_function(x: int) -> int: ...

class Capybara:
    @deprecated("no methodical capybaras")
    def deprecated_method(self, x: int) -> int: ...

@deprecated("no classy capybaras")
class DeprecatedCapybara: ...
