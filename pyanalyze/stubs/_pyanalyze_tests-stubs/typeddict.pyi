from typing_extensions import NotRequired, Required, TypedDict

class TD1(TypedDict):
    a: int
    b: str

class TD2(TypedDict, total=False):
    a: int
    b: str

class PEP655(TypedDict):
    a: NotRequired[int]
    b: Required[str]

class Inherited(TD1):
    c: float
