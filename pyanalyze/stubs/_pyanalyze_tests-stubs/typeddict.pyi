from typing_extensions import TypedDict

class TD1(TypedDict):
    a: int
    b: str

class TD2(TypedDict, total=False):
    a: int
    b: str
