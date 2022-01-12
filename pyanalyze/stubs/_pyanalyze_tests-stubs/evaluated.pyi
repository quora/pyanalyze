from pyanalyze.extensions import evaluated
from typing import Any, TextIO, BinaryIO, IO

@evaluated
def open(mode: str):
    if mode == "r":
        return TextIO
    elif mode == "rb":
        return BinaryIO
    else:
        return IO[Any]

@evaluated
def open2(mode: str) -> IO[Any]:
    if mode == "r":
        return TextIO
    elif mode == "rb":
        return BinaryIO
