from typing import Any, BinaryIO, IO, TextIO

from pyanalyze.extensions import evaluated

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
