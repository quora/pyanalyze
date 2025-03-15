import sys
from os import system
system("") # Enable ANSI color codes for Windows cmd using this strange workaround ( see https://github.com/python/cpython/issues/74261 )

from pyanalyze.name_check_visitor import NameCheckVisitor


def main() -> None:
    sys.exit(NameCheckVisitor.main())


if __name__ == "__main__":
    main()
