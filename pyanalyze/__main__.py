import sys
from os import system
from pyanalyze.name_check_visitor import NameCheckVisitor

# Enable ANSI color codes for Windows cmd using this strange workaround ( see https://github.com/python/cpython/issues/74261 )
system("")


def main() -> None:
    sys.exit(NameCheckVisitor.main())


if __name__ == "__main__":
    main()
