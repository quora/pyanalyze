import sys
import os

from pyanalyze.name_check_visitor import NameCheckVisitor


def main() -> None:
    if os.name == "nt":
        # Enable ANSI color codes for Windows cmd using this strange workaround
        # ( see https://github.com/python/cpython/issues/74261 )
        os.system("")
    sys.exit(NameCheckVisitor.main())


if __name__ == "__main__":
    main()
