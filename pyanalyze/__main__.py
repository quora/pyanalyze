import sys

from pyanalyze.name_check_visitor import NameCheckVisitor


def main() -> None:
    sys.exit(NameCheckVisitor.main())


if __name__ == "__main__":
    main()
