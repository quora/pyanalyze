from .stubwalk import stubwalk


def test_stubwalk() -> None:
    errors = stubwalk(allowlist={"typing._promote"})
    if errors:
        message = "".join(error.display() for error in errors)
        raise AssertionError(message)
