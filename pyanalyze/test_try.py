# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before


class TestExoticTry(TestNameCheckVisitorBase):
    @assert_passes()
    def test_except_everything(self):
        from typing import Tuple, Type, Union

        from typing_extensions import Literal, assert_type

        def capybara(
            typ: Literal[TypeError, ValueError],
            typ2: Union[Tuple[Literal[RuntimeError], ...], Literal[KeyError]],
            typ3: Union[Type[RuntimeError], Type[KeyError]],
            typ4: Union[Tuple[Type[RuntimeError], ...], Type[KeyError]],
            cond: bool,
        ):
            try:
                pass
            except typ as e1:
                assert_type(e1, Union[TypeError, ValueError])
            except typ2 as e2:
                assert_type(e2, Union[RuntimeError, KeyError])
            except typ3 as e3:
                assert_type(e3, Union[RuntimeError, KeyError])
            except typ4 as e4:
                assert_type(e4, Union[RuntimeError, KeyError])
            except FileNotFoundError if cond else FileExistsError as e5:
                assert_type(e5, Union[FileNotFoundError, FileExistsError])
            except (KeyError, (ValueError, (TypeError, RuntimeError))) as e6:
                assert_type(e6, Union[KeyError, ValueError, TypeError, RuntimeError])
            except GeneratorExit as e7:
                assert_type(e7, GeneratorExit)


class TestTryStar(TestNameCheckVisitorBase):
    @skip_before((3, 11))
    def test_eg_types(self):
        self.assert_passes(
            """
            from typing import assert_type

            def capybara():
                try:
                    pass
                except* ValueError as eg:
                    assert_type(eg, ExceptionGroup[ValueError])
                except* KeyboardInterrupt as eg:
                    assert_type(eg, BaseExceptionGroup[KeyboardInterrupt])
                except* (OSError, (RuntimeError, KeyError)) as eg:
                    assert_type(eg, ExceptionGroup[OSError | RuntimeError | KeyError])
                except *ExceptionGroup as eg:  # E: bad_except_handler
                    pass
                except *int as eg:  # E: bad_except_handler
                    pass
            """
        )

    @skip_before((3, 11))
    def test_variable_scope(self):
        self.assert_passes(
            """
            from typing import assert_type, Literal

            def capybara():
                x = 0
                try:
                    x = 1
                    assert_type(x, Literal[1])
                except* ValueError as eg:
                    assert_type(x, Literal[0, 1])
                    x = 2
                except* TypeError as eg:
                    assert_type(x, Literal[0, 1, 2])
                    x = 3
                assert_type(x, Literal[1, 2, 3])
            """
        )

    @skip_before((3, 11))
    def test_try_else(self):
        self.assert_passes(
            """
            from typing import assert_type, Literal

            def capybara():
                x = 0
                try:
                    x = 1
                    assert_type(x, Literal[1])
                except* ValueError as eg:
                    assert_type(x, Literal[0, 1])
                    x = 2
                except* TypeError as eg:
                    assert_type(x, Literal[0, 1, 2])
                    x = 3
                else:
                    assert_type(x, Literal[1])
                    x = 4
                assert_type(x, Literal[2, 3, 4])
            """
        )

    @skip_before((3, 11))
    def test_try_finally(self):
        self.assert_passes(
            """
            from typing import assert_type, Literal

            def capybara():
                x = 0
                try:
                    x = 1
                    assert_type(x, Literal[1])
                except* ValueError as eg:
                    assert_type(x, Literal[0, 1])
                    x = 2
                except* TypeError as eg:
                    assert_type(x, Literal[0, 1, 2])
                    x = 3
                finally:
                    assert_type(x, Literal[0, 1, 2, 3])
                    x = 4
                assert_type(x, Literal[4])
            """
        )
