# Copyright 2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
from mock import MagicMock
from pytest_mock import MockerFixture
from unit.util import powerset

import cupynumeric._array.util as m  # module under test


@m.add_boilerplate()
def _out_implicit(a, b, out):
    pass


@m.add_boilerplate("out")
def _out_explicit(a, b, out):
    pass


@m.add_boilerplate()
def _where_implicit(a, b, where):
    pass


@m.add_boilerplate("where")
def _where_explicit(a, b, where):
    pass


@pytest.fixture(autouse=True)
def mock_convert(mocker: MockerFixture) -> MagicMock:
    return mocker.patch(
        "cupynumeric._array.util.convert_to_cupynumeric_ndarray"
    )


class Test_add_boilerplate_bad:
    def test_bad_repeat(self) -> None:
        with pytest.raises(AssertionError):

            @m.add_boilerplate("a", "a")
            def _bad_repeat(a, b):
                pass

    def test_bad_extra(self) -> None:
        with pytest.raises(AssertionError):

            @m.add_boilerplate("c")
            def _bad_repeat(a, b):
                pass


class Test_add_boilerplate_args:
    @pytest.mark.parametrize("args", powerset("abc"))
    def test_args_positional_None(self, args, mock_convert: MagicMock) -> None:
        @m.add_boilerplate(*args)
        def func(a, b, c):
            pass

        func(None, None, None)

        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("abc"))
    def test_args_positional_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        @m.add_boilerplate(*args)
        def func(a, b, c):
            pass

        vals = (1, 2, 3)

        func(*vals)

        assert mock_convert.call_count == len(args)
        expected = (
            val for (arg, val) in zip(tuple("abc"), vals) if arg in args
        )
        for item in expected:
            mock_convert.assert_any_call(item)

    @pytest.mark.parametrize("args", powerset("abc"))
    def test_args_kwargs_None(self, args, mock_convert: MagicMock) -> None:
        @m.add_boilerplate(*args)
        def func(a, b, c):
            pass

        func(a=None, b=None, c=None)

        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("abc"))
    def test_args_kwargs_value(self, args, mock_convert: MagicMock) -> None:
        @m.add_boilerplate(*args)
        def func(a, b, c):
            pass

        vals = (1, 2, 3)

        func(**dict(zip(tuple("abc"), vals)))

        assert mock_convert.call_count == len(args)
        expected = (
            val for (arg, val) in zip(tuple("abc"), vals) if arg in args
        )
        for item in expected:
            mock_convert.assert_any_call(item)


class Test_add_boilerplate_out:
    def test_implicit_positional_None(self, mock_convert: MagicMock) -> None:
        _out_implicit(None, None, None)
        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("ab"))
    def test_implicit_positional_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        _out_implicit(None, None, 10)
        mock_convert.assert_called_once_with(10, share=True)

    def test_implicit_kwargs_None(self, mock_convert: MagicMock) -> None:
        _out_implicit(None, None, out=None)
        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("ab"))
    def test_implicit_kwargs_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        _out_implicit(None, None, out=10)
        mock_convert.assert_called_once_with(10, share=True)

    def test_explicit_positional_None(self, mock_convert: MagicMock) -> None:
        _out_explicit(None, None, None)
        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("ab"))
    def test_explicit_positional_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        _out_explicit(None, None, 10)
        mock_convert.assert_called_once_with(10, share=True)

    def test_explicit_kwargs_None(self, mock_convert: MagicMock) -> None:
        _out_explicit(None, None, out=None)
        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("ab"))
    def test_explicit_kwargs_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        _out_explicit(None, None, out=10)
        mock_convert.assert_called_once_with(10, share=True)


class Test_add_boilerplate_where:
    def test_implicit_positional_None(self, mock_convert: MagicMock) -> None:
        _where_implicit(None, None, None)
        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("ab"))
    def test_implicit_positional_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        _where_implicit(None, None, 10)
        mock_convert.assert_called_once_with(10)

    def test_implicit_kwargs_None(self, mock_convert: MagicMock) -> None:
        _where_implicit(None, None, where=None)
        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("ab"))
    def test_implicit_kwargs_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        _where_implicit(None, None, where=10)
        mock_convert.assert_called_once_with(10)

    def test_explicit_positional_None(self, mock_convert: MagicMock) -> None:
        _where_explicit(None, None, None)
        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("ab"))
    def test_explicit_positional_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        _where_explicit(None, None, 10)
        mock_convert.assert_called_once_with(10)

    def test_explicit_kwargs_None(self, mock_convert: MagicMock) -> None:
        _where_explicit(None, None, where=None)
        assert not mock_convert.called

    @pytest.mark.parametrize("args", powerset("ab"))
    def test_explicit_kwargs_value(
        self, args, mock_convert: MagicMock
    ) -> None:
        _where_explicit(None, None, where=10)
        mock_convert.assert_called_once_with(10)


def test_add_boilerplate_mixed(mock_convert: MagicMock) -> None:
    @m.add_boilerplate(
        "a",
        "b",
        "c",
    )
    def func(a, b=2, c=None, d=None, e=5, out=None, where=None):
        pass

    func(1, c=3, out=4, where=None)

    assert mock_convert.call_count == 3
    mock_convert.assert_any_call(1)
    mock_convert.assert_any_call(3)
    mock_convert.assert_any_call(4, share=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
