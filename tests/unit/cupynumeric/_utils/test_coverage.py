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

from types import ModuleType
from typing import Any

import pytest
from mock import MagicMock, patch

import cupynumeric
import cupynumeric._utils.coverage as m  # module under test
from cupynumeric.settings import settings


def test_FALLBACK_WARNING() -> None:
    assert m.FALLBACK_WARNING.format(what="foo") == (
        "cuPyNumeric has not implemented foo "
        + "and is falling back to canonical NumPy. "
        + "You may notice significantly decreased performance "
        + "for this function call."
    )


def test_MOD_INTERNAL() -> None:
    assert m.MOD_INTERNAL == {"__dir__", "__getattr__"}


class Test_filter_namespace:
    def test_empty(self) -> None:
        orig = dict()
        ns = dict(orig)

        result = m.filter_namespace(ns)
        assert orig == ns
        assert result == ns
        assert result is not ns

        result = m.filter_namespace(ns, omit_names=("foo",))
        assert orig == ns
        assert result == ns
        assert result is not ns

        result = m.filter_namespace(ns, omit_types=(int, str))
        assert orig == ns
        assert result == ns
        assert result is not ns

    def test_no_filters(self) -> None:
        orig = dict(foo=10)
        ns = dict(orig)

        result = m.filter_namespace(ns)
        assert orig == ns
        assert result == ns
        assert result is not ns

    def test_name_filters(self) -> None:
        orig = dict(foo=10, bar=20)
        ns = dict(orig)

        result = m.filter_namespace(ns, omit_names=("foo",))
        assert orig == ns
        assert result == dict(bar=20)

        result = m.filter_namespace(ns, omit_names=("foo", "bar"))
        assert orig == ns
        assert result == dict()

        result = m.filter_namespace(ns, omit_names=("foo", "baz"))
        assert orig == ns
        assert result == dict(bar=20)

    def test_type_filters(self) -> None:
        orig = dict(foo=10, bar="abc")
        ns = dict(orig)

        result = m.filter_namespace(ns, omit_types=(int,))
        assert orig == ns
        assert result == dict(bar="abc")

        result = m.filter_namespace(ns, omit_types=(int, str))
        assert orig == ns
        assert result == dict()

        result = m.filter_namespace(ns, omit_types=(int, float))
        assert orig == ns
        assert result == dict(bar="abc")


def _test_func(a: int, b: int) -> int:
    """docstring"""
    return a + b


class _Test_ufunc(cupynumeric._ufunc.ufunc):
    """docstring"""

    def __init__(self):
        super().__init__("_test_ufunc", "docstring")

    def __call__(self, a: int, b: int) -> int:
        return a + b


_test_ufunc = _Test_ufunc()


class Test_helpers:
    def test_is_wrapped_true(self) -> None:
        wrapped = m.implemented(_test_func, "foo", "_test_func")
        assert m.is_wrapped(wrapped)

    def test_is_wrapped_false(self) -> None:
        assert not m.is_wrapped(10)

    def test_is_implemented_true(self) -> None:
        wrapped = m.implemented(_test_func, "foo", "_test_func")
        assert m.is_implemented(wrapped)

    def test_is_implemented_false(self) -> None:
        wrapped = m.unimplemented(_test_func, "foo", "_test_func")
        assert not m.is_implemented(wrapped)


class Test_implemented:
    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_True_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.implemented(_test_func, "foo", "_test_func")

        assert wrapped.__name__ == _test_func.__name__
        assert wrapped.__qualname__ == _test_func.__qualname__
        assert wrapped.__doc__ == _test_func.__doc__
        assert wrapped.__wrapped__ is _test_func

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_called_once()
        assert mock_record_api_call.call_args[0] == ()
        assert mock_record_api_call.call_args[1]["name"] == "foo._test_func"
        assert mock_record_api_call.call_args[1]["implemented"]
        filename, lineno = mock_record_api_call.call_args[1]["location"].split(
            ":"
        )
        assert int(lineno)

    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_False_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.implemented(
            _test_func, "foo", "_test_func", reporting=False
        )

        assert wrapped.__name__ == _test_func.__name__
        assert wrapped.__qualname__ == _test_func.__qualname__
        assert wrapped.__doc__ == _test_func.__doc__
        assert wrapped.__wrapped__ is _test_func

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_not_called()

    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_True_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.implemented(_test_ufunc, "foo", "_test_ufunc")

        # these had to be special-cased, @wraps does not handle them
        assert wrapped.__name__ == _test_ufunc._name
        assert wrapped.__qualname__ == _test_ufunc._name

        assert wrapped.__doc__ == _test_ufunc.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_called_once()
        assert mock_record_api_call.call_args[0] == ()
        assert mock_record_api_call.call_args[1]["name"] == "foo._test_ufunc"
        assert mock_record_api_call.call_args[1]["implemented"]
        filename, lineno = mock_record_api_call.call_args[1]["location"].split(
            ":"
        )
        assert int(lineno)

    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_False_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.implemented(
            _test_ufunc, "foo", "_test_func", reporting=False
        )

        # these had to be special-cased, @wraps does not handle them
        assert wrapped.__name__ == _test_ufunc._name
        assert wrapped.__qualname__ == _test_ufunc._name

        assert wrapped.__doc__ == _test_ufunc.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_not_called()


class Test_unimplemented:
    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_True_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.unimplemented(_test_func, "foo", "_test_func")

        assert wrapped.__name__ == _test_func.__name__
        assert wrapped.__qualname__ == _test_func.__qualname__
        assert wrapped.__doc__ != _test_func.__doc__
        assert "not implemented" in wrapped.__doc__
        assert wrapped.__wrapped__ is _test_func

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_called_once()
        assert mock_record_api_call.call_args[0] == ()
        assert mock_record_api_call.call_args[1]["name"] == "foo._test_func"
        assert not mock_record_api_call.call_args[1]["implemented"]
        filename, lineno = mock_record_api_call.call_args[1]["location"].split(
            ":"
        )
        assert int(lineno)

    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_False_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.unimplemented(
            _test_func, "foo", "_test_func", reporting=False
        )

        assert wrapped.__name__ == _test_func.__name__
        assert wrapped.__qualname__ == _test_func.__qualname__
        assert wrapped.__doc__ != _test_func.__doc__
        assert "not implemented" in wrapped.__doc__
        assert wrapped.__wrapped__ is _test_func

        with pytest.warns(RuntimeWarning) as record:
            assert wrapped(10, 20) == 30

        assert len(record) == 1
        assert record[0].message.args[0] == m.FALLBACK_WARNING.format(
            what="foo._test_func"
        )

        mock_record_api_call.assert_not_called()

    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_True_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.unimplemented(_test_ufunc, "foo", "_test_ufunc")

        assert wrapped.__doc__ != _test_ufunc.__doc__
        assert "not implemented" in wrapped.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_called_once()
        assert mock_record_api_call.call_args[0] == ()
        assert mock_record_api_call.call_args[1]["name"] == "foo._test_ufunc"
        assert not mock_record_api_call.call_args[1]["implemented"]
        filename, lineno = mock_record_api_call.call_args[1]["location"].split(
            ":"
        )
        assert int(lineno)

    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_False_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        wrapped = m.unimplemented(
            _test_ufunc, "foo", "_test_ufunc", reporting=False
        )

        assert wrapped.__doc__ != _test_ufunc.__doc__
        assert "not implemented" in wrapped.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        with pytest.warns(RuntimeWarning) as record:
            assert wrapped(10, 20) == 30

        assert len(record) == 1
        assert record[0].message.args[0] == m.FALLBACK_WARNING.format(
            what="foo._test_ufunc"
        )

        mock_record_api_call.assert_not_called()


_OriginMod = ModuleType("origin")
exec(
    """

import numpy as np

def __getattr__(name):
    pass

def __dir__():
    pass

attr1 = 10
attr2 = 20

def function1():
    pass

def function2():
    pass

""",
    _OriginMod.__dict__,
)

_DestCode = """
def function2():
    pass

def extra():
    pass

attr2 = 30
"""


class Test_clone_module:
    def test_report_coverage_True(self) -> None:
        settings.report_coverage = True

        _Dest = ModuleType("dest")
        exec(_DestCode, _Dest.__dict__)

        m.clone_module(_OriginMod, _Dest.__dict__)

        for name in m.MOD_INTERNAL:
            assert name not in _Dest.__dict__

        assert "np" not in _Dest.__dict__

        assert _Dest.attr1 == 10
        assert _Dest.attr2 == 30

        assert _Dest.function1.__wrapped__ is _OriginMod.function1
        assert not _Dest.function1._cupynumeric_metadata.implemented

        assert _Dest.function2.__wrapped__
        assert _Dest.function2._cupynumeric_metadata.implemented

        assert not hasattr(_Dest.extra, "_cupynumeric")

        settings.report_coverage.unset_value()

    def test_report_coverage_False(self) -> None:
        settings.report_coverage = True

        _Dest = ModuleType("dest")
        exec(_DestCode, _Dest.__dict__)

        m.clone_module(_OriginMod, _Dest.__dict__)

        for name in m.MOD_INTERNAL:
            assert name not in _Dest.__dict__

        assert "np" not in _Dest.__dict__

        assert _Dest.attr1 == 10
        assert _Dest.attr2 == 30

        assert _Dest.function1.__wrapped__ is _OriginMod.function1
        assert not _Dest.function1._cupynumeric_metadata.implemented

        assert _Dest.function2.__wrapped__
        assert _Dest.function2._cupynumeric_metadata.implemented

        assert not hasattr(_Dest.extra, "_cupynumeric")

        settings.report_coverage.unset_value()


class _Orig_ndarray:
    def __array_prepare__(self):
        return "I am now ready"

    def foo(self, other):
        assert type(self) == _Orig_ndarray  # noqa
        assert type(other) == _Orig_ndarray  # noqa
        return "original foo"

    def bar(self, other):
        assert False, "must never get here"


OMIT_NAMES = {"__array_prepare__"}


def fallback(x: Any) -> Any:
    if isinstance(x, _Test_ndarray):
        return _Orig_ndarray()
    return x


@m.clone_class(_Orig_ndarray, OMIT_NAMES, fallback)
class _Test_ndarray:
    def bar(self, other):
        return "new bar"

    def extra(self, other):
        return "new extra"

    attr1 = 10
    attr2 = 30


class Test_clone_class:
    def test_report_coverage_True(self) -> None:
        settings.report_coverage = True

        for name in OMIT_NAMES:
            assert name not in _Test_ndarray.__dict__

        assert _Test_ndarray.attr1 == 10
        assert _Test_ndarray.attr2 == 30

        assert _Test_ndarray.foo.__wrapped__ is _Orig_ndarray.foo
        assert not _Test_ndarray.foo._cupynumeric_metadata.implemented

        assert _Test_ndarray.bar.__wrapped__
        assert _Test_ndarray.bar._cupynumeric_metadata.implemented

        assert not hasattr(_Test_ndarray.extra, "_cupynumeric")

        settings.report_coverage.unset_value()

    def test_report_coverage_False(self) -> None:
        settings.report_coverage = False

        for name in OMIT_NAMES:
            assert name not in _Test_ndarray.__dict__

        assert _Test_ndarray.attr1 == 10
        assert _Test_ndarray.attr2 == 30

        assert _Test_ndarray.foo.__wrapped__ is _Orig_ndarray.foo
        assert not _Test_ndarray.foo._cupynumeric_metadata.implemented

        assert _Test_ndarray.bar.__wrapped__
        assert _Test_ndarray.bar._cupynumeric_metadata.implemented

        assert not hasattr(_Test_ndarray.extra, "_cupynumeric")

        settings.report_coverage.unset_value()

    def test_fallback(self):
        a = _Test_ndarray()
        b = _Test_ndarray()
        assert a.foo(b) == "original foo"
        assert a.bar(b) == "new bar"
        assert a.extra(b) == "new extra"


def test_ufunc_methods_binary() -> None:
    import cupynumeric as np

    # reduce is implemented
    assert np.add.reduce.__wrapped__
    assert np.add.reduce._cupynumeric_metadata.implemented

    # the rest are not
    assert np.add.reduceat.__wrapped__
    assert not np.add.reduceat._cupynumeric_metadata.implemented
    assert np.add.outer.__wrapped__
    assert not np.add.outer._cupynumeric_metadata.implemented
    assert np.add.at.__wrapped__
    assert not np.add.at._cupynumeric_metadata.implemented
    assert np.add.accumulate.__wrapped__
    assert not np.add.accumulate._cupynumeric_metadata.implemented


def test_ufunc_methods_unary() -> None:
    import cupynumeric as np

    assert np.negative.reduce.__wrapped__
    assert not np.negative.reduce._cupynumeric_metadata.implemented
    assert np.negative.reduceat.__wrapped__
    assert not np.negative.reduceat._cupynumeric_metadata.implemented
    assert np.negative.outer.__wrapped__
    assert not np.negative.outer._cupynumeric_metadata.implemented
    assert np.negative.at.__wrapped__
    assert not np.negative.at._cupynumeric_metadata.implemented
    assert np.negative.accumulate.__wrapped__
    assert not np.negative.accumulate._cupynumeric_metadata.implemented


def test_implemented_decorator_actual() -> None:
    settings.report_coverage = True
    
    def test_func():
        return "test"
        
    decorated_func = m.implemented(
        func=test_func,
        prefix="test_module",
        name="test_function",
        reporting=True
    )
    
    result = decorated_func()
    assert result == "test"
    
    assert hasattr(decorated_func, "_cupynumeric_metadata")
    assert decorated_func._cupynumeric_metadata.implemented


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
