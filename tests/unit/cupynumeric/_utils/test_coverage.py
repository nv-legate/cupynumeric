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

import sys
from types import ModuleType
from typing import Any

import pytest
from mock import MagicMock, patch

import cupynumeric
import cupynumeric._utils.coverage as m  # module under test
from cupynumeric.settings import settings


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
        super().__init__("_test_ufunc", "docstring", "op_code")

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


class Test_implemented:
    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_True_func(
        self, mock_record_api_call: MagicMock
    ) -> None:
        settings.report_coverage = True
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
        settings.report_coverage = False
        wrapped = m.implemented(_test_func, "foo", "_test_func")

        assert wrapped.__name__ == _test_func.__name__
        assert wrapped.__qualname__ == _test_func.__qualname__
        assert wrapped.__doc__ == _test_func.__doc__
        assert wrapped.__wrapped__ is _test_func

        assert wrapped(10, 20) == 30

        mock_record_api_call.assert_not_called()

    @patch("legate.core.get_legate_runtime")
    @patch("cupynumeric.runtime.record_api_call")
    def test_profiling_enabled_reporting_False(
        self,
        mock_record_api_call: MagicMock,
        mock_get_legate_runtime: MagicMock,
    ) -> None:
        # Mock legate runtime for profiling
        mock_runtime = MagicMock()
        mock_config = MagicMock()
        mock_config.profile = True
        mock_runtime.config.return_value = mock_config
        mock_get_legate_runtime.return_value = mock_runtime

        # Clear cache to use our mock
        m._profiling_enabled.cache_clear()

        try:
            settings.report_coverage = False
            wrapped = m.implemented(_test_func, "foo", "_test_func")
            result = wrapped(10, 20)
            assert result == 30
            mock_record_api_call.assert_not_called()
        finally:
            m._profiling_enabled.cache_clear()
            settings.report_coverage.unset_value()

    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_enabled_profiling_disabled(
        self, mock_record_api_call: MagicMock
    ) -> None:
        try:
            settings.report_coverage = True
            wrapped = m.implemented(_test_func, "foo", "_test_func")

            result = wrapped(10, 20)
            assert result == 30

            mock_record_api_call.assert_called_once()
            assert (
                mock_record_api_call.call_args[1]["name"] == "foo._test_func"
            )
            assert mock_record_api_call.call_args[1]["implemented"]
        finally:
            settings.report_coverage.unset_value()

    @patch("cupynumeric._utils.coverage._profiling_enabled")
    @patch("legate.core.get_legate_runtime")
    def test_profiling_only_no_reporting(
        self,
        mock_get_legate_runtime: MagicMock,
        mock_profiling_enabled: MagicMock,
    ) -> None:
        # Mock profiling to return True
        mock_profiling_enabled.return_value = True

        mock_runtime = MagicMock()
        mock_get_legate_runtime.return_value = mock_runtime

        try:
            settings.report_coverage = False
            wrapped = m.implemented(_test_func, "foo", "_test_func")

            result = wrapped(10, 20)
            assert result == 30
        finally:
            settings.report_coverage.unset_value()

    @patch("cupynumeric.runtime.record_api_call")
    def test_reporting_True_ufunc(
        self, mock_record_api_call: MagicMock
    ) -> None:
        settings.report_coverage = True
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
        settings.report_coverage = False
        wrapped = m.implemented(_test_ufunc, "foo", "_test_func")

        # these had to be special-cased, @wraps does not handle them
        assert wrapped.__name__ == _test_ufunc._name
        assert wrapped.__qualname__ == _test_ufunc._name

        assert wrapped.__doc__ == _test_ufunc.__doc__
        assert wrapped.__wrapped__ is _test_ufunc

        assert wrapped(10, 20) == 30

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

        # Non-function attributes are still copied
        assert _Dest.attr1 == 10
        assert _Dest.attr2 == 30

        # Unimplemented functions are NO LONGER copied (no fallback)
        # function1 is not in _Dest, should raise AttributeError
        with pytest.raises(
            AttributeError, match="has no attribute 'function1'"
        ):
            _ = _Dest.function1

        # Implemented functions are wrapped with metadata
        assert _Dest.function2.__wrapped__
        assert _Dest.function2._cupynumeric_metadata.implemented

        # Extra functions not in origin remain unwrapped
        assert not hasattr(_Dest.extra, "_cupynumeric_metadata")

        settings.report_coverage.unset_value()

    def test_report_coverage_False(self) -> None:
        settings.report_coverage = False

        _Dest = ModuleType("dest")
        exec(_DestCode, _Dest.__dict__)

        m.clone_module(_OriginMod, _Dest.__dict__)

        for name in m.MOD_INTERNAL:
            assert name not in _Dest.__dict__

        assert "np" not in _Dest.__dict__

        # Non-function attributes are still copied
        assert _Dest.attr1 == 10
        assert _Dest.attr2 == 30

        # Unimplemented functions are NO LONGER copied (no fallback)
        # function1 is not in _Dest, should raise AttributeError
        with pytest.raises(
            AttributeError, match="has no attribute 'function1'"
        ):
            _ = _Dest.function1

        # Implemented functions are wrapped with metadata
        assert _Dest.function2.__wrapped__
        assert _Dest.function2._cupynumeric_metadata.implemented

        # Extra functions not in origin remain unwrapped
        assert not hasattr(_Dest.extra, "_cupynumeric_metadata")

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

        # Non-method attributes are still copied
        assert _Test_ndarray.attr1 == 10
        assert _Test_ndarray.attr2 == 30

        # Unimplemented methods are NO LONGER copied (no fallback)
        # foo is not in _Test_ndarray, should raise AttributeError
        with pytest.raises(AttributeError, match="has no attribute 'foo'"):
            _ = _Test_ndarray.foo

        # Implemented methods are wrapped with metadata
        assert _Test_ndarray.bar.__wrapped__
        assert _Test_ndarray.bar._cupynumeric_metadata.implemented

        # Extra methods not in origin remain unwrapped
        assert not hasattr(_Test_ndarray.extra, "_cupynumeric_metadata")

        settings.report_coverage.unset_value()

    def test_report_coverage_False(self) -> None:
        settings.report_coverage = False

        for name in OMIT_NAMES:
            assert name not in _Test_ndarray.__dict__

        # Non-method attributes are still copied
        assert _Test_ndarray.attr1 == 10
        assert _Test_ndarray.attr2 == 30

        # Unimplemented methods are NO LONGER copied (no fallback)
        # foo is not in _Test_ndarray, should raise AttributeError
        with pytest.raises(AttributeError, match="has no attribute 'foo'"):
            _ = _Test_ndarray.foo

        # Implemented methods are wrapped with metadata
        assert _Test_ndarray.bar.__wrapped__
        assert _Test_ndarray.bar._cupynumeric_metadata.implemented

        # Extra methods not in origin remain unwrapped
        assert not hasattr(_Test_ndarray.extra, "_cupynumeric_metadata")

        settings.report_coverage.unset_value()

    def test_no_fallback(self):
        """Test that fallback to NumPy has been removed."""
        a = _Test_ndarray()
        b = _Test_ndarray()

        # Unimplemented methods should raise AttributeError (no fallback)
        with pytest.raises(AttributeError, match="has no attribute 'foo'"):
            a.foo(b)

        # Implemented methods still work
        assert a.bar(b) == "new bar"
        assert a.extra(b) == "new extra"


def test_ufunc_methods_binary() -> None:
    import cupynumeric as np

    # reduce is implemented
    assert np.add.reduce.__wrapped__
    assert np.add.reduce._cupynumeric_metadata.implemented

    # Unimplemented ufunc methods are NO LONGER available (no fallback)
    with pytest.raises(AttributeError):
        _ = np.add.reduceat
    with pytest.raises(AttributeError):
        _ = np.add.outer
    with pytest.raises(AttributeError):
        _ = np.add.at
    with pytest.raises(AttributeError):
        _ = np.add.accumulate


def test_ufunc_methods_unary() -> None:
    import cupynumeric as np

    # All ufunc methods for unary ufuncs are unimplemented
    # After removing fallback, they should not exist (raise AttributeError)
    with pytest.raises(AttributeError):
        _ = np.negative.reduce
    with pytest.raises(AttributeError):
        _ = np.negative.reduceat
    with pytest.raises(AttributeError):
        _ = np.negative.outer
    with pytest.raises(AttributeError):
        _ = np.negative.at
    with pytest.raises(AttributeError):
        _ = np.negative.accumulate


def test_implemented_decorator_actual() -> None:
    settings.report_coverage = True

    def test_func():
        return "test"

    decorated_func = m.implemented(
        func=test_func, prefix="test_module", name="test_function"
    )

    result = decorated_func()
    assert result == "test"

    assert hasattr(decorated_func, "_cupynumeric_metadata")
    assert decorated_func._cupynumeric_metadata.implemented


class TestInferMode:
    def test_infer_mode_none(self) -> None:
        result = m._infer_mode(None)
        assert result == "N/A"

    def test_infer_mode_with_thunk_deferred(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_result = MagicMock()
        mock_result._thunk = MagicMock()
        monkeypatch.setattr(
            "cupynumeric._utils.coverage.runtime.is_deferred_array",
            lambda x: True,
        )
        result = m._infer_mode(mock_result)
        assert result == "deferred"

    def test_infer_mode_with_thunk_eager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_result = MagicMock()
        mock_result._thunk = MagicMock()
        monkeypatch.setattr(
            "cupynumeric._utils.coverage.runtime.is_deferred_array",
            lambda x: False,
        )
        result = m._infer_mode(mock_result)
        assert result == "eager"

    def test_infer_mode_numpy_ndarray(self) -> None:
        import numpy as np

        arr = np.array([1, 2, 3])
        result = m._infer_mode(arr)
        assert result == "eager"

    def test_infer_mode_else(self) -> None:
        result = m._infer_mode("some_string")
        assert result == "N/A"


class TestProfileRange:
    def test_profile_range_with_result(self) -> None:
        try:
            with m.ProfileRange("test_func", "test.py:10"):
                m.upr_result.set("test_result")
        except Exception:
            # Ignore any errors
            pass


class TestHelperFunctions:
    def test_is_single_not_wrapped(self) -> None:
        result = m.is_single("not_wrapped_object")
        assert result == m.GPUSupport.NO

    def test_is_single_wrapped(self) -> None:
        wrapped = m.implemented(_test_func, "test", "func")
        result = m.is_single(wrapped)
        # Check it returns a GPUSupport value
        assert isinstance(result, m.GPUSupport)

    def test_is_multi_not_wrapped(self) -> None:
        result = m.is_multi("not_wrapped_object")
        assert result == m.GPUSupport.NO

    def test_is_multi_wrapped(self) -> None:
        wrapped = m.implemented(_test_func, "test", "func")
        result = m.is_multi(wrapped)
        # Check it returns a GPUSupport value
        assert isinstance(result, m.GPUSupport)


def test_finish_triggers_shutdown(monkeypatch) -> None:
    runtime_mod = sys.modules["cupynumeric.runtime"]
    monkeypatch.setattr(runtime_mod.runtime, "destroyed", False)
    mock_legate_runtime = MagicMock()

    def mock_finish():
        runtime_mod._shutdown_callback()

    mock_legate_runtime.finish = mock_finish
    monkeypatch.setattr(runtime_mod, "legate_runtime", mock_legate_runtime)
    runtime_mod.legate_runtime.finish()
    assert runtime_mod.runtime.destroyed is True


def test_bitgenerator_destroy_else(monkeypatch) -> None:
    runtime = cupynumeric.runtime
    runtime_mod = sys.modules["cupynumeric.runtime"]

    mock_runtime = MagicMock()
    monkeypatch.setattr(runtime_mod, "legate_runtime", mock_runtime)
    mock_task = MagicMock()
    mock_runtime.create_manual_task.return_value = mock_task

    runtime.current_random_bitgen_zombies = (123,)
    handle = 456
    runtime.bitgenerator_destroy(handle, disposing=False)

    mock_runtime.issue_execution_fence.assert_called_once()
    mock_runtime.create_manual_task.assert_called_once()
    mock_task.execute.assert_called_once()
    assert runtime.current_random_bitgen_zombies == ()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
