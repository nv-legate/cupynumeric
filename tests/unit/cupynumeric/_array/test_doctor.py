# Copyright 2025 NVIDIA Corporation
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
from typing import Any

import numpy as np
import pytest

import cupynumeric._array.doctor as m  # module under test
from cupynumeric.settings import settings


class MockCheckup(m.Checkup):
    description = "mock checkup"
    reference = "some link"

    def run(self, func: str, args: Any, _kwargs: Any) -> m.Diagnostic | None:
        return None


class Test_lookup_source:
    def test_good(self) -> None:
        assert m.lookup_source(__file__, 1) == (
            "# Copyright 2025 NVIDIA Corporation"
        )

    def test_bad(self) -> None:
        assert m.lookup_source("junk", 1) is None

    def test_lineno_out_of_range(self) -> None:
        assert m.lookup_source(__file__, 99999) is None


class Test_is_scalar_key:
    @pytest.mark.parametrize("key", (-1, 0, 1, (1,)))
    def test_good_1d(self, key) -> None:
        assert m.is_scalar_key(key, 1)

    @pytest.mark.parametrize("n", range(2, 10))
    def test_good_nd(self, n) -> None:
        key = tuple(range(n))
        assert m.is_scalar_key(key, n)

    def test_bad_dim(self) -> None:
        assert not m.is_scalar_key(1, 2)
        assert not m.is_scalar_key((1, 2), 1)

    @pytest.mark.parametrize("key", ([0, 1], ..., np.array([1, 2])))
    def test_bad_type(self, key) -> None:
        assert not m.is_scalar_key(key, 2)


class TestCheckup:
    def test_locate(self) -> None:
        checkup = MockCheckup()
        locator = checkup.locate()

        # Not alot we can reliably assert here. With the current code
        # organization, the stack frame lookup lands inside the pytest
        # package. Since the pytest modules are out of our control it
        # would be fragile to check for any specific line number.
        assert locator.lineno > 10

    def test_locate_returns_none(self, monkeypatch) -> None:
        checkup = MockCheckup()
        monkeypatch.setattr(m, "find_last_user_frame", lambda: None)
        locator = checkup.locate()
        assert locator is None

    def test_info(self) -> None:
        checkup = MockCheckup()
        locator = m.CheckupLocator(__file__, lineno=5, traceback="tb")
        info = checkup.info(locator)

        assert info == m.Diagnostic(
            filename=__file__,
            lineno=5,
            traceback="tb",
            source="# You may obtain a copy of the License at",
            description="mock checkup",
            reference="some link",
        )

    def test_report_first(self):
        checkup = MockCheckup()
        locator = checkup.locate()
        info = checkup.report(locator)
        assert info is not None

    def test_report_subsequent(self):
        checkup = MockCheckup()
        for i in range(5):
            locator = checkup.locate()
            info = checkup.report(locator)
            if i > 0:
                assert info is None


info1 = m.Diagnostic("fn1", 11, "tb1", "src1", "desc1", "ref1")
info2 = m.Diagnostic("fn2", 12, "tb2", "src2", "desc2", "ref2")


class TestDoctor:
    def test_default_checks(self) -> None:
        d = m.Doctor()
        assert {x.__class__ for x in d.checks} == set(m.ALL_CHECKS)

    def test_results_empty(self) -> None:
        d = m.Doctor()
        assert d.results == ()

    def test_results(self) -> None:
        d = m.Doctor()
        d._results = [info1, info2]
        assert d.results == (info1, info2)

    def test_output_empty(self) -> None:
        d = m.Doctor()
        assert d.output is None

    def test_output_plain(self) -> None:
        settings.doctor_format = "plain"
        d = m.Doctor()
        d._results = [info1, info2]
        assert (
            d.output
            == """
!!! cuPyNumeric Doctor reported issues !!!

- issue: desc1
  detected on: line 11 of file 'fn1':

    src1

  refer to: ref1

- issue: desc2
  detected on: line 12 of file 'fn2':

    src2

  refer to: ref2
"""
        )
        settings.doctor_format.unset_value()

    def test_output_plain_with_traceback(self) -> None:
        settings.doctor_format = "plain"
        settings.doctor_traceback = True
        d = m.Doctor()
        d._results = [info1]
        output = d.output
        assert output is not None
        assert "FULL TRACEBACK" in output
        assert "tb1" in output
        settings.doctor_format.unset_value()
        settings.doctor_traceback.unset_value()

    def test_output_json(self):
        settings.doctor_format = "json"
        d = m.Doctor()
        d._results = [info1, info2]
        assert (
            d.output
            == """\
[{"filename": "fn1", "lineno": 11, "traceback": "", "source": "src1", "description": "desc1", "reference": "ref1"}, {"filename": "fn2", "lineno": 12, "traceback": "", "source": "src2", "description": "desc2", "reference": "ref2"}]
"""  # noqa: E501
        )
        settings.doctor_format.unset_value()

    def test_output_csv(self):
        settings.doctor_format = "csv"
        d = m.Doctor()
        d._results = [info1, info2]
        assert (
            d.output
            == """\
filename,lineno,traceback,source,description,reference
fn1,11,,src1,desc1,ref1
fn2,12,,src2,desc2,ref2
"""
        )
        settings.doctor_format.unset_value()

    def test_output_csv_with_traceback(self) -> None:
        settings.doctor_format = "csv"
        settings.doctor_traceback = True
        d = m.Doctor()
        d._results = [info1]
        output = d.output
        assert output is not None
        # Newlines should be escaped in CSV format
        assert "\\n" in output or "tb1" in output
        settings.doctor_format.unset_value()
        settings.doctor_traceback.unset_value()

    def test_output_exception(self) -> None:
        settings.doctor_format = "invalid_format"
        d = m.Doctor()
        d._results = [info1]

        with pytest.warns(
            UserWarning, match="exception occurred generating output"
        ):
            output = d.output
            assert output is None

        settings.doctor_format.unset_value()

    def test_diagnose(self):
        # Test Doctor.diagnose method
        d = m.Doctor()
        initial_count = len(d.results)

        # Call diagnose with parameters that will trigger a diagnostic
        # Using RepeatedItemOps which triggers after 10+ calls
        for i in range(15):
            d.diagnose("__getitem__", (_Ndim(1), 10), {})

        assert len(d.results) > initial_count


# simulate an array with given ndim
class _Ndim:
    def __init__(self, ndim: int) -> None:
        self.ndim = ndim


class TestRepeatedItemOps:
    def test_ITEMOP_THRESHOLD(self) -> None:
        checkup = m.RepeatedItemOps()
        assert checkup.ITEMOP_THRESHOLD == 10

    def test_default(self) -> None:
        checkup = m.RepeatedItemOps()
        assert (
            checkup.description
            == "multiple scalar item accesses repeated on the same line"
        )
        assert checkup.reference is not None

    def test_run_non_itemop(self) -> None:
        checkup = m.RepeatedItemOps()
        for i in range(checkup.ITEMOP_THRESHOLD + 1):
            info = checkup.run("junk", (_Ndim(1), 10), {})
            assert info is None

    @pytest.mark.parametrize("func", ("__getitem__", "__setitem__"))
    def test_run_itemop_under_threshold(self, func: str) -> None:
        checkup = m.RepeatedItemOps()
        for i in range(checkup.ITEMOP_THRESHOLD):
            info = checkup.run(func, (_Ndim(1), 10), {})
            assert info is None

    @pytest.mark.parametrize("func", ("__getitem__", "__setitem__"))
    def test_run_itemop_at_threshold(self, func: str) -> None:
        checkup = m.RepeatedItemOps()
        for i in range(checkup.ITEMOP_THRESHOLD + 1):
            info = checkup.run(func, (_Ndim(1), 10), {})
        assert info is not None

    @pytest.mark.parametrize("func", ("__getitem__", "__setitem__"))
    def test_run_itemop_over_threshold(self, func: str) -> None:
        checkup = m.RepeatedItemOps()
        for i in range(checkup.ITEMOP_THRESHOLD + 2):
            info = checkup.run(func, (_Ndim(1), 10), {})
        assert info is None

    @pytest.mark.parametrize("func", ("__getitem__", "__setitem__"))
    def test_run_itemop_locator_none(self, func: str, monkeypatch) -> None:
        checkup = m.RepeatedItemOps()
        monkeypatch.setattr(checkup, "locate", lambda: None)
        info = checkup.run(func, (_Ndim(1), 10), {})
        assert info is None


class TestArrayGatherCheck:
    def test_run_numpy_array(self) -> None:
        checkup = m.ArrayGatherCheck()
        info = checkup.run("__numpy_array__", (), {})
        assert info is not None

    def test_run_other_func(self) -> None:
        checkup = m.ArrayGatherCheck()
        info = checkup.run("some_other_func", (), {})
        assert info is None


def test_ALL_CHECKS() -> None:
    assert m.ALL_CHECKS == (m.RepeatedItemOps, m.ArrayGatherCheck)


def test_doctor() -> None:
    assert isinstance(m.doctor, m.Doctor)


class TestAtexitRegistration:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for atexit tests"""
        # Save current settings
        old_doctor = (
            settings.doctor._value
            if hasattr(settings.doctor, "_value")
            else None
        )
        old_filename = (
            settings.doctor_filename._value
            if hasattr(settings.doctor_filename, "_value")
            else None
        )

        yield

        # Restore settings
        if old_doctor is not None:
            settings.doctor._value = old_doctor
        else:
            settings.doctor.unset_value()
        if old_filename is not None:
            settings.doctor_filename._value = old_filename
        else:
            settings.doctor_filename.unset_value()

    def _reload_and_call_atexit(self, results=None):
        """Helper to reload module and call atexit"""
        import importlib
        import sys

        settings.doctor = True

        if "cupynumeric._array.doctor" in sys.modules:
            importlib.reload(m)

        if results is not None:
            m.doctor._results = results

        if hasattr(m, "_doctor_atexit"):
            m._doctor_atexit()

    def test_doctor_atexit_with_output(self, tmp_path) -> None:
        test_file = tmp_path / "doctor_output.txt"
        settings.doctor_filename = str(test_file)

        self._reload_and_call_atexit(results=[info1])

        assert test_file.exists()
        content = test_file.read_text()
        assert "desc1" in content

    def test_doctor_atexit_no_output(self, capsys) -> None:
        self._reload_and_call_atexit(results=[])

        captured = capsys.readouterr()
        assert "!!! cuPyNumeric Doctor reported issues !!!" not in captured.out

    def test_doctor_atexit_print_to_stdout(self, capsys) -> None:
        settings.doctor_filename.unset_value()

        self._reload_and_call_atexit(results=[info1])

        captured = capsys.readouterr()
        assert "desc1" in captured.out


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
