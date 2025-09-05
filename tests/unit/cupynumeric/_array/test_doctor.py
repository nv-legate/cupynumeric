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


def test_ALL_CHECKS() -> None:
    assert m.ALL_CHECKS == (m.RepeatedItemOps, m.ArrayGatherCheck)


def test_doctor() -> None:
    assert isinstance(m.doctor, m.Doctor)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
