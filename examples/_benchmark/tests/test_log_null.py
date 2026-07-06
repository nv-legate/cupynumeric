# Copyright 2026 NVIDIA Corporation
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

from __future__ import annotations

import sys

import pytest
from legate.util.benchmark import BenchmarkLog

from _benchmark.log_null import BenchmarkLogNull


def test_is_benchmark_log_subclass() -> None:
    assert issubclass(BenchmarkLogNull, BenchmarkLog)


def test_init_takes_no_arguments() -> None:
    log = BenchmarkLogNull()
    assert isinstance(log, BenchmarkLogNull)


def test_enter_returns_self() -> None:
    log = BenchmarkLogNull()
    assert log.__enter__() is log


def test_context_manager_usage() -> None:
    with BenchmarkLogNull() as log:
        assert isinstance(log, BenchmarkLogNull)


def test_context_manager_does_not_swallow_exceptions() -> None:
    with pytest.raises(ValueError, match="propagate"):
        with BenchmarkLogNull():
            raise ValueError("propagate")


def test_log_inside_context_manager() -> None:
    with BenchmarkLogNull() as log:
        assert log.log(value=42) is None  # type: ignore[func-returns-value]


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
