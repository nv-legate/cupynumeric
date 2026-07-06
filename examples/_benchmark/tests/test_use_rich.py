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
from typing import TextIO
from unittest.mock import MagicMock

import pytest

from _benchmark import use_rich as use_rich_mod
from _benchmark.use_rich import use_rich


def _make_stream(*, isatty: bool) -> TextIO:
    stream = MagicMock(spec=["isatty"])
    stream.isatty.return_value = isatty
    return stream  # type: ignore[no-any-return]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LEGATE_BENCHMARK_USE_RICH", raising=False)
    monkeypatch.delenv("LEGATE_LIMIT_STDOUT", raising=False)


@pytest.fixture
def runtime_not_started(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "_benchmark.use_rich.runtime_has_started", lambda: False
    )


@pytest.fixture
def runtime_started(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "_benchmark.use_rich.runtime_has_started", lambda: True
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch, *, num_nodes: int
) -> MagicMock:
    machine = MagicMock()
    machine.get_node_range.return_value = (0, num_nodes)
    runtime = MagicMock()
    runtime.get_machine.return_value = machine
    monkeypatch.setattr("legate.core.get_legate_runtime", lambda: runtime)
    return runtime


# ---------------------------------------------------------------------------
# early returns
# ---------------------------------------------------------------------------


def test_returns_false_when_rich_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", False)
    assert use_rich(_make_stream(isatty=True)) is False


def test_returns_false_when_stream_is_not_a_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    assert use_rich(_make_stream(isatty=False)) is False


def test_returns_false_when_env_var_disables(
    monkeypatch: pytest.MonkeyPatch, runtime_not_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    monkeypatch.setenv("LEGATE_BENCHMARK_USE_RICH", "0")
    assert use_rich(_make_stream(isatty=True)) is False


def test_does_not_touch_runtime_when_disabled_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", False)
    sentinel = MagicMock(
        side_effect=AssertionError("runtime_has_started should not be called")
    )
    monkeypatch.setattr("_benchmark.use_rich.runtime_has_started", sentinel)
    use_rich(_make_stream(isatty=True))
    sentinel.assert_not_called()


# ---------------------------------------------------------------------------
# runtime not started
# ---------------------------------------------------------------------------


def test_returns_true_when_runtime_not_started_and_not_forced(
    monkeypatch: pytest.MonkeyPatch, runtime_not_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    # Do not start runtime; should short-circuit to True without touching it.
    sentinel = MagicMock(
        side_effect=AssertionError("get_legate_runtime should not be called")
    )
    monkeypatch.setattr("legate.core.get_legate_runtime", sentinel)
    assert use_rich(_make_stream(isatty=True)) is True
    sentinel.assert_not_called()


def test_returns_true_when_env_var_default_is_used(
    monkeypatch: pytest.MonkeyPatch, runtime_not_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    # LEGATE_BENCHMARK_USE_RICH is unset by the autouse fixture; default is "1".
    assert use_rich(_make_stream(isatty=True)) is True


def test_returns_true_when_env_var_is_any_non_zero_value(
    monkeypatch: pytest.MonkeyPatch, runtime_not_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    monkeypatch.setenv("LEGATE_BENCHMARK_USE_RICH", "true")
    assert use_rich(_make_stream(isatty=True)) is True


# ---------------------------------------------------------------------------
# runtime started / start_runtime
# ---------------------------------------------------------------------------


def test_returns_true_when_runtime_started_with_single_node(
    monkeypatch: pytest.MonkeyPatch, runtime_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    _install_fake_runtime(monkeypatch, num_nodes=1)
    assert use_rich(_make_stream(isatty=True)) is True


def test_returns_false_when_runtime_started_with_multi_node(
    monkeypatch: pytest.MonkeyPatch, runtime_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    _install_fake_runtime(monkeypatch, num_nodes=4)
    assert use_rich(_make_stream(isatty=True)) is False


def test_returns_true_when_multi_node_but_stdout_limited(
    monkeypatch: pytest.MonkeyPatch, runtime_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    _install_fake_runtime(monkeypatch, num_nodes=4)
    monkeypatch.setenv("LEGATE_LIMIT_STDOUT", "1")
    assert use_rich(_make_stream(isatty=True)) is True


def test_returns_false_when_multi_node_and_stdout_limit_zero(
    monkeypatch: pytest.MonkeyPatch, runtime_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    _install_fake_runtime(monkeypatch, num_nodes=2)
    monkeypatch.setenv("LEGATE_LIMIT_STDOUT", "0")
    assert use_rich(_make_stream(isatty=True)) is False


# ---------------------------------------------------------------------------
# start_runtime keyword forces the runtime branch
# ---------------------------------------------------------------------------


def test_start_runtime_true_consults_runtime_even_when_not_started(
    monkeypatch: pytest.MonkeyPatch, runtime_not_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    runtime = _install_fake_runtime(monkeypatch, num_nodes=1)
    assert use_rich(_make_stream(isatty=True), start_runtime=True) is True
    runtime.get_machine.assert_called_once_with()


def test_start_runtime_true_returns_false_for_multi_node_unlimited(
    monkeypatch: pytest.MonkeyPatch, runtime_not_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    _install_fake_runtime(monkeypatch, num_nodes=3)
    assert use_rich(_make_stream(isatty=True), start_runtime=True) is False


def test_start_runtime_true_returns_true_when_stdout_limited(
    monkeypatch: pytest.MonkeyPatch, runtime_not_started: None
) -> None:
    monkeypatch.setattr(use_rich_mod, "HAVE_RICH", True)
    _install_fake_runtime(monkeypatch, num_nodes=8)
    monkeypatch.setenv("LEGATE_LIMIT_STDOUT", "1")
    assert use_rich(_make_stream(isatty=True), start_runtime=True) is True


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
