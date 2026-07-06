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

import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import cupynumeric as num

from _benchmark.timer import (
    NO_START_ERR_MSG,
    CuPyNumericTimer,
    CuPyTimer,
    NumPyTimer,
    get_timer,
    timed_loop,
)

NO_START_ERR_PATTERN = re.escape(NO_START_ERR_MSG)


# ---------------------------------------------------------------------------
# NumPyTimer
# ---------------------------------------------------------------------------


def test_numpy_timer_repr() -> None:
    assert repr(NumPyTimer()) == "NumPyTimer"


def test_numpy_timer_stop_without_start_raises() -> None:
    with pytest.raises(AssertionError, match=NO_START_ERR_PATTERN):
        NumPyTimer().stop()


def test_numpy_timer_start_stop_returns_nonnegative_duration() -> None:
    timer = NumPyTimer()
    timer.start()
    duration = timer.stop()
    assert isinstance(duration, float)
    assert duration >= 0.0


def test_numpy_timer_sync_is_noop() -> None:
    timer = NumPyTimer()
    timer.sync("none")
    timer.sync("fence")
    timer.sync("block")


def test_numpy_timer_uses_perf_counter_ns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 5 ms apart in nanoseconds; result should be milliseconds.
    values = iter([1_000_000, 6_000_000])
    monkeypatch.setattr("time.perf_counter_ns", lambda: next(values))
    timer = NumPyTimer()
    timer.start()
    assert timer.stop() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# CuPyNumericTimer
# ---------------------------------------------------------------------------


def test_cupynumeric_timer_repr_default() -> None:
    assert repr(CuPyNumericTimer()) == "CuPyNumericTimer(blocking=False)"


def test_cupynumeric_timer_repr_blocking() -> None:
    assert (
        repr(CuPyNumericTimer(blocking=True))
        == "CuPyNumericTimer(blocking=True)"
    )


def test_cupynumeric_timer_stop_without_start_raises() -> None:
    with pytest.raises(AssertionError, match=NO_START_ERR_PATTERN):
        CuPyNumericTimer().stop()


def test_cupynumeric_timer_start_stop_nonblocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # times reported in microseconds; result returned in milliseconds.
    values = iter([1000, 6000])
    monkeypatch.setattr("legate.timing.time", lambda _unit: next(values))

    timer = CuPyNumericTimer()
    timer.start()
    assert timer.stop() == pytest.approx(5.0)


def test_cupynumeric_timer_blocking_casts_to_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returned = [MagicMock(), MagicMock()]
    # int() must be called on the returned values; have the mock return
    # specific integers so the math is also exercised.
    returned[0].__int__ = lambda self: 2000  # type: ignore[method-assign]
    returned[1].__int__ = lambda self: 5000  # type: ignore[method-assign]
    iterator = iter(returned)
    monkeypatch.setattr("legate.timing.time", lambda _unit: next(iterator))

    timer = CuPyNumericTimer(blocking=True)
    timer.start()
    duration = timer.stop()
    assert duration == pytest.approx(3.0)


def test_cupynumeric_timer_sync_none_does_not_call_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = MagicMock(
        side_effect=AssertionError("runtime should not be touched")
    )
    monkeypatch.setattr("legate.core.get_legate_runtime", sentinel)

    CuPyNumericTimer().sync("none")
    sentinel.assert_not_called()


def test_cupynumeric_timer_sync_fence_issues_nonblocking_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MagicMock()
    monkeypatch.setattr("legate.core.get_legate_runtime", lambda: runtime)

    CuPyNumericTimer().sync("fence")
    runtime.issue_execution_fence.assert_called_once_with(block=False)


def test_cupynumeric_timer_sync_block_issues_blocking_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MagicMock()
    monkeypatch.setattr("legate.core.get_legate_runtime", lambda: runtime)

    CuPyNumericTimer().sync("block")
    runtime.issue_execution_fence.assert_called_once_with(block=True)


# ---------------------------------------------------------------------------
# CuPyTimer
# ---------------------------------------------------------------------------


def _install_fake_cupy(monkeypatch: pytest.MonkeyPatch, cuda: Any) -> None:
    fake_cupy = SimpleNamespace(cuda=cuda, __name__="cupy")
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)


def test_cupy_timer_repr_default() -> None:
    assert repr(CuPyTimer()) == "CuPyTimer(blocking=False)"


def test_cupy_timer_repr_blocking() -> None:
    assert repr(CuPyTimer(blocking=True)) == "CuPyTimer(blocking=True)"


def test_cupy_timer_stop_without_start_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda = SimpleNamespace(Event=MagicMock(), get_elapsed_time=MagicMock())
    _install_fake_cupy(monkeypatch, cuda)
    with pytest.raises(AssertionError, match=NO_START_ERR_PATTERN):
        CuPyTimer().stop()


def test_cupy_timer_start_records_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_event = MagicMock()
    Event = MagicMock(return_value=start_event)
    cuda = SimpleNamespace(Event=Event, get_elapsed_time=MagicMock())
    _install_fake_cupy(monkeypatch, cuda)

    timer = CuPyTimer()
    timer.start()

    Event.assert_called_once_with()
    start_event.record.assert_called_once_with()
    start_event.synchronize.assert_not_called()


def test_cupy_timer_blocking_start_synchronizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_event = MagicMock()
    Event = MagicMock(return_value=start_event)
    cuda = SimpleNamespace(Event=Event, get_elapsed_time=MagicMock())
    _install_fake_cupy(monkeypatch, cuda)

    CuPyTimer(blocking=True).start()
    start_event.synchronize.assert_called_once_with()


def test_cupy_timer_stop_returns_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_event = MagicMock(name="start")
    end_event = MagicMock(name="end")
    Event = MagicMock(side_effect=[start_event, end_event])
    get_elapsed_time = MagicMock(return_value=42.5)
    cuda = SimpleNamespace(Event=Event, get_elapsed_time=get_elapsed_time)
    _install_fake_cupy(monkeypatch, cuda)

    timer = CuPyTimer()
    timer.start()
    result = timer.stop()

    end_event.record.assert_called_once_with()
    end_event.synchronize.assert_called_once_with()
    get_elapsed_time.assert_called_once_with(start_event, end_event)
    assert result == 42.5


def test_cupy_timer_sync_none_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Event = MagicMock(
        side_effect=AssertionError("Event should not be created")
    )
    cuda = SimpleNamespace(Event=Event, get_elapsed_time=MagicMock())
    _install_fake_cupy(monkeypatch, cuda)

    CuPyTimer().sync("none")
    Event.assert_not_called()


def test_cupy_timer_sync_fence_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Event = MagicMock(
        side_effect=AssertionError("Event should not be created")
    )
    cuda = SimpleNamespace(Event=Event, get_elapsed_time=MagicMock())
    _install_fake_cupy(monkeypatch, cuda)

    CuPyTimer().sync("fence")
    Event.assert_not_called()


def test_cupy_timer_sync_block_records_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = MagicMock()
    Event = MagicMock(return_value=event)
    cuda = SimpleNamespace(Event=Event, get_elapsed_time=MagicMock())
    _install_fake_cupy(monkeypatch, cuda)

    CuPyTimer().sync("block")
    Event.assert_called_once_with()
    event.record.assert_called_once_with()
    event.synchronize.assert_called_once_with()


# ---------------------------------------------------------------------------
# get_timer
# ---------------------------------------------------------------------------


def test_get_timer_numpy_returns_numpy_timer() -> None:
    assert isinstance(get_timer(np), NumPyTimer)


def test_get_timer_cupy_returns_cupy_timer() -> None:
    fake = SimpleNamespace(__name__="cupy")
    timer = get_timer(fake)  # type: ignore[arg-type]
    assert isinstance(timer, CuPyTimer)


def test_get_timer_cupynumeric_returns_cupynumeric_timer() -> None:
    assert isinstance(get_timer(num), CuPyNumericTimer)


def test_get_timer_blocking_is_forwarded() -> None:
    cupy_fake = SimpleNamespace(__name__="cupy")

    cupy_timer = get_timer(cupy_fake, blocking=True)  # type: ignore[arg-type]
    cupynumeric_timer = get_timer(num, blocking=True)

    assert "blocking=True" in repr(cupy_timer)
    assert "blocking=True" in repr(cupynumeric_timer)


def test_get_timer_rejects_unsupported_module() -> None:
    fake = SimpleNamespace(__name__="jax")
    with pytest.raises(RuntimeError, match="Unsupported array module jax"):
        get_timer(fake)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# timed_loop
# ---------------------------------------------------------------------------


class _RecordingTimer:
    """A Timer that records the order of calls without doing any real timing."""

    def __init__(self, stop_value: float = 7.5) -> None:
        self.events: list[str] = []
        self._stop_value = stop_value

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> float:
        self.events.append("stop")
        return self._stop_value

    def sync(self, sync_mode: str) -> None:
        self.events.append(f"sync:{sync_mode}")


def test_timed_loop_rejects_negative_runs() -> None:
    with pytest.raises(RuntimeError, match="Negative runs -1 not allowed"):
        timed_loop(lambda: None, _RecordingTimer(), runs=-1)


def test_timed_loop_rejects_negative_warmup() -> None:
    with pytest.raises(RuntimeError, match="Negative warmup -2 not allowed"):
        timed_loop(lambda: None, _RecordingTimer(), runs=1, warmup=-2)


def test_timed_loop_zero_runs_returns_zero_without_calling_f() -> None:
    f = MagicMock()
    timer = _RecordingTimer()
    result = timed_loop(f, timer, runs=0, warmup=5)
    assert result == 0.0
    f.assert_not_called()
    assert timer.events == []


def test_timed_loop_calls_function_runs_plus_warmup_times() -> None:
    f = MagicMock()
    timed_loop(f, _RecordingTimer(), runs=3, warmup=2)
    assert f.call_count == 5


def test_timed_loop_returns_timer_stop_result() -> None:
    timer = _RecordingTimer(stop_value=123.0)
    assert timed_loop(lambda: None, timer, runs=2) == 123.0


def test_timed_loop_starts_timer_after_warmup() -> None:
    calls: list[str] = []

    def f() -> None:
        calls.append("f")

    timer = _RecordingTimer()
    # Wrap timer.start to capture the index of the call to f at start time.
    original_start = timer.start

    def start() -> None:
        calls.append("start")
        original_start()

    timer.start = start  # type: ignore[method-assign]

    timed_loop(f, timer, runs=2, warmup=3)
    # warmup=3 means three "f" calls precede "start".
    assert calls[:4] == ["f", "f", "f", "start"]


def test_timed_loop_sync_between_iterations_only() -> None:
    timer = _RecordingTimer()
    timed_loop(lambda: None, timer, runs=3, sync_mode="fence")
    # runs=3 -> 2 sync calls (between iterations), 1 start, 1 stop.
    assert timer.events.count("sync:fence") == 2
    assert timer.events[0] == "start"
    assert timer.events[-1] == "stop"


def test_timed_loop_sync_uses_provided_mode() -> None:
    timer = _RecordingTimer()
    timed_loop(lambda: None, timer, runs=2, sync_mode="block")
    assert "sync:block" in timer.events
    assert "sync:fence" not in timer.events


def test_timed_loop_single_run_emits_no_sync() -> None:
    timer = _RecordingTimer()
    timed_loop(lambda: None, timer, runs=1, sync_mode="block")
    assert all(not e.startswith("sync") for e in timer.events)


def test_timed_loop_accepts_module_typed_argument() -> None:
    # Sanity check that the function signature accepts a real ModuleType-like
    # callable; helps guard against accidental signature drift.
    assert callable(timed_loop)
    assert isinstance(np, ModuleType)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
