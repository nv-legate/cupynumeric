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

from types import ModuleType
from typing import Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

__all__ = ["Timer", "CuPyTimer", "CuPyNumericTimer", "NumPyTimer"]

NO_START_ERR_MSG: str = "Timer.stop() called without preceding Timer.start()"


class Timer(Protocol):
    def start(self) -> None: ...

    def stop(self) -> float:
        """
        Blocks execution until everything before it has completed. Returns the
        duration since the last call to start(), in milliseconds.
        """
        ...


class CuPyNumericTimer(Timer):
    _start_time: Any | None

    def __init__(self) -> None:
        self._start_time = None

    def start(self) -> None:
        from legate.timing import time

        self._start_time = time("us")

    def stop(self) -> float:
        from legate.timing import time

        assert self._start_time is not None, NO_START_ERR_MSG

        end_future = time("us")
        return float((end_future - self._start_time) / 1000.0)


class CuPyTimer(Timer):
    # TODO(tisaac): Add correct annotation when cupy provides stubs
    _start_event: Any

    def __init__(self) -> None:
        self._start_event = None

    def start(self) -> None:
        from cupy import cuda  # type: ignore[import-untyped]

        self._start_event = cuda.Event()
        self._start_event.record()

    def stop(self) -> float:
        from cupy import cuda  # type: ignore[import-untyped]

        assert self._start_event is not None, NO_START_ERR_MSG

        end_event = cuda.Event()
        end_event.record()
        end_event.synchronize()
        out: float = cuda.get_elapsed_time(self._start_event, end_event)
        return out


class NumPyTimer(Timer):
    _start_time: float | None

    def __init__(self) -> None:
        self._start_time = None

    def start(self) -> None:
        from time import perf_counter_ns

        self._start_time = perf_counter_ns() / 1000.0

    def stop(self) -> float:
        from time import perf_counter_ns

        assert self._start_time is not None, NO_START_ERR_MSG

        end_time = perf_counter_ns() / 1000.0
        return (end_time - self._start_time) / 1000.0


def get_timer(np: ModuleType) -> Timer:
    """Get a timer appropriate for an array package.

    The timer as :py:meth:`Timer.start` and :py:meth:`Timer.stop`.  ``stop()``
    waits for the completion of work initiated by the array package since
    ``start()`` before returning.
    """
    match np.__name__:
        case "numpy":
            return NumPyTimer()
        case "cupy":
            return CuPyTimer()
        case "cupynumeric":
            return CuPyNumericTimer()
        case _:
            raise RuntimeError(f"Unsupported array module {np.__name__}")


def timed_loop(
    f: Callable[..., Any], timer: Timer, runs: int, warmup: int = 0
) -> float:
    """Calls ``f()`` in a loop and reports the time for ``runs`` iterations.

    ``f()`` is called ``warmup`` times before the timer starts.
    """
    if runs < 0:
        raise RuntimeError(f"Negative runs {runs} not allowed.")
    if warmup < 0:
        raise RuntimeError(f"Negative warmup {warmup} not allowed.")
    if runs == 0:
        return 0.0
    for i in range(runs + warmup):
        if i == warmup:
            timer.start()
        f()
    return timer.stop()
