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
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import cupynumeric as num

from _benchmark.harness import (
    ArrayPackage,
    BenchmarkHarness,
    BenchmarkHarnessConfig,
    CupyAllocator,
    SummarizeFlush,
    TimerMode,
    parse_with_harness,
)
from _benchmark.timer import CuPyNumericTimer, NumPyTimer


# ---------------------------------------------------------------------------
# Helpers for BenchmarkHarness construction
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> BenchmarkHarnessConfig:
    defaults: dict[str, Any] = {
        "repeat": 0,
        "package": ArrayPackage.NUMPY,
        "short": True,
        "cupy_allocator": CupyAllocator.DEFAULT,
        "log_conda_list": False,
        "log_metadata_extra": {},
        "summarize": None,
        "summarize_flush": SummarizeFlush.RUN,
        "timer_mode": TimerMode.EXECUTION,
    }
    defaults.update(overrides)
    return BenchmarkHarnessConfig(**defaults)


def _make_namespace(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = {
        "__cpn_repeat": 0,
        "__cpn_package": ArrayPackage.NUMPY,
        "__cpn_short": True,
        "__cpn_summarize": False,
        "__cpn_summarize_flush": SummarizeFlush.RUN,
        "__cpn_cupy_allocator": CupyAllocator.DEFAULT,
        "__cpn_log_conda_list": False,
        "__cpn_metadata_extra": [],
        "__cpn_timer_mode": TimerMode.EXECUTION,
    }
    defaults.update(overrides)
    ns = Namespace()
    for key, value in defaults.items():
        setattr(ns, key, value)
    return ns


# ---------------------------------------------------------------------------
# BenchmarkHarness construction
# ---------------------------------------------------------------------------


def test_harness_init_from_config() -> None:
    harness = BenchmarkHarness(_make_config())
    assert harness.package == "numpy"
    assert harness.repeat == 0
    assert harness.short_metadata is True


def test_harness_init_from_namespace() -> None:
    harness = BenchmarkHarness(_make_namespace(__cpn_repeat=3))
    assert harness.repeat == 3


def test_harness_init_rejects_invalid_type() -> None:
    with pytest.raises(RuntimeError, match="unrecognized type"):
        BenchmarkHarness(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("package", "expected"),
    [(ArrayPackage.NUMPY, np), (ArrayPackage.LEGATE, num)],
)
def test_harness_np_attribute_is_package(
    package: ArrayPackage, expected: ModuleType
) -> None:
    harness = BenchmarkHarness(_make_config(package=package))
    assert harness.np is expected


@pytest.mark.parametrize(
    ("package", "expected"),
    [
        (ArrayPackage.NUMPY, NumPyTimer),
        (ArrayPackage.LEGATE, CuPyNumericTimer),
    ],
)
def test_harness_timers_are_package_timers(
    package: ArrayPackage, expected: type
) -> None:
    harness = BenchmarkHarness(_make_config(package=package))
    assert isinstance(harness.execution_timer, expected)
    assert isinstance(harness.wall_timer, expected)


def test_harness_repeat_property_proxies_config() -> None:
    assert BenchmarkHarness(_make_config(repeat=7)).repeat == 7


def test_harness_package_property_returns_str() -> None:
    harness = BenchmarkHarness(_make_config(package=ArrayPackage.NUMPY))
    assert harness.package == "numpy"


def test_harness_short_metadata_property() -> None:
    assert BenchmarkHarness(_make_config(short=False)).short_metadata is False


def test_harness_summarize_property_default_none() -> None:
    assert BenchmarkHarness(_make_config()).summarize is None


def test_harness_summarize_property_proxies_config() -> None:
    sentinel = MagicMock()
    harness = BenchmarkHarness(_make_config(summarize=sentinel))
    assert harness.summarize is sentinel


def test_harness_summarize_flush_property() -> None:
    harness = BenchmarkHarness(
        _make_config(summarize_flush=SummarizeFlush.NEVER)
    )
    assert harness.summarize_flush == SummarizeFlush.NEVER


def test_harness_timer_mode_property() -> None:
    harness = BenchmarkHarness(_make_config(timer_mode=TimerMode.WALL))
    assert harness.timer_mode == TimerMode.WALL


def test_harness_timer_returns_execution_timer_in_execution_mode() -> None:
    harness = BenchmarkHarness(_make_config(timer_mode=TimerMode.EXECUTION))
    assert harness.timer is harness.execution_timer


def test_harness_timer_returns_wall_timer_in_wall_mode() -> None:
    harness = BenchmarkHarness(_make_config(timer_mode=TimerMode.WALL))
    assert harness.timer is harness.wall_timer


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_harness_context_manager_enter_returns_self() -> None:
    harness = BenchmarkHarness(_make_config())
    with harness as h:
        assert h is harness


def test_harness_exit_summarize_flush_exit_calls_flush() -> None:
    summarize = MagicMock()
    harness = BenchmarkHarness(
        _make_config(summarize=summarize, summarize_flush=SummarizeFlush.EXIT)
    )
    with harness:
        summarize.flush.assert_not_called()
    summarize.flush.assert_called_once_with()


def test_harness_exit_summarize_flush_run_does_not_flush() -> None:
    summarize = MagicMock()
    harness = BenchmarkHarness(
        _make_config(summarize=summarize, summarize_flush=SummarizeFlush.RUN)
    )
    with harness:
        pass
    summarize.flush.assert_not_called()


def test_harness_exit_summarize_flush_never_does_not_flush() -> None:
    summarize = MagicMock()
    harness = BenchmarkHarness(
        _make_config(summarize=summarize, summarize_flush=SummarizeFlush.NEVER)
    )
    with harness:
        pass
    summarize.flush.assert_not_called()


def test_harness_exit_with_no_summarize_is_noop() -> None:
    # Even with summarize_flush=EXIT, no flush should happen when summarize
    # itself is None.
    harness = BenchmarkHarness(
        _make_config(summarize=None, summarize_flush=SummarizeFlush.EXIT)
    )
    with harness:
        pass


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_harness_short_metadata_skips_legate_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = MagicMock(
        side_effect=AssertionError("legate_info should not be called")
    )
    monkeypatch.setattr("_benchmark.harness.legate_info", sentinel)
    harness = BenchmarkHarness(_make_config(short=True))
    sentinel.assert_not_called()
    assert harness.metadata == {}


def test_harness_short_false_collects_legate_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "_benchmark.harness.legate_info",
        lambda **kwargs: {"info_key": "info_value"},
    )
    harness = BenchmarkHarness(
        _make_config(short=False, package=ArrayPackage.NUMPY)
    )
    assert harness.metadata["info_key"] == "info_value"


def test_harness_cupy_package_updates_metadata_version_and_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With `--package cupy` and not `--short`, the harness should populate
    # the "cupy" entry in both the "Package versions" and "Package details"
    # sub-dicts of the metadata returned by legate_info().
    # Fake cupy so `import cupy` succeeds in __init__ without touching real
    # GPU bindings. The default cupy_allocator skips any .cuda access.
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(__name__="cupy"))
    monkeypatch.setattr(
        "_benchmark.harness.legate_info",
        lambda **kwargs: {
            "Package versions": {"numpy": "x.y.z"},
            "Package details": {"numpy": "dist (channel)"},
        },
    )
    monkeypatch.setattr(
        "_benchmark.harness._try_version", lambda module_name, attr: "13.0.0"
    )
    monkeypatch.setattr(
        "_benchmark.harness._cupy_package_details",
        lambda: "cupy-13.0.0 (conda-forge)",
    )

    harness = BenchmarkHarness(
        _make_config(
            package=ArrayPackage.CUPY,
            short=False,
            cupy_allocator=CupyAllocator.DEFAULT,
        )
    )

    assert harness.metadata["Package versions"]["cupy"] == "13.0.0"
    assert (
        harness.metadata["Package details"]["cupy"]
        == "cupy-13.0.0 (conda-forge)"
    )
    # Pre-existing numpy entries should remain untouched.
    assert harness.metadata["Package versions"]["numpy"] == "x.y.z"


def test_harness_cupy_allocator_off_disables_cupy_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `CupyAllocator.OFF` should call `cupy.cuda.set_allocator(None)`,
    # disabling cupy's memory pool entirely.
    set_allocator = MagicMock()
    fake_cupy = SimpleNamespace(
        __name__="cupy", cuda=SimpleNamespace(set_allocator=set_allocator)
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    BenchmarkHarness(
        _make_config(
            package=ArrayPackage.CUPY,
            short=True,
            cupy_allocator=CupyAllocator.OFF,
        )
    )

    set_allocator.assert_called_once_with(None)


def test_harness_cupy_allocator_managed_installs_managed_pool_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `CupyAllocator.MANAGED` should build a managed-memory pool via
    # `cupy.cuda.MemoryPool(cupy.cuda.malloc_managed)` and install its
    # `.malloc` as the allocator.
    pool = MagicMock(name="memory_pool")
    MemoryPool = MagicMock(return_value=pool)
    set_allocator = MagicMock()
    malloc_managed = MagicMock(name="malloc_managed")

    fake_cupy = SimpleNamespace(
        __name__="cupy",
        cuda=SimpleNamespace(
            set_allocator=set_allocator,
            MemoryPool=MemoryPool,
            malloc_managed=malloc_managed,
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    BenchmarkHarness(
        _make_config(
            package=ArrayPackage.CUPY,
            short=True,
            cupy_allocator=CupyAllocator.MANAGED,
        )
    )

    MemoryPool.assert_called_once_with(malloc_managed)
    set_allocator.assert_called_once_with(pool.malloc)


def test_harness_cupy_allocator_default_does_not_call_set_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `CupyAllocator.DEFAULT` should leave cupy's allocator alone: neither
    # `set_allocator` nor `MemoryPool` should be touched.
    set_allocator = MagicMock(
        side_effect=AssertionError("set_allocator should not be called")
    )
    MemoryPool = MagicMock(
        side_effect=AssertionError("MemoryPool should not be called")
    )
    fake_cupy = SimpleNamespace(
        __name__="cupy",
        cuda=SimpleNamespace(
            set_allocator=set_allocator, MemoryPool=MemoryPool
        ),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    BenchmarkHarness(
        _make_config(
            package=ArrayPackage.CUPY,
            short=True,
            cupy_allocator=CupyAllocator.DEFAULT,
        )
    )

    set_allocator.assert_not_called()
    MemoryPool.assert_not_called()


def test_harness_log_metadata_extra_applied_to_metadata() -> None:
    harness = BenchmarkHarness(
        _make_config(short=True, log_metadata_extra={"a": "b"})
    )
    assert harness.metadata["a"] == "b"


def test_harness_log_conda_list_invokes_conda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "_benchmark.harness._conda_list", lambda: {"pkg": "1.0  build  ch"}
    )
    harness = BenchmarkHarness(_make_config(short=True, log_conda_list=True))
    assert harness.metadata["Conda list"] == {"pkg": "1.0  build  ch"}


# ---------------------------------------------------------------------------
# run / run_with_generator / run_timed (using BenchmarkLogNull when repeat=0)
# ---------------------------------------------------------------------------


def test_harness_run_calls_function_once_with_scalar_args() -> None:
    calls: list[tuple[Any, ...]] = []

    def f(a: int, b: int) -> None:
        calls.append((a, b))

    BenchmarkHarness(_make_config(repeat=0)).run(f, 1, 2)
    assert calls == [(1, 2)]


def test_harness_run_expands_list_args_to_cartesian_product() -> None:
    calls: list[tuple[Any, ...]] = []

    def f(a: int, b: int) -> None:
        calls.append((a, b))

    BenchmarkHarness(_make_config(repeat=0)).run(f, [1, 2], [3, 4])
    assert calls == [(1, 3), (1, 4), (2, 3), (2, 4)]


def test_harness_run_raises_on_positional_arg_count_mismatch() -> None:
    def f(a: int, b: int) -> None:
        pass

    harness = BenchmarkHarness(_make_config(repeat=0))
    with pytest.raises(RuntimeError, match="positional arguments, expected"):
        harness.run(f, 1, 2, 3)


def test_harness_run_with_generator_uses_provided_tuples() -> None:
    calls: list[tuple[Any, ...]] = []

    def f(a: int, b: int) -> None:
        calls.append((a, b))

    BenchmarkHarness(_make_config(repeat=0)).run_with_generator(
        None, f, [(1, 10), (2, 20)]
    )
    assert calls == [(1, 10), (2, 20)]


def test_harness_run_timed_external_when_function_does_not_return_time() -> (
    None
):
    # When `info.returns_time < 0` (the function does not return a runtime),
    # `run_timed` must use TIMED_EXTERNAL mode: it calls `timer.start()`
    # before `f`, then `timer.stop()` after, capturing wall-clock-like
    # duration externally instead of reading it from `f`'s return value.
    from _benchmark.info import benchmark_info
    from _benchmark.timer import SyncMode

    events: list[str] = []

    @benchmark_info(returns_time=False, output_names="result")
    def f(n: int) -> int:
        events.append(f"f({n})")
        return n * 2

    class RecordingTimer:
        def start(self) -> None:
            events.append("start")

        def stop(self) -> float:
            events.append("stop")
            return 1.23

        def sync(self, sync_mode: SyncMode) -> None:
            events.append(f"sync:{sync_mode}")

    harness = BenchmarkHarness(_make_config(repeat=0))
    harness.execution_timer = RecordingTimer()
    harness.run_timed(f, 5)

    # start -> f -> stop confirms TIMED_EXTERNAL ordering. TIMED_INTERNAL
    # would call neither start nor stop; UNTIMED would call f only.
    assert events == ["start", "f(5)", "stop"]


def test_harness_run_timed_calls_function_once_per_args() -> None:
    from _benchmark.info import benchmark_info

    calls: list[int] = []

    # returning None will test `output_vals is None` branch
    @benchmark_info(returns_time=False, output_names=())
    def f(n: int) -> None:
        calls.append(n)

    BenchmarkHarness(_make_config(repeat=0)).run_timed(f, [1, 2, 3])
    assert calls == [1, 2, 3]


def test_harness_run_timed_logs_renamed_input_and_named_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # input_names renames the "x" column to "X"; output_names supplies the
    # two output column names; returns_time=0 marks the first tuple element
    # ("time") as the runtime, so this exercises TIMED_INTERNAL with a
    # multi-column output. Spying on `BenchmarkLogNull` lets us verify the
    # exact kwargs (column names) passed to `b.log()`.
    from _benchmark.info import benchmark_info

    @benchmark_info(
        input_names={"x": "X"},
        output_names=["time", "result"],  # type: ignore[arg-type]
        returns_time=0,
    )
    def f(x: int) -> tuple[float, int]:
        return 1.5, x * 10

    log_calls: list[dict[str, Any]] = []

    class RecordingLog:
        def __enter__(self) -> RecordingLog:
            return self

        def __exit__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log(self, **kwargs: Any) -> None:
            log_calls.append(kwargs)

    monkeypatch.setattr("_benchmark.harness.BenchmarkLogNull", RecordingLog)

    BenchmarkHarness(_make_config(repeat=0)).run_timed(f, 7)

    assert log_calls == [{"X": 7, "time": 1.5, "result": 70}]


def test_harness_run_with_info_uses_supplied_info() -> None:
    format_calls: list[Any] = []

    def f(n: int) -> None:
        pass

    def fmt(v: Any) -> str:
        format_calls.append(v)
        return str(v)

    from _benchmark.info import get_benchmark_info

    # A custom `formats` entry only fires if `_run` consults the supplied
    # `info`; calling `fmt` proves the supplied info was actually used.
    info = get_benchmark_info(f).replace(formats={"n": fmt})
    BenchmarkHarness(_make_config(repeat=0)).run_with_info(info, f, 5)
    assert format_calls == [5]


# ---------------------------------------------------------------------------
# parse_with_harness
# ---------------------------------------------------------------------------


def test_harness_warns_when_issue_execution_fence_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The LEGATE branch in `_run` issues an execution fence between plan
    # iterations; if `issue_execution_fence` raises, the harness should
    # swallow the error and warn rather than propagate. Init with NUMPY for
    # a cheap, side-effect-free harness, then flip the config to LEGATE so
    # `_run` enters the fence branch.
    harness = BenchmarkHarness(_make_config(package=ArrayPackage.NUMPY))
    harness._config.package = ArrayPackage.LEGATE

    runtime = MagicMock()
    runtime.issue_execution_fence.side_effect = RuntimeError("boom")
    monkeypatch.setattr("legate.core.get_legate_runtime", lambda: runtime)

    def f(n: int) -> float:
        return 0.0

    with pytest.warns(UserWarning, match="issue_execution_fence failed.*boom"):
        harness.run_timed(f, 5)

    runtime.issue_execution_fence.assert_called_once_with(block=True)


def test_harness_summarize_flush_run_writes_and_flushes_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _benchmark.info import benchmark_info

    # `--summarize` makes from_args() build a Summarize. `--summarize-flush
    # run` makes the harness flush after every _run (not at __exit__). Spy
    # on Summarize to verify the harness calls write+flush per run.
    spy = MagicMock()
    monkeypatch.setattr(
        "_benchmark.harness.Summarize", MagicMock(return_value=spy)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--package",
            "legate",
            "--summarize",
            "--summarize-flush",
            "run",
        ],
    )

    _args, harness = parse_with_harness(ArgumentParser())
    assert harness.summarize is spy
    assert harness.summarize_flush == SummarizeFlush.RUN

    @benchmark_info(formats={"n": lambda n: f"{n:,}"})
    def f(n: int) -> float:
        return 0.0

    with harness:
        harness.run_timed_with_generator(None, f, [(1000,)])
        # `flush run` must flush immediately after each _run, before exit.
        assert spy.flush.call_count == 1
    # Exiting the context must not re-flush in `run` mode.
    assert spy.flush.call_count == 1
    spy.write.assert_called_once()
    spy.write.assert_called_with("f", {"n": "1,000"}, [0.0])
    spy.flush.assert_called_with(title="f")


def test_parse_with_harness_returns_args_and_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--package", "numpy"])
    args, harness = parse_with_harness(ArgumentParser())
    assert isinstance(args, Namespace)
    assert isinstance(harness, BenchmarkHarness)
    assert harness.package == "numpy"


def test_parse_with_harness_tolerates_unknown_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["prog", "--package", "numpy", "--unknown-flag", "value"]
    )
    _args, harness = parse_with_harness(ArgumentParser())
    assert harness.package == "numpy"


def test_parse_with_harness_parses_repeat_short_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "-b", "4", "--package", "numpy"])
    _args, harness = parse_with_harness(ArgumentParser())
    assert harness.repeat == 4


def test_parse_with_harness_preserves_user_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["prog", "--package", "numpy", "--user-arg", "myval"]
    )
    parser = ArgumentParser()
    parser.add_argument("--user-arg", type=str)
    args, _harness = parse_with_harness(parser)
    assert args.user_arg == "myval"


def test_parse_with_harness_adds_named_argument_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])
    parser = ArgumentParser()
    titles_before = [g.title for g in parser._action_groups]
    parse_with_harness(parser)
    titles_after = [g.title for g in parser._action_groups]
    assert len(titles_after) > len(titles_before)
    assert BenchmarkHarness.name in titles_after


def test_benchmark_harness_uses_legate_benchmark_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When LEGATE_BENCHMARK_OUT points at a directory, legate's
    # `benchmark_log` writes a CSV named `{name}_{uid:016x}.{node_id}.csv`
    # into that directory. Pointing it at a tmp_path lets us assert the
    # harness actually invoked legate's logger with `repeat=1`.
    monkeypatch.setenv("LEGATE_BENCHMARK_OUT", str(tmp_path))
    # The env-only setting caches on first read; clear the cache so the
    # new value is picked up.
    from legate.util._benchmark.settings import settings as benchmark_settings

    monkeypatch.setattr(benchmark_settings.out, "_cached", False)

    def f() -> float:
        return 1.0

    harness = BenchmarkHarness(_make_config(repeat=1, short=True))
    harness.run(f)

    matches = list(tmp_path.glob("f_*.0.csv"))
    assert len(matches) == 1


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
