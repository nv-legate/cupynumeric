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
from argparse import ArgumentParser, Namespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from _benchmark.harness import (
    ArrayPackage,
    BenchmarkHarness,
    CupyAllocator,
    SummarizeFlush,
    TimerMode,
)
from _benchmark.microbenchmark_info import (
    get_microbenchmark_info,
    microbenchmark,
)
from _benchmark.microbenchmark_utilities import (
    MicrobenchmarkConfig,
    MicrobenchmarkSuite,
)
from _benchmark.sizing import SizeRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> MicrobenchmarkConfig:
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
        "runs": 5,
        "warmup": 2,
        "include": None,
        "exclude": None,
        "verbosity": 0,
        "dry_run": False,
    }
    defaults.update(overrides)
    return MicrobenchmarkConfig(**defaults)


def _make_suite(**overrides: Any) -> MicrobenchmarkSuite:
    return MicrobenchmarkSuite(_make_config(**overrides), Namespace())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_suite_inherits_benchmark_harness() -> None:
    suite = _make_suite()
    assert isinstance(suite, BenchmarkHarness)
    assert suite.np is np


def test_suite_init_counters_start_at_zero() -> None:
    suite = _make_suite()
    assert suite.benchmark_count == 0
    assert suite.benchmark_variant_count == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_runs_property_proxies_config() -> None:
    assert _make_suite(runs=7).runs == 7


def test_warmup_property_proxies_config() -> None:
    assert _make_suite(warmup=4).warmup == 4


def test_dry_run_property_proxies_config() -> None:
    assert _make_suite(dry_run=True).dry_run is True


def test_check_run_delegates_to_config_include() -> None:
    suite = _make_suite(include=re.compile("keep"))
    assert suite.check_run("keep_this") is True
    assert suite.check_run("drop_this") is False


def test_check_run_delegates_to_config_exclude() -> None:
    suite = _make_suite(exclude=re.compile("skip"))
    assert suite.check_run("good_one") is True
    assert suite.check_run("skip_me") is False


# ---------------------------------------------------------------------------
# Extension-point defaults
# ---------------------------------------------------------------------------


def test_add_suite_parser_group_is_noop() -> None:
    parser = ArgumentParser()
    titles_before = [g.title for g in parser._action_groups]
    MicrobenchmarkSuite.add_suite_parser_group(parser)
    titles_after = [g.title for g in parser._action_groups]
    assert titles_after == titles_before


def test_print_config_default_is_noop() -> None:
    _make_suite().print_config()


def test_dtypes_default_is_empty() -> None:
    assert _make_suite().dtypes() == []


def test_default_arguments_contains_np_runs_warmup_timer() -> None:
    suite = _make_suite(runs=3, warmup=1)
    d = suite.default_arguments()
    assert d["np"] is suite.np
    assert d["runs"] == 3
    assert d["warmup"] == 1
    assert d["timer"] is suite.timer


# ---------------------------------------------------------------------------
# info / debug / print_panel delegate to config with self._console
# ---------------------------------------------------------------------------


def test_info_delegates_to_config_with_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _make_suite()
    spy = MagicMock()
    monkeypatch.setattr(suite._config, "info", spy)
    sentinel = object()
    suite._console = sentinel
    suite.info("hi")
    spy.assert_called_once_with("hi", console=sentinel)


def test_debug_delegates_to_config_with_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _make_suite()
    spy = MagicMock()
    monkeypatch.setattr(suite._config, "debug", spy)
    sentinel = object()
    suite._console = sentinel
    suite.debug("dbg")
    spy.assert_called_once_with("dbg", console=sentinel)


def test_print_panel_delegates_to_config_with_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _make_suite()
    spy = MagicMock()
    monkeypatch.setattr(suite._config, "print_panel", spy)
    sentinel = object()
    suite._console = sentinel
    suite.print_panel(["a"], title="T")
    spy.assert_called_once_with(["a"], "T", console=sentinel)


# ---------------------------------------------------------------------------
# print_suite_summary
# ---------------------------------------------------------------------------


def test_print_suite_summary_silent_when_count_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _make_suite()
    spy = MagicMock()
    monkeypatch.setattr(suite, "print_panel", spy)
    suite.print_suite_summary()
    spy.assert_not_called()


def test_print_suite_summary_count_equals_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _make_suite()
    suite.benchmark_count = 3
    suite.benchmark_variant_count = 3
    spy = MagicMock()
    monkeypatch.setattr(suite, "print_panel", spy)
    suite.print_suite_summary()
    spy.assert_called_once_with(
        ["Total benchmarks run: 3"], f"SUITE COMPLETE: {suite.name}"
    )


def test_print_suite_summary_with_variant_count_larger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _make_suite()
    suite.benchmark_count = 3
    suite.benchmark_variant_count = 7
    spy = MagicMock()
    monkeypatch.setattr(suite, "print_panel", spy)
    suite.print_suite_summary()
    spy.assert_called_once_with(
        ["Total benchmarks run: 3 (7 variants)"],
        f"SUITE COMPLETE: {suite.name}",
    )


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_enter_returns_self() -> None:
    suite = _make_suite()
    with suite as s:
        assert s is suite


def test_context_manager_calls_print_config_on_enter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _make_suite()
    spy = MagicMock()
    monkeypatch.setattr(suite, "print_config", spy)
    with suite:
        pass
    spy.assert_called_once_with()


def test_context_manager_calls_print_suite_summary_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _make_suite()
    spy = MagicMock()
    monkeypatch.setattr(suite, "print_suite_summary", spy)
    with suite:
        pass
    spy.assert_called_once_with()


# ---------------------------------------------------------------------------
# _plan_microbenchmark
# ---------------------------------------------------------------------------


def test_plan_microbenchmark_exact_size() -> None:
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, plan_kwargs = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[100])
    )
    assert plan_args == [(suite.np, 100)]
    assert plan_kwargs == {}


def test_plan_microbenchmark_multiple_exact_sizes() -> None:
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, _ = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[10, 20, 30])
    )
    assert plan_args == [(suite.np, 10), (suite.np, 20), (suite.np, 30)]


def test_plan_microbenchmark_memory_target_resolves_size() -> None:
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size * 8)
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, _ = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(memory_target_bytes=[1000])
    )
    # estimate_value(size) = size*8; largest size with 8*size <= 1000 is 125.
    assert plan_args == [(suite.np, 125)]


def test_plan_microbenchmark_rescale_by_work_produces_multiple_sizes() -> None:
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_arrays=lambda size: [("a", size)])
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, _ = suite._plan_microbenchmark(
        info,
        suite.bench,
        SizeRequest(exact_size=[100], rescale_by_work=[1.0, 2.0]),
    )
    # base size=100, base_work=100. rescale=1.0 -> size=100; rescale=2.0 ->
    # search for largest size with size <= 200 -> size=200.
    assert plan_args == [(suite.np, 100), (suite.np, 200)]


def test_plan_microbenchmark_exclude_filter_excludes_matched() -> None:
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(exclude=re.compile("bench")), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, _ = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[100])
    )
    assert plan_args == []


def test_plan_microbenchmark_include_filter_excludes_unmatched() -> None:
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(
        _make_config(include=re.compile("nothing_matches")), Namespace()
    )
    info = get_microbenchmark_info(suite.bench)
    plan_args, _ = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[100])
    )
    assert plan_args == []


def test_plan_microbenchmark_dtypes_expands_plan() -> None:
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        def dtypes(self) -> list[Any]:
            return ["float32", "float64"]

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int, dtype: Any) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, _ = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[100])
    )
    # One entry per dtype.
    assert plan_args == [
        (suite.np, 100, "float32"),
        (suite.np, 100, "float64"),
    ]


def test_plan_microbenchmark_size_injected_when_missing() -> None:
    # If "size" has no default, it should be auto-set to SIZE so the plan
    # uses the size_request's value directly.
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, _ = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[42])
    )
    assert plan_args[0][1] == 42


def test_plan_microbenchmark_should_skip_excludes_entry() -> None:
    # A callable ``skip`` that evaluates to True removes the matching plan
    # entry (the ``b_info.should_skip(...)`` branch in _plan_microbenchmark).
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(
            args_to_bytes=lambda size: size, skip=lambda size: True
        )
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, plan_kwargs = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[100])
    )
    assert plan_args == []
    assert plan_kwargs == {}


def test_plan_microbenchmark_should_skip_filters_subset_of_sizes() -> None:
    # ``should_skip`` is evaluated per plan entry, so only the sizes for
    # which it returns True are dropped.
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(
            args_to_bytes=lambda size: size, skip=lambda size: size > 15
        )
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    plan_args, _ = suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[10, 20, 30])
    )
    # 20 and 30 are skipped; only size=10 survives.
    assert plan_args == [(suite.np, 10)]


def test_plan_microbenchmark_explain_work_appears_in_debug_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A user-supplied ``explain_work`` is rendered into the debug stream
    # under a "work explanation" heading.
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(
            args_to_bytes=lambda size: size,
            explain_work=lambda size: [f"work is {size}"],
        )
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    messages: list[str] = []
    monkeypatch.setattr(suite, "debug", messages.append)
    suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[100])
    )
    debug_text = "\n".join(messages)
    assert "work explanation" in debug_text
    assert "work is 100" in debug_text


def test_plan_microbenchmark_no_explain_work_no_heading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without ``explain_work`` the "work explanation" heading is absent.
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    messages: list[str] = []
    monkeypatch.setattr(suite, "debug", messages.append)
    suite._plan_microbenchmark(
        info, suite.bench, SizeRequest(exact_size=[100])
    )
    assert "work explanation" not in "\n".join(messages)


def test_plan_microbenchmark_inconsistent_kwargs_raises() -> None:
    # plan entries that vary in keyword-only args are not allowed: the
    # harness invokes the benchmark via a generator that requires a single
    # kwargs dict across the whole run.
    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(
            args_to_bytes=lambda size: size,
            plan=[{"flag": "a"}, {"flag": "b"}],
        )
        def bench(np: Any, size: int, *, flag: str = "x") -> float:
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    info = get_microbenchmark_info(suite.bench)
    with pytest.raises(RuntimeError, match="keyword must be the same"):
        suite._plan_microbenchmark(
            info, suite.bench, SizeRequest(exact_size=[100])
        )


# ---------------------------------------------------------------------------
# run_suite
# ---------------------------------------------------------------------------


def test_run_suite_calls_decorated_methods() -> None:
    calls: list[int] = []

    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            calls.append(size)
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    suite.run_suite(SizeRequest(exact_size=[100]))
    assert calls == [100]
    assert suite.benchmark_count == 1
    assert suite.benchmark_variant_count == 1


def test_run_suite_increments_variant_count_per_plan_entry() -> None:
    calls: list[int] = []

    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            calls.append(size)
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    suite.run_suite(SizeRequest(exact_size=[10, 20, 30]))
    assert calls == [10, 20, 30]
    assert suite.benchmark_count == 1
    assert suite.benchmark_variant_count == 3


def test_run_suite_skips_when_microbenchmark_skip_is_true() -> None:
    calls: list[int] = []

    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(skip=True)
        def bench(np: Any, size: int) -> float:
            calls.append(size)
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    suite.run_suite(SizeRequest(exact_size=[100]))
    assert calls == []
    assert suite.benchmark_count == 0


def test_run_suite_skips_when_plan_is_empty_due_to_filter() -> None:
    calls: list[int] = []

    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            calls.append(size)
            return 1.0

    suite = _Suite(
        _make_config(include=re.compile("never_matches")), Namespace()
    )
    suite.run_suite(SizeRequest(exact_size=[100]))
    assert calls == []
    assert suite.benchmark_count == 0


def test_run_suite_dry_run_does_not_call_bench_or_increment() -> None:
    calls: list[int] = []

    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench(np: Any, size: int) -> float:
            calls.append(size)
            return 1.0

    suite = _Suite(_make_config(dry_run=True), Namespace())
    suite.run_suite(SizeRequest(exact_size=[100]))
    assert calls == []
    assert suite.benchmark_count == 0
    assert suite.benchmark_variant_count == 0


def test_run_suite_runs_multiple_benches_independently() -> None:
    calls: list[tuple[str, int]] = []

    class _Suite(MicrobenchmarkSuite):
        name = "test"

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench_a(np: Any, size: int) -> float:
            calls.append(("a", size))
            return 1.0

        @microbenchmark(args_to_bytes=lambda size: size)
        def bench_b(np: Any, size: int) -> float:
            calls.append(("b", size))
            return 1.0

    suite = _Suite(_make_config(), Namespace())
    suite.run_suite(SizeRequest(exact_size=[100]))
    # Order is alphabetical from inspect.getmembers.
    assert calls == [("a", 100), ("b", 100)]
    assert suite.benchmark_count == 2
    assert suite.benchmark_variant_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
