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
from typing import Any

import pytest

from _benchmark.harness import (
    ArrayPackage,
    BenchmarkHarnessConfig,
    CupyAllocator,
    Summarize,
    SummarizeFlush,
    TimerMode,
)


# ---------------------------------------------------------------------------
# add_parser_group
# ---------------------------------------------------------------------------


def _parse(*argv: str) -> Namespace:
    parser = ArgumentParser()
    BenchmarkHarnessConfig.add_parser_group(parser, "test")
    return parser.parse_args(list(argv))


def test_add_parser_group_returns_argument_group_with_title() -> None:
    parser = ArgumentParser()
    group = BenchmarkHarnessConfig.add_parser_group(parser, "harness")
    assert group.title == "harness"


def test_add_parser_group_defaults() -> None:
    vargs = vars(_parse())
    assert vargs["__cpn_repeat"] == 0
    assert vargs["__cpn_package"] == ArrayPackage.LEGATE
    assert vargs["__cpn_short"] is False
    assert vargs["__cpn_summarize"] is False
    assert vargs["__cpn_summarize_flush"] == SummarizeFlush.RUN
    assert vargs["__cpn_cupy_allocator"] == CupyAllocator.DEFAULT
    assert vargs["__cpn_log_conda_list"] is False
    assert vargs["__cpn_metadata_extra"] == []
    assert vargs["__cpn_timer_mode"] == TimerMode.EXECUTION


def test_add_parser_group_benchmark_short_flag() -> None:
    assert vars(_parse("-b", "5"))["__cpn_repeat"] == 5


def test_add_parser_group_benchmark_long_flag() -> None:
    assert vars(_parse("--benchmark", "3"))["__cpn_repeat"] == 3


@pytest.mark.parametrize(
    "value,expected",
    [
        ("numpy", ArrayPackage.NUMPY),
        ("cupy", ArrayPackage.CUPY),
        ("legate", ArrayPackage.LEGATE),
    ],
)
def test_add_parser_group_package_choices(
    value: str, expected: ArrayPackage
) -> None:
    assert vars(_parse("--package", value))["__cpn_package"] == expected


def test_add_parser_group_invalid_package_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse("--package", "jax")


def test_add_parser_group_short_flag() -> None:
    assert vars(_parse("--short"))["__cpn_short"] is True


def test_add_parser_group_summarize_flag() -> None:
    assert vars(_parse("--summarize"))["__cpn_summarize"] is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("run", SummarizeFlush.RUN),
        ("exit", SummarizeFlush.EXIT),
        ("never", SummarizeFlush.NEVER),
    ],
)
def test_add_parser_group_summarize_flush_choices(
    value: str, expected: SummarizeFlush
) -> None:
    assert (
        vars(_parse("--summarize-flush", value))["__cpn_summarize_flush"]
        == expected
    )


def test_add_parser_group_invalid_summarize_flush_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse("--summarize-flush", "nope")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("default", CupyAllocator.DEFAULT),
        ("managed", CupyAllocator.MANAGED),
        ("off", CupyAllocator.OFF),
    ],
)
def test_add_parser_group_cupy_allocator_choices(
    value: str, expected: CupyAllocator
) -> None:
    assert (
        vars(_parse("--cupy-allocator", value))["__cpn_cupy_allocator"]
        == expected
    )


def test_add_parser_group_invalid_cupy_allocator_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse("--cupy-allocator", "bogus")


def test_add_parser_group_log_conda_list() -> None:
    assert vars(_parse("--log-conda-list"))["__cpn_log_conda_list"] is True


def test_add_parser_group_log_metadata_extra_pairs() -> None:
    vargs = vars(_parse("--log-metadata-extra", "a=1", "b=2"))
    assert vargs["__cpn_metadata_extra"] == [("a", "1"), ("b", "2")]


def test_add_parser_group_log_metadata_extra_invalid_pair() -> None:
    with pytest.raises(RuntimeError, match="expected 'key=value' pair"):
        _parse("--log-metadata-extra", "missingvalue")


@pytest.mark.parametrize(
    "value,expected",
    [("execution", TimerMode.EXECUTION), ("wall", TimerMode.WALL)],
)
def test_add_parser_group_timer_mode_choices(
    value: str, expected: TimerMode
) -> None:
    assert vars(_parse("--timer-mode", value))["__cpn_timer_mode"] == expected


def test_add_parser_group_invalid_timer_mode_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse("--timer-mode", "cpu")


# ---------------------------------------------------------------------------
# from_args
# ---------------------------------------------------------------------------


def _namespace(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = {
        "__cpn_repeat": 0,
        "__cpn_package": ArrayPackage.NUMPY,
        "__cpn_short": False,
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


def test_from_args_returns_config() -> None:
    config = BenchmarkHarnessConfig.from_args(_namespace())
    assert isinstance(config, BenchmarkHarnessConfig)


def test_from_args_summarize_false_yields_none() -> None:
    config = BenchmarkHarnessConfig.from_args(_namespace())
    assert config.summarize is None


def test_from_args_summarize_true_creates_summarize() -> None:
    # Use NUMPY package to avoid the LEGATE import branch.
    config = BenchmarkHarnessConfig.from_args(
        _namespace(__cpn_summarize=True, __cpn_package=ArrayPackage.NUMPY)
    )
    assert isinstance(config.summarize, Summarize)


def test_from_args_propagates_simple_fields() -> None:
    config = BenchmarkHarnessConfig.from_args(
        _namespace(
            __cpn_repeat=4,
            __cpn_package=ArrayPackage.CUPY,
            __cpn_short=True,
            __cpn_cupy_allocator=CupyAllocator.OFF,
            __cpn_log_conda_list=True,
            __cpn_summarize_flush=SummarizeFlush.NEVER,
            __cpn_timer_mode=TimerMode.WALL,
        )
    )
    assert config.repeat == 4
    assert config.package == ArrayPackage.CUPY
    assert config.short is True
    assert config.cupy_allocator == CupyAllocator.OFF
    assert config.log_conda_list is True
    assert config.summarize_flush == SummarizeFlush.NEVER
    assert config.timer_mode == TimerMode.WALL


def test_from_args_log_metadata_extra_converted_to_dict() -> None:
    config = BenchmarkHarnessConfig.from_args(
        _namespace(__cpn_metadata_extra=[("a", "1"), ("b", "2")])
    )
    assert config.log_metadata_extra == {"a": "1", "b": "2"}


def test_from_args_log_metadata_extra_empty_dict() -> None:
    config = BenchmarkHarnessConfig.from_args(_namespace())
    assert config.log_metadata_extra == {}


# ---------------------------------------------------------------------------
# add_parser_group + from_args round trip
# ---------------------------------------------------------------------------


def test_parser_to_config_round_trip() -> None:
    parser = ArgumentParser()
    BenchmarkHarnessConfig.add_parser_group(parser, "harness")
    args = parser.parse_args(
        [
            "-b",
            "2",
            "--package",
            "numpy",
            "--short",
            "--summarize-flush",
            "exit",
            "--cupy-allocator",
            "managed",
            "--log-conda-list",
            "--log-metadata-extra",
            "k1=v1",
            "k2=v2",
            "--timer-mode",
            "wall",
        ]
    )
    config = BenchmarkHarnessConfig.from_args(args)
    assert config.repeat == 2
    assert config.package == ArrayPackage.NUMPY
    assert config.short is True
    assert config.summarize is None
    assert config.summarize_flush == SummarizeFlush.EXIT
    assert config.cupy_allocator == CupyAllocator.MANAGED
    assert config.log_conda_list is True
    assert config.log_metadata_extra == {"k1": "v1", "k2": "v2"}
    assert config.timer_mode == TimerMode.WALL


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "repeat": 1,
        "package": ArrayPackage.NUMPY,
        "short": False,
        "cupy_allocator": CupyAllocator.DEFAULT,
        "log_conda_list": False,
        "log_metadata_extra": {},
        "summarize": None,
        "summarize_flush": SummarizeFlush.RUN,
        "timer_mode": TimerMode.EXECUTION,
    }
    defaults.update(overrides)
    return BenchmarkHarnessConfig(**defaults)


def test_to_dict_contains_all_fields() -> None:
    result = _make_config().to_dict()
    assert set(result.keys()) == {
        "repeat",
        "package",
        "short",
        "cupy_allocator",
        "log_conda_list",
        "log_metadata_extra",
        "summarize",
        "summarize_flush",
        "timer_mode",
    }


def test_to_dict_values_match_attributes() -> None:
    config = _make_config(
        repeat=7,
        package=ArrayPackage.CUPY,
        short=True,
        cupy_allocator=CupyAllocator.MANAGED,
        log_conda_list=True,
        log_metadata_extra={"a": "b"},
        summarize_flush=SummarizeFlush.EXIT,
        timer_mode=TimerMode.WALL,
    )
    result = config.to_dict()
    assert result["repeat"] == 7
    assert result["package"] == ArrayPackage.CUPY
    assert result["short"] is True
    assert result["cupy_allocator"] == CupyAllocator.MANAGED
    assert result["log_conda_list"] is True
    assert result["log_metadata_extra"] == {"a": "b"}
    assert result["summarize"] is None
    assert result["summarize_flush"] == SummarizeFlush.EXIT
    assert result["timer_mode"] == TimerMode.WALL


def test_to_dict_is_shallow_copy() -> None:
    extra = {"key": "value"}
    config = _make_config(log_metadata_extra=extra)
    result = config.to_dict()
    # New top-level dict, but inner mutable values are shared by reference.
    assert result is not config.__dict__
    assert result["log_metadata_extra"] is extra


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
