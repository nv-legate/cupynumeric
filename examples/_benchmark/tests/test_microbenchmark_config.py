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

import pytest

from _benchmark.harness import (
    ArrayPackage,
    CupyAllocator,
    SummarizeFlush,
    TimerMode,
)
from _benchmark.microbenchmark_utilities import MicrobenchmarkConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> MicrobenchmarkConfig:
    defaults: dict[str, Any] = {
        # BenchmarkHarnessConfig fields
        "repeat": 0,
        "package": ArrayPackage.NUMPY,
        "short": True,
        "cupy_allocator": CupyAllocator.DEFAULT,
        "log_conda_list": False,
        "log_metadata_extra": {},
        "summarize": None,
        "summarize_flush": SummarizeFlush.RUN,
        "timer_mode": TimerMode.EXECUTION,
        # MicrobenchmarkConfig fields
        "runs": 5,
        "warmup": 2,
        "include": None,
        "exclude": None,
        "verbosity": 0,
        "dry_run": False,
    }
    defaults.update(overrides)
    return MicrobenchmarkConfig(**defaults)


def _make_namespace(**overrides: Any) -> Namespace:
    defaults: dict[str, Any] = {
        # BenchmarkHarnessConfig args
        "__cpn_repeat": 0,
        "__cpn_package": ArrayPackage.NUMPY,
        "__cpn_short": True,
        "__cpn_summarize": False,
        "__cpn_summarize_flush": SummarizeFlush.RUN,
        "__cpn_cupy_allocator": CupyAllocator.DEFAULT,
        "__cpn_log_conda_list": False,
        "__cpn_metadata_extra": [],
        "__cpn_timer_mode": TimerMode.EXECUTION,
        # MicrobenchmarkConfig args
        "__cpn_runs": 5,
        "__cpn_warmup": 2,
        "__cpn_include": None,
        "__cpn_exclude": None,
        "__cpn_verbose": False,
        "__cpn_debug": False,
        "__cpn_dry_run": False,
    }
    defaults.update(overrides)
    ns = Namespace()
    for key, value in defaults.items():
        setattr(ns, key, value)
    return ns


def _parse(*argv: str) -> Namespace:
    parser = ArgumentParser()
    MicrobenchmarkConfig.add_parser_group(parser, "test")
    return parser.parse_args(list(argv))


# ---------------------------------------------------------------------------
# add_parser_group
# ---------------------------------------------------------------------------


def test_add_parser_group_returns_argument_group_with_title() -> None:
    parser = ArgumentParser()
    group = MicrobenchmarkConfig.add_parser_group(parser, "microbench")
    assert group.title == "microbench"


def test_add_parser_group_defaults() -> None:
    vargs = vars(_parse())
    assert vargs["__cpn_runs"] == 5
    assert vargs["__cpn_warmup"] == 2
    assert vargs["__cpn_include"] is None
    assert vargs["__cpn_exclude"] is None
    assert vargs["__cpn_verbose"] is False
    assert vargs["__cpn_debug"] is False
    assert vargs["__cpn_dry_run"] is False


def test_add_parser_group_runs() -> None:
    assert vars(_parse("--runs", "10"))["__cpn_runs"] == 10


def test_add_parser_group_warmup() -> None:
    assert vars(_parse("--warmup", "3"))["__cpn_warmup"] == 3


def test_add_parser_group_include() -> None:
    assert vars(_parse("--include", "foo"))["__cpn_include"] == "foo"


def test_add_parser_group_exclude() -> None:
    assert vars(_parse("--exclude", "bar"))["__cpn_exclude"] == "bar"


def test_add_parser_group_verbose_flag() -> None:
    assert vars(_parse("--verbose"))["__cpn_verbose"] is True


def test_add_parser_group_debug_flag() -> None:
    assert vars(_parse("--debug"))["__cpn_debug"] is True


def test_add_parser_group_dry_run_flag() -> None:
    assert vars(_parse("--dry-run"))["__cpn_dry_run"] is True


def test_add_parser_group_inherits_base_arguments() -> None:
    # `add_parser_group` chains to BenchmarkHarnessConfig, so harness flags
    # such as `--package` and `-b` must still be parseable.
    vargs = vars(_parse("--package", "numpy", "-b", "3"))
    assert vargs["__cpn_package"] == ArrayPackage.NUMPY
    assert vargs["__cpn_repeat"] == 3


# ---------------------------------------------------------------------------
# from_args
# ---------------------------------------------------------------------------


def test_from_args_returns_microbenchmark_config() -> None:
    config = MicrobenchmarkConfig.from_args(_make_namespace())
    assert isinstance(config, MicrobenchmarkConfig)


def test_from_args_defaults() -> None:
    config = MicrobenchmarkConfig.from_args(_make_namespace())
    assert config.runs == 5
    assert config.warmup == 2
    assert config.include is None
    assert config.exclude is None
    assert config.verbosity == 0
    assert config.dry_run is False


def test_from_args_runs_and_warmup_propagated() -> None:
    config = MicrobenchmarkConfig.from_args(
        _make_namespace(__cpn_runs=12, __cpn_warmup=4)
    )
    assert config.runs == 12
    assert config.warmup == 4


def test_from_args_dry_run_propagated() -> None:
    config = MicrobenchmarkConfig.from_args(
        _make_namespace(__cpn_dry_run=True)
    )
    assert config.dry_run is True


def test_from_args_include_compiled_to_regex() -> None:
    config = MicrobenchmarkConfig.from_args(
        _make_namespace(__cpn_include="foo.*bar")
    )
    assert isinstance(config.include, re.Pattern)
    assert config.include.pattern == "foo.*bar"


def test_from_args_exclude_compiled_to_regex() -> None:
    config = MicrobenchmarkConfig.from_args(
        _make_namespace(__cpn_exclude="skip_.*")
    )
    assert isinstance(config.exclude, re.Pattern)
    assert config.exclude.pattern == "skip_.*"


def test_from_args_verbose_only_yields_verbosity_one() -> None:
    config = MicrobenchmarkConfig.from_args(
        _make_namespace(__cpn_verbose=True)
    )
    assert config.verbosity == 1


def test_from_args_debug_yields_verbosity_two() -> None:
    config = MicrobenchmarkConfig.from_args(_make_namespace(__cpn_debug=True))
    assert config.verbosity == 2


def test_from_args_debug_overrides_verbose() -> None:
    # When both flags are set, --debug wins.
    config = MicrobenchmarkConfig.from_args(
        _make_namespace(__cpn_verbose=True, __cpn_debug=True)
    )
    assert config.verbosity == 2


def test_from_args_inherits_base_fields() -> None:
    config = MicrobenchmarkConfig.from_args(
        _make_namespace(
            __cpn_repeat=7,
            __cpn_package=ArrayPackage.NUMPY,
            __cpn_short=True,
            __cpn_timer_mode=TimerMode.WALL,
        )
    )
    assert config.repeat == 7
    assert config.package == ArrayPackage.NUMPY
    assert config.short is True
    assert config.timer_mode == TimerMode.WALL


# ---------------------------------------------------------------------------
# add_parser_group + from_args round trip
# ---------------------------------------------------------------------------


def test_parser_to_config_round_trip() -> None:
    parser = ArgumentParser()
    MicrobenchmarkConfig.add_parser_group(parser, "microbench")
    args = parser.parse_args(
        [
            "--package",
            "numpy",
            "--runs",
            "8",
            "--warmup",
            "1",
            "--include",
            "keep_.*",
            "--exclude",
            "skip_.*",
            "--debug",
            "--dry-run",
        ]
    )
    config = MicrobenchmarkConfig.from_args(args)
    assert config.runs == 8
    assert config.warmup == 1
    assert config.include is not None
    assert config.include.pattern == "keep_.*"
    assert config.exclude is not None
    assert config.exclude.pattern == "skip_.*"
    assert config.verbosity == 2
    assert config.dry_run is True
    assert config.package == ArrayPackage.NUMPY


# ---------------------------------------------------------------------------
# check_run
# ---------------------------------------------------------------------------


def test_check_run_no_filters_returns_true() -> None:
    assert _make_config().check_run("anything") is True


def test_check_run_include_match_returns_true() -> None:
    config = _make_config(include=re.compile("foo"))
    assert config.check_run("foobar") is True


def test_check_run_include_no_match_returns_false() -> None:
    config = _make_config(include=re.compile("foo"))
    assert config.check_run("bar") is False


def test_check_run_exclude_match_returns_false() -> None:
    config = _make_config(exclude=re.compile("bar"))
    assert config.check_run("foobar") is False


def test_check_run_exclude_no_match_returns_true() -> None:
    config = _make_config(exclude=re.compile("bar"))
    assert config.check_run("foo") is True


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # include "foo" matches, exclude "skip" does not -> allowed
        ("foo_keep", True),
        # include "foo" matches, but exclude "skip" also matches -> rejected
        ("skip_foo", False),
        # include "foo" does not match -> rejected (exclude irrelevant)
        ("bar", False),
        # include "foo" does not match; exclude matches too -> rejected
        ("skip_bar", False),
    ],
)
def test_check_run_with_both_filters_requires_include_and_not_exclude(
    name: str, expected: bool
) -> None:
    # With both filters set, `check_run` is True only when the name matches
    # `include` AND does not match `exclude`.
    config = _make_config(
        include=re.compile("foo"), exclude=re.compile("skip")
    )
    assert config.check_run(name) is expected


def test_check_run_both_filters_overlapping_pattern_rejects_match() -> None:
    # When include and exclude are the same pattern, every match is also
    # excluded, so no name can be allowed.
    config = _make_config(include=re.compile("foo"), exclude=re.compile("foo"))
    assert config.check_run("foo") is False
    assert config.check_run("foobar") is False
    # A name that misses include is rejected regardless of exclude.
    assert config.check_run("bar") is False


# ---------------------------------------------------------------------------
# info / debug / _print_msg
# ---------------------------------------------------------------------------


@pytest.fixture
def _force_plain_print(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the non-rich branch of `_print_msg` so output goes through
    # the built-in `print` and is captured by capsys.
    monkeypatch.setattr("_benchmark.microbenchmark_utilities.HAVE_RICH", False)


@pytest.mark.parametrize("verbosity", [0])
def test_info_silent_at_verbosity_zero(
    verbosity: int,
    capsys: pytest.CaptureFixture[str],
    _force_plain_print: None,
) -> None:
    _make_config(verbosity=verbosity).info("hello")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("verbosity", [1, 2])
def test_info_prints_at_verbosity_one_or_higher(
    verbosity: int,
    capsys: pytest.CaptureFixture[str],
    _force_plain_print: None,
) -> None:
    _make_config(verbosity=verbosity).info("hello")
    assert "hello" in capsys.readouterr().out


@pytest.mark.parametrize("verbosity", [0, 1])
def test_debug_silent_below_verbosity_two(
    verbosity: int,
    capsys: pytest.CaptureFixture[str],
    _force_plain_print: None,
) -> None:
    _make_config(verbosity=verbosity).debug("hello")
    assert capsys.readouterr().out == ""


def test_debug_prints_at_verbosity_two(
    capsys: pytest.CaptureFixture[str], _force_plain_print: None
) -> None:
    _make_config(verbosity=2).debug("hello")
    assert "hello" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# print_panel
# ---------------------------------------------------------------------------


def test_print_panel_with_title_renders_title_and_separators(
    capsys: pytest.CaptureFixture[str], _force_plain_print: None
) -> None:
    _make_config().print_panel(["line one", "line two"], title="SUMMARY")
    out = capsys.readouterr().out
    assert "SUMMARY" in out
    assert "line one" in out
    assert "line two" in out
    # Title separator block uses "=" * 80 and a "-" * 80 divider.
    assert "=" * 80 in out
    assert "-" * 80 in out


def test_print_panel_without_title_omits_title_line(
    capsys: pytest.CaptureFixture[str], _force_plain_print: None
) -> None:
    _make_config().print_panel(["just one"])
    out = capsys.readouterr().out
    assert "just one" in out
    assert "=" * 80 in out


def test_print_panel_ignores_verbosity(
    capsys: pytest.CaptureFixture[str], _force_plain_print: None
) -> None:
    # Unlike info/debug, print_panel always emits regardless of verbosity.
    _make_config(verbosity=0).print_panel(["always shown"])
    assert "always shown" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _print_msg / print_panel with rich
# ---------------------------------------------------------------------------


@pytest.fixture
def _force_rich_print(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the rich branch of `_print_msg` and `print_panel`. Skips the
    # test if rich is not installed in this environment.
    pytest.importorskip("rich")
    monkeypatch.setattr("_benchmark.microbenchmark_utilities.HAVE_RICH", True)
    monkeypatch.setattr(
        "_benchmark.microbenchmark_utilities.use_rich",
        lambda *args, **kwargs: True,
    )


def test_info_rich_auto_creates_console_writing_to_stdout(
    capsys: pytest.CaptureFixture[str], _force_rich_print: None
) -> None:
    # With no console supplied, `_print_msg` builds a rich Console targeting
    # sys.stdout, so the rendered output is captured by capsys.
    _make_config(package=ArrayPackage.LEGATE, verbosity=1).info("hello-rich")
    assert "hello-rich" in capsys.readouterr().out


def test_info_rich_uses_supplied_console(_force_rich_print: None) -> None:
    # A user-supplied rich Console must pass the isinstance check in
    # `_print_msg` and receive the message via `.print()`.
    import io

    from rich.console import Console

    buf = io.StringIO()
    user_console = Console(file=buf, force_terminal=False)
    _make_config(verbosity=1).info("via-user-console", console=user_console)
    assert "via-user-console" in buf.getvalue()


def test_print_panel_rich_uses_supplied_console(
    _force_rich_print: None,
) -> None:
    # A user-supplied rich Console must pass the isinstance check in
    # `print_panel` and receive the message via `.print()`.
    import io

    from rich.console import Console

    buf = io.StringIO()
    user_console = Console(file=buf, force_terminal=False)
    _make_config(package=ArrayPackage.LEGATE, verbosity=1).print_panel(
        ["via-user-console"], console=user_console
    )
    assert "via-user-console" in buf.getvalue()


def test_debug_rich_prints_at_verbosity_two(
    capsys: pytest.CaptureFixture[str], _force_rich_print: None
) -> None:
    _make_config(verbosity=2).debug("debug-rich")
    assert "debug-rich" in capsys.readouterr().out


def test_print_panel_rich_with_title_renders_title_and_lines(
    capsys: pytest.CaptureFixture[str], _force_rich_print: None
) -> None:
    # rich's Panel.fit renders a box-drawn panel; title and content lines
    # all appear as substrings in the captured output.
    _make_config().print_panel(["line one", "line two"], title="HEAD")
    out = capsys.readouterr().out
    assert "HEAD" in out
    assert "line one" in out
    assert "line two" in out
    # The plain-print "=" * 80 / "-" * 80 separators must NOT appear: this
    # confirms we took the rich branch rather than the fallback.
    assert "=" * 80 not in out
    assert "-" * 80 not in out


def test_print_panel_rich_without_title(
    capsys: pytest.CaptureFixture[str], _force_rich_print: None
) -> None:
    _make_config().print_panel(["only line"])
    out = capsys.readouterr().out
    assert "only line" in out


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
