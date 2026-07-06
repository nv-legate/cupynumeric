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

import io
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from _benchmark import summarize as summarize_mod
from _benchmark.info import _TIME
from _benchmark.summarize import Summarize, Summary


# ---------------------------------------------------------------------------
# Summary dataclass
# ---------------------------------------------------------------------------


def test_summary_holds_fields() -> None:
    s = Summary(
        id_string="foo",
        num_samples=3,
        minimum=1.0,
        maximum=5.0,
        mean=3.0,
        standard_deviation=1.5,
    )
    assert s.id_string == "foo"
    assert s.num_samples == 3
    assert s.minimum == 1.0
    assert s.maximum == 5.0
    assert s.mean == 3.0
    assert s.standard_deviation == 1.5


# ---------------------------------------------------------------------------
# Summarize.__init__
# ---------------------------------------------------------------------------


def test_init_default_summary_column_name() -> None:
    s = Summarize(io.StringIO())
    assert s.summary_column_name == _TIME


def test_init_custom_summary_column_name() -> None:
    s = Summarize(io.StringIO(), summary_column_name="flops")
    assert s.summary_column_name == "flops"


def test_init_starts_with_empty_data() -> None:
    s = Summarize(io.StringIO())
    assert s.data == {}


def test_init_stores_stream() -> None:
    stream = io.StringIO()
    s = Summarize(stream)
    assert s.stream is stream


def test_init_use_rich_false_for_non_tty_stream() -> None:
    # io.StringIO().isatty() returns False, so use_rich(out) is False.
    s = Summarize(io.StringIO())
    assert s.use_rich is False
    # No Console attribute is created in the non-rich branch.
    assert not hasattr(s, "console")


def test_init_creates_console_when_use_rich_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_console = MagicMock(name="Console")
    monkeypatch.setattr(
        summarize_mod, "use_rich", lambda *_a, **_k: True, raising=True
    )
    monkeypatch.setattr(
        summarize_mod, "Console", lambda **kw: fake_console, raising=True
    )
    stream = io.StringIO()
    s = Summarize(stream)
    assert s.use_rich is True
    assert s.console is fake_console


# ---------------------------------------------------------------------------
# Summarize.write
# ---------------------------------------------------------------------------


def test_write_creates_entry_under_new_name() -> None:
    s = Summarize(io.StringIO())
    s.write("bench", {"n": 1}, [1.0, 2.0])
    assert "bench" in s.data
    assert s.data["bench"] == [({"n": 1}, [1.0, 2.0])]


def test_write_appends_to_existing_name() -> None:
    s = Summarize(io.StringIO())
    s.write("bench", {"n": 1}, [1.0])
    s.write("bench", {"n": 2}, [3.0, 4.0])
    assert s.data["bench"] == [({"n": 1}, [1.0]), ({"n": 2}, [3.0, 4.0])]


def test_write_keeps_names_separate() -> None:
    s = Summarize(io.StringIO())
    s.write("a", {}, [1.0])
    s.write("b", {}, [2.0])
    assert set(s.data.keys()) == {"a", "b"}
    assert s.data["a"] == [({}, [1.0])]
    assert s.data["b"] == [({}, [2.0])]


# ---------------------------------------------------------------------------
# Summarize.flush — empty/skip paths
# ---------------------------------------------------------------------------


def test_flush_on_empty_data_writes_nothing() -> None:
    stream = io.StringIO()
    Summarize(stream).flush()
    assert stream.getvalue() == ""


def test_flush_skips_entries_with_no_times() -> None:
    stream = io.StringIO()
    s = Summarize(stream)
    s.write("bench", {}, [])  # empty -> skipped
    s.flush()
    # With no surviving summaries, nothing is written.
    assert stream.getvalue() == ""


def test_flush_clears_data() -> None:
    s = Summarize(io.StringIO())
    s.write("bench", {}, [1.0])
    s.flush()
    assert s.data == {}


def test_flush_clears_data_even_when_all_entries_skipped() -> None:
    s = Summarize(io.StringIO())
    s.write("bench", {}, [])
    s.flush()
    assert s.data == {}


# ---------------------------------------------------------------------------
# Summarize.flush — single-sample text rendering
# ---------------------------------------------------------------------------


def test_flush_single_sample_renders_two_column_table() -> None:
    stream = io.StringIO()
    s = Summarize(stream)
    s.write("bench", {}, [1.5])
    s.flush()
    output = stream.getvalue()
    assert "SUMMARY" in output
    assert "benchmark" in output
    assert _TIME in output
    assert "1.5" in output


@pytest.mark.parametrize(
    "times",
    [
        pytest.param([1.5], id="single-sample"),
        pytest.param([1.0, 2.0, 3.0], id="multi-sample"),
    ],
)
def test_flush_title_appears_in_header(times: list[float]) -> None:
    stream = io.StringIO()
    s = Summarize(stream)
    s.write("bench", {}, times)
    s.flush(title="my run")
    output = stream.getvalue()
    assert "SUMMARY: my run" in output


@pytest.mark.parametrize(
    "times",
    [
        pytest.param([1.5], id="single-sample"),
        pytest.param([1.0, 2.0, 3.0], id="multi-sample"),
    ],
)
def test_flush_without_title_uses_plain_summary_header(
    times: list[float],
) -> None:
    stream = io.StringIO()
    s = Summarize(stream)
    s.write("bench", {}, times)
    s.flush()
    output = stream.getvalue()
    assert "SUMMARY\n" in output
    assert "SUMMARY:" not in output


def test_flush_uses_custom_summary_column_name() -> None:
    stream = io.StringIO()
    s = Summarize(stream, summary_column_name="flops")
    s.write("bench", {}, [42.0])
    s.flush()
    assert "flops" in stream.getvalue()


# ---------------------------------------------------------------------------
# Summarize.flush — multi-sample text rendering
# ---------------------------------------------------------------------------


def test_flush_multi_sample_renders_full_table() -> None:
    stream = io.StringIO()
    s = Summarize(stream)
    s.write("bench", {}, [1.0, 2.0, 3.0, 4.0])
    s.flush()
    output = stream.getvalue()
    assert "# samples" in output
    assert "min" in output
    assert "max" in output
    assert "mean" in output
    assert "std. dev." in output


def test_flush_max_n_drives_full_table_for_mixed_entries() -> None:
    stream = io.StringIO()
    s = Summarize(stream)
    s.write("a", {}, [1.0])
    s.write("b", {}, [2.0, 3.0])
    s.flush()
    # max_n == 2 -> full table layout
    assert "# samples" in stream.getvalue()


# ---------------------------------------------------------------------------
# Summarize.flush — statistics
# ---------------------------------------------------------------------------


def _flush_and_get_summary(s: Summarize) -> list[Summary]:
    captured: list[Summary] = []
    original = summarize_mod.Summary

    def capture(*args: Any, **kwargs: Any) -> Summary:
        summary = original(*args, **kwargs)
        captured.append(summary)
        return summary

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(summarize_mod, "Summary", capture)
        s.flush()
    return captured


def test_flush_statistics_n_eq_1() -> None:
    s = Summarize(io.StringIO())
    s.write("bench", {}, [7.0])
    summaries = _flush_and_get_summary(s)
    assert len(summaries) == 1
    [summary] = summaries
    assert summary.num_samples == 1
    assert summary.minimum == 7.0
    assert summary.maximum == 7.0
    assert summary.mean == 7.0
    assert summary.standard_deviation == 0.0


def test_flush_statistics_n_eq_2_drops_only_min() -> None:
    # n=2: samples = [1.0, 3.0]; the min (1.0) is removed,
    # leaving [3.0]; mean=3.0, stddev=0.
    s = Summarize(io.StringIO())
    s.write("bench", {}, [1.0, 3.0])
    summaries = _flush_and_get_summary(s)
    [summary] = summaries
    assert summary.num_samples == 2
    assert summary.minimum == 1.0
    assert summary.maximum == 3.0
    assert summary.mean == pytest.approx(3.0)
    assert summary.standard_deviation == pytest.approx(0.0)


def test_flush_statistics_n_eq_3_drops_min_and_max() -> None:
    # n=3: [1, 5, 10] -> drop 10 then 1 -> [5] -> mean=5, stddev=0
    s = Summarize(io.StringIO())
    s.write("bench", {}, [1.0, 5.0, 10.0])
    summaries = _flush_and_get_summary(s)
    [summary] = summaries
    assert summary.minimum == 1.0
    assert summary.maximum == 10.0
    assert summary.mean == pytest.approx(5.0)
    assert summary.standard_deviation == pytest.approx(0.0)


def test_flush_statistics_n_eq_4_uses_middle_samples() -> None:
    # n=4: [1, 2, 3, 4] -> drop 4 then 1 -> [2, 3]
    # mean=2.5; variance=((2-2.5)^2 + (3-2.5)^2)/2 = 0.25; stddev=0.5
    s = Summarize(io.StringIO())
    s.write("bench", {}, [1.0, 2.0, 3.0, 4.0])
    summaries = _flush_and_get_summary(s)
    [summary] = summaries
    assert summary.num_samples == 4
    assert summary.minimum == 1.0
    assert summary.maximum == 4.0
    assert summary.mean == pytest.approx(2.5)
    assert summary.standard_deviation == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Summarize.flush — id_string / varying args
# ---------------------------------------------------------------------------


def test_flush_id_string_omits_non_varying_args() -> None:
    s = Summarize(io.StringIO())
    s.write("bench", {"n": 10, "dtype": "float32"}, [1.0])
    s.write("bench", {"n": 10, "dtype": "float32"}, [2.0])
    summaries = _flush_and_get_summary(s)
    assert all(summary.id_string == "bench" for summary in summaries)


def test_flush_id_string_includes_varying_args() -> None:
    s = Summarize(io.StringIO())
    s.write("bench", {"n": 10, "dtype": "float32"}, [1.0])
    s.write("bench", {"n": 20, "dtype": "float32"}, [2.0])
    summaries = _flush_and_get_summary(s)
    ids = [summary.id_string for summary in summaries]
    # "dtype" is constant, "n" varies; only "n" should appear.
    assert "bench::(n=10)" in ids
    assert "bench::(n=20)" in ids
    for sid in ids:
        assert "dtype" not in sid


def test_flush_keeps_each_benchmark_name_separate() -> None:
    s = Summarize(io.StringIO())
    s.write("alpha", {}, [1.0])
    s.write("beta", {}, [2.0])
    summaries = _flush_and_get_summary(s)
    ids = {summary.id_string for summary in summaries}
    assert ids == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Summarize.flush — rich rendering
# ---------------------------------------------------------------------------


def _enable_rich(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, MagicMock]:
    """Force the rich branch and return the (Console, Table) mocks."""
    fake_console = MagicMock(name="Console")
    fake_table_cls = MagicMock(name="Table")
    monkeypatch.setattr(
        summarize_mod, "use_rich", lambda *_a, **_k: True, raising=True
    )
    monkeypatch.setattr(
        summarize_mod, "Console", lambda **kw: fake_console, raising=True
    )
    monkeypatch.setattr(summarize_mod, "Table", fake_table_cls, raising=True)
    return fake_console, fake_table_cls


def test_flush_rich_single_sample_uses_two_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console, Table = _enable_rich(monkeypatch)
    s = Summarize(io.StringIO())
    s.write("bench", {}, [1.5])
    s.flush()
    table = Table.return_value
    column_names = [call.args[0] for call in table.add_column.call_args_list]
    assert column_names == ["benchmark", _TIME]
    table.add_row.assert_called_once()
    console.print.assert_called_once_with(table)


def test_flush_rich_multi_sample_uses_full_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, Table = _enable_rich(monkeypatch)
    s = Summarize(io.StringIO())
    s.write("bench", {}, [1.0, 2.0, 3.0])
    s.flush()
    table = Table.return_value
    column_names = [call.args[0] for call in table.add_column.call_args_list]
    assert column_names == [
        "benchmark",
        "# samples",
        "min (ms)",
        "max (ms)",
        "mean ± std. dev. (ms)",
    ]


def test_flush_rich_title_is_passed_to_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, Table = _enable_rich(monkeypatch)
    s = Summarize(io.StringIO())
    s.write("bench", {}, [1.0])
    s.flush(title="my run")
    title_arg = Table.call_args.kwargs["title"]
    assert "my run" in title_arg


def test_flush_rich_without_title_uses_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, Table = _enable_rich(monkeypatch)
    s = Summarize(io.StringIO())
    s.write("bench", {}, [1.0])
    s.flush()
    title_arg = Table.call_args.kwargs["title"]
    assert "Summary" in title_arg


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
