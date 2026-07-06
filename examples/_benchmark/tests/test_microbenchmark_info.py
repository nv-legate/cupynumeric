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
from typing import Any
from unittest.mock import MagicMock

import pytest

from _benchmark.info import INFO, BenchmarkInfo
from _benchmark.microbenchmark_info import (
    MISSING,
    SIZE,
    MicrobenchmarkInfo,
    _explain_ndarray_bytes,
    _ndarray_bytes,
    get_microbenchmark_info,
    microbenchmark,
)


def _make_info(**overrides: Any) -> MicrobenchmarkInfo:
    defaults: dict[str, Any] = {
        "name": "bench",
        "input_names": {},
        "output_names": "time per run (ms)",
        "formats": {},
        "returns_time": 0,
        "size_to_args": None,
        "args_to_bytes": None,
        "explain_bytes": None,
        "args_to_arrays": None,
        "args_to_work": None,
        "explain_work": None,
        "plan": None,
        "skip": False,
    }
    defaults.update(overrides)
    return MicrobenchmarkInfo(**defaults)


# ---------------------------------------------------------------------------
# _ndarray_bytes
# ---------------------------------------------------------------------------


def test_ndarray_bytes_int_shape_default_dtype() -> None:
    # 10 elements of float64 (8 bytes) = 80 bytes
    assert _ndarray_bytes(10) == 80


def test_ndarray_bytes_tuple_shape() -> None:
    assert _ndarray_bytes((3, 4)) == 3 * 4 * 8


@pytest.mark.parametrize(
    ("shape", "dtype", "expected"),
    [
        (5, "int32", 5 * 4),
        ((2, 3), "float32", 2 * 3 * 4),
        (4, "int8", 4),
        ((10,), "complex128", 10 * 16),
    ],
)
def test_ndarray_bytes_explicit_dtype(
    shape: int | tuple[int, ...], dtype: str, expected: int
) -> None:
    assert _ndarray_bytes(shape, dtype=dtype) == expected


def test_ndarray_bytes_empty_tuple_returns_itemsize() -> None:
    # math.prod(()) == 1; default float64 itemsize is 8.
    assert _ndarray_bytes(()) == 8


# ---------------------------------------------------------------------------
# _explain_ndarray_bytes
# ---------------------------------------------------------------------------


def test_explain_ndarray_bytes_int_shape() -> None:
    assert (
        _explain_ndarray_bytes("a", 10)
        == "a: 10 float64s x 8 bytes / float64 = 80 bytes"
    )


def test_explain_ndarray_bytes_int_shape_uses_thousands_separator() -> None:
    text = _explain_ndarray_bytes("a", 1_000_000)
    assert "1,000,000 float64s" in text
    assert "8,000,000 bytes" in text


def test_explain_ndarray_bytes_tuple_shape() -> None:
    assert (
        _explain_ndarray_bytes("m", (3, 4))
        == "m: (3 x 4 float64s) x 8 bytes / float64 = 96 bytes"
    )


def test_explain_ndarray_bytes_explicit_dtype() -> None:
    assert (
        _explain_ndarray_bytes("v", 4, in_dtype="int32")
        == "v: 4 int32s x 4 bytes / int32 = 16 bytes"
    )


# ---------------------------------------------------------------------------
# SIZE / MISSING sentinels
# ---------------------------------------------------------------------------


def test_size_repr() -> None:
    assert repr(SIZE) == "SIZE"


def test_missing_repr() -> None:
    assert repr(MISSING) == "MISSING"


# ---------------------------------------------------------------------------
# MicrobenchmarkInfo dataclass
# ---------------------------------------------------------------------------


def test_microbenchmark_info_inherits_benchmark_info() -> None:
    assert isinstance(_make_info(), BenchmarkInfo)


def test_microbenchmark_info_replace_returns_new_instance() -> None:
    original = _make_info()
    updated = original.replace(name="renamed")
    assert updated is not original
    assert updated.name == "renamed"
    assert original.name == "bench"


def test_microbenchmark_info_replace_changes_microbenchmark_field() -> None:
    original = _make_info(skip=False)
    updated = original.replace(skip=True)
    assert updated.skip is True
    assert original.skip is False


# ---------------------------------------------------------------------------
# complete_args_from_size
# ---------------------------------------------------------------------------


def test_complete_args_no_sentinels_returns_original() -> None:
    info = _make_info()
    args = {"a": 1, "b": "hello"}
    # No sentinel values present so the original dict is returned as-is.
    assert info.complete_args_from_size(100, args) is args


def test_complete_args_size_sentinel_filled() -> None:
    info = _make_info()
    result = info.complete_args_from_size(42, {"n": SIZE, "other": "x"})
    assert result == {"n": 42, "other": "x"}


def test_complete_args_only_size_sentinels_skips_size_to_args() -> None:
    # When only SIZE sentinels are present (no MISSING), size_to_args is
    # not required even if it is None.
    info = _make_info()
    result = info.complete_args_from_size(7, {"n": SIZE})
    assert result == {"n": 7}


def test_complete_args_missing_without_size_to_args_raises() -> None:
    info = _make_info()
    with pytest.raises(RuntimeError, match=r"size_to_args.*was not passed"):
        info.complete_args_from_size(10, {"x": MISSING})


def test_complete_args_missing_with_size_to_args() -> None:
    def size_to_args(size: int) -> dict[str, Any]:
        return {"x": size * 2}

    info = _make_info(size_to_args=size_to_args)
    assert info.complete_args_from_size(5, {"x": MISSING}) == {"x": 10}


def test_complete_args_size_to_args_uses_other_args() -> None:
    def size_to_args(size: int, multiplier: int) -> dict[str, Any]:
        return {"x": size * multiplier}

    info = _make_info(size_to_args=size_to_args)
    result = info.complete_args_from_size(5, {"x": MISSING, "multiplier": 3})
    assert result == {"x": 15, "multiplier": 3}


def test_complete_args_size_to_args_unknown_param_raises() -> None:
    def size_to_args(size: int, unknown: int) -> dict[str, Any]:
        return {"x": 0}

    info = _make_info(size_to_args=size_to_args)
    with pytest.raises(
        RuntimeError,
        match=r"size_to_args.*'unknown'.*does not have the\s+same name",
    ):
        info.complete_args_from_size(5, {"x": MISSING})


def test_complete_args_size_to_args_missing_other_param_raises() -> None:
    def size_to_args(size: int, other: int) -> dict[str, Any]:
        return {"x": 0}

    info = _make_info(size_to_args=size_to_args)
    with pytest.raises(
        RuntimeError, match="cannot have 'other' as an argument"
    ):
        info.complete_args_from_size(5, {"x": MISSING, "other": MISSING})


def test_complete_args_size_to_args_missing_output_value_raises() -> None:
    def size_to_args(size: int) -> dict[str, Any]:
        return {}

    info = _make_info(size_to_args=size_to_args)
    with pytest.raises(RuntimeError, match="did not computed a value for 'x'"):
        info.complete_args_from_size(5, {"x": MISSING})


def test_complete_args_size_and_missing_together() -> None:
    def size_to_args(size: int) -> dict[str, Any]:
        return {"x": size + 1}

    info = _make_info(size_to_args=size_to_args)
    result = info.complete_args_from_size(
        10, {"n": SIZE, "x": MISSING, "other": "z"}
    )
    assert result == {"n": 10, "x": 11, "other": "z"}


def test_complete_args_size_to_args_keyword_only_param() -> None:
    def size_to_args(*, size: int) -> dict[str, Any]:
        return {"x": size * 3}

    info = _make_info(size_to_args=size_to_args)
    assert info.complete_args_from_size(4, {"x": MISSING}) == {"x": 12}


# ---------------------------------------------------------------------------
# get_bytes
# ---------------------------------------------------------------------------


def test_get_bytes_uses_args_to_bytes() -> None:
    def args_to_bytes(n: int) -> int:
        return n * 16

    info = _make_info(args_to_bytes=args_to_bytes)
    assert info.get_bytes(0, {"n": 8}) == 128


def test_get_bytes_with_kwargs() -> None:
    def args_to_bytes(*, n: int) -> int:
        return n

    info = _make_info(args_to_bytes=args_to_bytes)
    assert info.get_bytes(0, {"n": 8}) == 8


def test_get_bytes_uses_args_to_arrays_when_args_to_bytes_none() -> None:
    def args_to_arrays(n: int) -> list[Any]:
        return [("a", n), ("b", (n, n), "int32")]

    info = _make_info(args_to_arrays=args_to_arrays)
    # a: 4 * 8 = 32 bytes; b: 4 * 4 * 4 = 64 bytes
    assert info.get_bytes(0, {"n": 4}) == 32 + 64


def test_get_bytes_resolves_size_sentinel_before_callback() -> None:
    def args_to_bytes(n: int) -> int:
        return n

    info = _make_info(args_to_bytes=args_to_bytes)
    assert info.get_bytes(99, {"n": SIZE}) == 99


def test_get_bytes_neither_callback_set_raises() -> None:
    info = _make_info()
    with pytest.raises(
        RuntimeError, match="Neither `args_to_bytes` nor `args_to_arrays`"
    ):
        info.get_bytes(10, {})


def test_get_bytes_prefers_args_to_bytes_over_args_to_arrays() -> None:
    def args_to_bytes(n: int) -> int:
        return 1

    def args_to_arrays(n: int) -> list[Any]:
        raise AssertionError("args_to_arrays should not be called")

    info = _make_info(
        args_to_bytes=args_to_bytes, args_to_arrays=args_to_arrays
    )
    assert info.get_bytes(0, {"n": 4}) == 1


# ---------------------------------------------------------------------------
# get_work
# ---------------------------------------------------------------------------


def test_get_work_uses_args_to_work() -> None:
    def args_to_work(n: int) -> float:
        return float(n * n)

    info = _make_info(args_to_work=args_to_work)
    assert info.get_work(0, {"n": 4}) == 16.0


def test_get_work_falls_back_to_bytes_when_no_args_to_work() -> None:
    def args_to_bytes(n: int) -> int:
        return n * 100

    info = _make_info(args_to_bytes=args_to_bytes)
    assert info.get_work(0, {"n": 3}) == 300.0


def test_get_work_casts_to_float() -> None:
    def args_to_work(n: int) -> int:
        return n

    info = _make_info(args_to_work=args_to_work)
    result = info.get_work(0, {"n": 5})
    assert isinstance(result, float)
    assert result == 5.0


# ---------------------------------------------------------------------------
# get_plan
# ---------------------------------------------------------------------------


def test_get_plan_none_returns_args_list() -> None:
    info = _make_info()
    args = {"a": 1}
    assert info.get_plan(MagicMock(), args) == [args]


def test_get_plan_dict_returns_single_list() -> None:
    plan = {"x": 1}
    info = _make_info(plan=plan)
    assert info.get_plan(MagicMock(), {}) == [plan]


def test_get_plan_callable_invoked_with_suite() -> None:
    suite = MagicMock()

    def planner(s: Any) -> list[dict[str, Any]]:
        assert s is suite
        return [{"a": 1}, {"a": 2}]

    info = _make_info(plan=planner)
    assert info.get_plan(suite, {}) == [{"a": 1}, {"a": 2}]


def test_get_plan_iterable_returned_as_is() -> None:
    plan = [{"a": 1}, {"a": 2}]
    info = _make_info(plan=plan)
    assert info.get_plan(MagicMock(), {}) is plan


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


def test_should_skip_bool_true() -> None:
    info = _make_info(skip=True)
    assert info.should_skip(MagicMock(), {}) is True


def test_should_skip_bool_false() -> None:
    info = _make_info(skip=False)
    assert info.should_skip(MagicMock(), {}) is False


def test_should_skip_callable_uses_skip_signature() -> None:
    # `should_skip` inspects the callable `skip`'s own signature to gather
    # forwarded arguments. `args_to_bytes` is irrelevant here.
    def skip(suite: Any, n: int) -> bool:
        return n > 10

    info = _make_info(skip=skip)
    suite = MagicMock()
    assert info.should_skip(suite, {"n": 5}) is False
    assert info.should_skip(suite, {"n": 50}) is True


def test_should_skip_callable_receives_suite_argument() -> None:
    received: list[Any] = []

    def skip(suite: Any) -> bool:
        received.append(suite)
        return False

    info = _make_info(skip=skip)
    suite = MagicMock()
    assert info.should_skip(suite, {}) is False
    assert received == [suite]


def test_should_skip_callable_unknown_param_raises() -> None:
    def skip(suite: Any, missing_arg: int) -> bool:
        return False

    info = _make_info(skip=skip)
    with pytest.raises(
        RuntimeError, match=r"skip.*'missing_arg'.*does not have"
    ):
        info.should_skip(MagicMock(), {"n": 1})


# ---------------------------------------------------------------------------
# get_explain_bytes
# ---------------------------------------------------------------------------


def test_get_explain_bytes_uses_explain_bytes_callback() -> None:
    def explain_bytes(n: int) -> list[str]:
        return [f"n*4={n * 4}"]

    info = _make_info(explain_bytes=explain_bytes)
    assert info.get_explain_bytes(0, {"n": 5}) == ["n*4=20"]


def test_get_explain_bytes_falls_back_to_args_to_arrays() -> None:
    def args_to_arrays(n: int) -> list[Any]:
        return [("a", n)]

    info = _make_info(args_to_arrays=args_to_arrays)
    assert info.get_explain_bytes(0, {"n": 4}) == [
        "a: 4 float64s x 8 bytes / float64 = 32 bytes"
    ]


def test_get_explain_bytes_no_callbacks_returns_empty() -> None:
    info = _make_info()
    assert info.get_explain_bytes(0, {}) == []


# ---------------------------------------------------------------------------
# get_explain_work
# ---------------------------------------------------------------------------


def test_get_explain_work_none_returns_empty() -> None:
    info = _make_info()
    assert info.get_explain_work({}) == []


def test_get_explain_work_invokes_callback() -> None:
    def explain_work(n: int) -> list[str]:
        return [f"work=n^2={n * n}"]

    info = _make_info(explain_work=explain_work)
    assert info.get_explain_work({"n": 3}) == ["work=n^2=9"]


# ---------------------------------------------------------------------------
# format_search_string
# ---------------------------------------------------------------------------


def test_format_search_string_uses_positional_args() -> None:
    info = _make_info(name="myb")

    def f(a: int, b: int) -> None:
        pass

    assert info.format_search_string(f, (1, 2), {}) == "myb(a=1,b=2)"


def test_format_search_string_falls_back_to_default_value() -> None:
    info = _make_info(name="myb")

    def f(a: int, b: int = 7) -> None:
        pass

    assert info.format_search_string(f, (1,), {}) == "myb(a=1,b=7)"


def test_format_search_string_kwargs_override_positional() -> None:
    info = _make_info(name="myb")

    def f(a: int, b: int) -> None:
        pass

    assert info.format_search_string(f, (1,), {"b": 99}) == "myb(a=1,b=99)"


def test_format_search_string_marks_unspecified_arg_as_missing() -> None:
    info = _make_info(name="myb")

    def f(a: int) -> None:
        pass

    assert info.format_search_string(f, (), {}) == "myb(a=MISSING)"


def test_format_search_string_applies_format_under_input_name() -> None:
    info = _make_info(
        name="myb",
        input_names={"a": "Aval"},
        formats={"Aval": lambda v: f"<{v}>"},
    )

    def f(a: int) -> None:
        pass

    assert info.format_search_string(f, (5,), {}) == "myb(a=<5>)"


# ---------------------------------------------------------------------------
# pretty_args
# ---------------------------------------------------------------------------


def test_pretty_args_passes_missing_through() -> None:
    info = _make_info()
    assert info.pretty_args({"x": MISSING}) == {"x": MISSING}


def test_pretty_args_passes_size_through() -> None:
    info = _make_info()
    assert info.pretty_args({"n": SIZE}) == {"n": SIZE}


def test_pretty_args_applies_format_under_input_name() -> None:
    info = _make_info(
        input_names={"a": "Aval"}, formats={"Aval": lambda v: f"~{v}~"}
    )
    assert info.pretty_args({"a": 5}) == {"a": "~5~"}


def test_pretty_args_no_formatting_passes_value_through() -> None:
    info = _make_info()
    assert info.pretty_args({"x": 42}) == {"x": 42}


# ---------------------------------------------------------------------------
# microbenchmark decorator
# ---------------------------------------------------------------------------


def test_microbenchmark_decorator_returns_staticmethod() -> None:
    @microbenchmark()
    def f() -> float:
        return 0.0

    assert isinstance(f, staticmethod)


def test_microbenchmark_decorator_attaches_info_attribute() -> None:
    @microbenchmark()
    def f() -> float:
        return 0.0

    assert hasattr(f, INFO)
    info = getattr(f, INFO)
    assert isinstance(info, MicrobenchmarkInfo)
    assert info.name == "f"


def test_microbenchmark_decorator_default_output_names_uses_ms_label() -> None:
    @microbenchmark()
    def f() -> float:
        return 0.0

    info = getattr(f, INFO)
    assert info.output_names == "time per run (ms)"
    assert info.returns_time == 0


def test_microbenchmark_decorator_returns_time_false() -> None:
    @microbenchmark(returns_time=False, output_names="out")
    def f() -> float:
        return 0.0

    info = getattr(f, INFO)
    assert info.returns_time == -1
    assert info.output_names == "out"


def test_microbenchmark_decorator_returns_time_integer_replaces_label() -> (
    None
):
    @microbenchmark(returns_time=1, output_names=("a", "x", "b"))
    def f() -> tuple[int, float, int]:
        return 0, 0.0, 0

    info = getattr(f, INFO)
    assert info.output_names == ("a", "time per run (ms)", "b")
    assert info.returns_time == 1


def test_microbenchmark_decorator_propagates_plan_and_skip() -> None:
    plan = [{"n": 1}, {"n": 2}]

    @microbenchmark(plan=plan, skip=True)
    def f(n: int) -> float:
        return 0.0

    info = getattr(f, INFO)
    assert info.plan == plan
    assert info.skip is True


def test_microbenchmark_decorator_propagates_size_callbacks() -> None:
    def size_to_args(size: int) -> dict[str, Any]:
        return {"n": size}

    def args_to_bytes(n: int) -> int:
        return n

    @microbenchmark(size_to_args=size_to_args, args_to_bytes=args_to_bytes)
    def f(n: int) -> float:
        return 0.0

    info = getattr(f, INFO)
    assert info.size_to_args is size_to_args
    assert info.args_to_bytes is args_to_bytes


def test_microbenchmark_decorator_explicit_name_overrides() -> None:
    @microbenchmark(name="custom")
    def f() -> float:
        return 0.0

    assert getattr(f, INFO).name == "custom"


# ---------------------------------------------------------------------------
# get_microbenchmark_info
# ---------------------------------------------------------------------------


def test_get_microbenchmark_info_returns_attached_info() -> None:
    @microbenchmark(name="explicit")
    def f() -> float:
        return 0.0

    info = get_microbenchmark_info(f)
    assert isinstance(info, MicrobenchmarkInfo)
    assert info.name == "explicit"


def test_get_microbenchmark_info_generates_default_when_missing() -> None:
    def f(n: int) -> float:
        return 0.0

    info = get_microbenchmark_info(f)
    assert isinstance(info, MicrobenchmarkInfo)
    assert info.name == "f"
    # _create_microbenchmark_info applies the "ms" default label.
    assert info.output_names == "time per run (ms)"


def test_get_microbenchmark_info_default_does_not_attach_attribute() -> None:
    def f() -> float:
        return 0.0

    get_microbenchmark_info(f)
    assert not hasattr(f, INFO)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
