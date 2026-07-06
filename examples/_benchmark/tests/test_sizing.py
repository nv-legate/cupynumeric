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

import argparse
import sys
from argparse import ArgumentParser, Namespace
from typing import Any

import pytest

from _benchmark.sizing import (
    DEFAULT_PROBLEM_SIZE,
    SizeRequest,
    add_size_request_parser_group,
    nthroot,
    parse_memory_size,
    parse_work_scale,
    resolve_size_by_monotonic_search,
)


# ---------------------------------------------------------------------------
# parse_memory_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1B", 1),
        ("512B", 512),
        ("1KiB", 1024),
        ("2KiB", 2 * 1024),
        ("1MiB", 1 << 20),
        ("3MiB", 3 * (1 << 20)),
        ("1GiB", 1 << 30),
        ("1TiB", 1 << 40),
    ],
)
def test_parse_memory_size_valid_units(value: str, expected: int) -> None:
    assert parse_memory_size(value) == expected


def test_parse_memory_size_strips_whitespace() -> None:
    assert parse_memory_size("  4KiB  ") == 4 * 1024


@pytest.mark.parametrize(
    "value",
    ["", "foo", "1mb", "1.5MiB", "1 KiB", "KiB", "1KB", "-1MiB", "+1MiB"],
)
def test_parse_memory_size_invalid_format_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="memory size must"):
        parse_memory_size(value)


def test_parse_memory_size_zero_rejected() -> None:
    with pytest.raises(
        argparse.ArgumentTypeError, match="memory size must be positive"
    ):
        parse_memory_size("0MiB")


# ---------------------------------------------------------------------------
# parse_work_scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", 1.0), ("1.0", 1.0), ("2.5", 2.5), ("0.5", 0.5), ("1e2", 100.0)],
)
def test_parse_work_scale_valid(value: str, expected: float) -> None:
    assert parse_work_scale(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["abc", "", "1.0x", "--", "nope"])
def test_parse_work_scale_non_numeric_rejected(value: str) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError, match="work scale must be a positive"
    ):
        parse_work_scale(value)


@pytest.mark.parametrize("value", ["0", "0.0", "-1", "-2.5"])
def test_parse_work_scale_non_positive_rejected(value: str) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError, match="work scale must be a positive"
    ):
        parse_work_scale(value)


@pytest.mark.parametrize("value", ["inf", "nan", "-inf"])
def test_parse_work_scale_non_finite_rejected(value: str) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError, match="work scale must be a positive"
    ):
        parse_work_scale(value)


# ---------------------------------------------------------------------------
# add_size_request_parser_group
# ---------------------------------------------------------------------------


def _parse(*argv: str) -> Namespace:
    parser = ArgumentParser()
    add_size_request_parser_group(parser)
    return parser.parse_args(list(argv))


def test_add_size_request_parser_group_no_flags_yields_empty_namespace() -> (
    None
):
    ns = _parse()
    assert not hasattr(ns, "size")
    assert not hasattr(ns, "memory_size")
    assert not hasattr(ns, "rescale_by_work")


def test_add_size_request_parser_group_size_flag() -> None:
    ns = _parse("--size", "100", "200")
    assert ns.size == [100, 200]


def test_add_size_request_parser_group_memory_size_flag() -> None:
    ns = _parse("--memory-size", "1KiB", "2MiB")
    assert ns.memory_size == [1024, 2 * (1 << 20)]


def test_add_size_request_parser_group_rescale_by_work_flag() -> None:
    ns = _parse("--rescale-by-work", "1.5", "2.0")
    assert ns.rescale_by_work == pytest.approx([1.5, 2.0])


def test_add_size_request_parser_group_size_and_memory_size_mutex() -> None:
    with pytest.raises(SystemExit):
        _parse("--size", "100", "--memory-size", "1MiB")


def test_add_size_request_parser_group_rejects_invalid_memory_size() -> None:
    with pytest.raises(SystemExit):
        _parse("--memory-size", "notasize")


def test_add_size_request_parser_group_rejects_invalid_work_scale() -> None:
    with pytest.raises(SystemExit):
        _parse("--rescale-by-work", "0")


def test_add_size_request_parser_group_rescale_with_size() -> None:
    ns = _parse("--size", "10", "--rescale-by-work", "2.0")
    assert ns.size == [10]
    assert ns.rescale_by_work == pytest.approx([2.0])


# ---------------------------------------------------------------------------
# SizeRequest.__post_init__
# ---------------------------------------------------------------------------


def test_size_request_default_rescale_is_one() -> None:
    sr = SizeRequest(exact_size=[10])
    assert sr.rescale_by_work == [1.0]


def test_size_request_empty_rescale_raises() -> None:
    with pytest.raises(
        RuntimeError, match="--rescale-by-work must specify at least one value"
    ):
        SizeRequest(exact_size=[10], rescale_by_work=[])


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_size_request_invalid_rescale_value_raises(bad: float) -> None:
    with pytest.raises(
        RuntimeError, match="--rescale-by-work values must be positive"
    ):
        SizeRequest(exact_size=[10], rescale_by_work=[1.0, bad])


def test_size_request_is_frozen() -> None:
    sr = SizeRequest(exact_size=[10])
    with pytest.raises(AttributeError):
        sr.exact_size = [20]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SizeRequest.from_namespace
# ---------------------------------------------------------------------------


def _ns(**attrs: Any) -> Namespace:
    ns = Namespace()
    for key, value in attrs.items():
        setattr(ns, key, value)
    return ns


def test_from_namespace_uses_default_when_no_flags() -> None:
    sr = SizeRequest.from_namespace(_ns())
    assert sr.exact_size == [DEFAULT_PROBLEM_SIZE]
    assert sr.memory_target_bytes is None
    assert sr.rescale_by_work == [1.0]


def test_from_namespace_uses_explicit_default() -> None:
    sr = SizeRequest.from_namespace(_ns(), default_exact_size=42)
    assert sr.exact_size == [42]


def test_from_namespace_no_default_raises_when_no_flags() -> None:
    with pytest.raises(
        RuntimeError, match="size request must specify a sizing mode"
    ):
        SizeRequest.from_namespace(_ns(), default_exact_size=None)


def test_from_namespace_size_only() -> None:
    sr = SizeRequest.from_namespace(_ns(size=[5, 10]))
    assert sr.exact_size == [5, 10]
    assert sr.memory_target_bytes is None


def test_from_namespace_memory_size_only() -> None:
    sr = SizeRequest.from_namespace(_ns(memory_size=[1024]))
    assert sr.exact_size is None
    assert sr.memory_target_bytes == [1024]


def test_from_namespace_both_flags_raises() -> None:
    with pytest.raises(
        RuntimeError, match="--size and --memory-size must be mutually"
    ):
        SizeRequest.from_namespace(_ns(size=[5], memory_size=[1024]))


def test_from_namespace_propagates_rescale_by_work() -> None:
    sr = SizeRequest.from_namespace(
        _ns(size=[5], rescale_by_work=[1.0, 2.0, 4.0])
    )
    assert sr.rescale_by_work == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# SizeRequest.uses_work_rescale
# ---------------------------------------------------------------------------


def test_uses_work_rescale_false_for_default() -> None:
    sr = SizeRequest(exact_size=[10])
    assert sr.uses_work_rescale is False


def test_uses_work_rescale_false_for_explicit_single_one() -> None:
    sr = SizeRequest(exact_size=[10], rescale_by_work=[1.0])
    assert sr.uses_work_rescale is False


def test_uses_work_rescale_true_for_non_one_single() -> None:
    sr = SizeRequest(exact_size=[10], rescale_by_work=[2.0])
    assert sr.uses_work_rescale is True


def test_uses_work_rescale_true_for_multiple_values() -> None:
    sr = SizeRequest(exact_size=[10], rescale_by_work=[1.0, 2.0])
    assert sr.uses_work_rescale is True


# ---------------------------------------------------------------------------
# SizeRequest.config_lines
# ---------------------------------------------------------------------------


def test_config_lines_exact_size_no_rescale() -> None:
    sr = SizeRequest(exact_size=[1_000_000])
    lines = sr.config_lines()
    assert lines == ["Sizing: exact (--size)", "Suite-defined size: 1,000,000"]


def test_config_lines_exact_size_multiple_values() -> None:
    sr = SizeRequest(exact_size=[100, 2_000])
    lines = sr.config_lines()
    assert lines == [
        "Sizing: exact (--size)",
        "Suite-defined size: 100, 2,000",
    ]


def test_config_lines_exact_size_with_rescale() -> None:
    sr = SizeRequest(exact_size=[100], rescale_by_work=[1.0, 2.5])
    lines = sr.config_lines()
    assert lines == [
        "Sizing: exact (--size)",
        "Suite-defined size: 100",
        "Work rescale factors: 1.0, 2.5",
    ]


def test_config_lines_memory_target() -> None:
    sr = SizeRequest(memory_target_bytes=[1024, 2048])
    lines = sr.config_lines()
    assert lines == [
        "Sizing: working-set target (--memory-size)",
        "Approximate working-set target (bytes): 1,024, 2,048",
    ]


def test_config_lines_memory_target_with_rescale() -> None:
    sr = SizeRequest(memory_target_bytes=[1024], rescale_by_work=[2.0])
    lines = sr.config_lines()
    assert lines == [
        "Sizing: working-set target (--memory-size)",
        "Approximate working-set target (bytes): 1,024",
        "Work rescale factors: 2.0",
    ]


# ---------------------------------------------------------------------------
# resolve_size_by_monotonic_search
# ---------------------------------------------------------------------------


def test_resolve_size_negative_target_raises() -> None:
    with pytest.raises(ValueError, match="target must be non-negative"):
        resolve_size_by_monotonic_search(
            -1, estimate_value=lambda n: n, initial_guess=10
        )


def test_resolve_size_returns_minimum_when_min_estimate_exceeds_target() -> (
    None
):
    # estimate(1) = 100 > target = 50, should return 1
    result = resolve_size_by_monotonic_search(
        50, estimate_value=lambda n: n * 100, initial_guess=10
    )
    assert result == 1


def test_resolve_size_linear_function() -> None:
    # estimate(n) = n, find largest n with n <= target
    result = resolve_size_by_monotonic_search(
        100, estimate_value=lambda n: n, initial_guess=1
    )
    assert result == 100


def test_resolve_size_linear_function_with_large_initial_guess() -> None:
    result = resolve_size_by_monotonic_search(
        100, estimate_value=lambda n: n, initial_guess=10_000
    )
    assert result == 100


def test_resolve_size_quadratic_function() -> None:
    # estimate(n) = n*n; find largest n with n*n <= 10000 -> n == 100
    result = resolve_size_by_monotonic_search(
        10_000, estimate_value=lambda n: n * n, initial_guess=2
    )
    assert result == 100


def test_resolve_size_float_target_and_estimate() -> None:
    result = resolve_size_by_monotonic_search(
        10.5, estimate_value=lambda n: float(n), initial_guess=2
    )
    assert result == 10


def test_resolve_size_constant_estimate_returns_high_bound() -> None:
    # estimate is constant 0, so any size satisfies target=10.
    # The loop multiplies high by 2 for 64 iterations until it gives up,
    # because high_value never exceeds target. So this should raise.
    with pytest.raises(
        RuntimeError, match="Unable to find upper bound for binary search"
    ):
        resolve_size_by_monotonic_search(
            10, estimate_value=lambda n: 0, initial_guess=2
        )


def test_resolve_size_non_monotonic_initial_high_raises() -> None:
    # estimate(1) = 0, estimate(>=2) = -1: high_value < low_value triggers
    # the monotonicity check.
    def estimate(n: int) -> int:
        return 0 if n == 1 else -1

    with pytest.raises(
        RuntimeError, match="Binary search function violates monotonicity"
    ):
        resolve_size_by_monotonic_search(
            10, estimate_value=estimate, initial_guess=2
        )


def test_resolve_size_non_monotonic_during_bisection_raises() -> None:
    # Enter the bisection loop with a valid (low, high) bracket but have
    # the midpoint estimate violate monotonicity. With initial_guess=10,
    # the loop bounds become low=1 (estimate=0) and high=10 (estimate=100,
    # already above target=50), so doubling is skipped and we go directly
    # to bisection where estimate(mid) returns a value below low_value.
    def estimate(n: int) -> int:
        if n == 1:
            return 0
        if n == 10:
            return 100
        return -1

    with pytest.raises(
        RuntimeError, match="Binary search function violates monotonicity"
    ):
        resolve_size_by_monotonic_search(
            50, estimate_value=estimate, initial_guess=10
        )


def test_resolve_size_non_monotonic_during_doubling_raises() -> None:
    # Monotonic for the initial (low=1, high=2) range so we enter the
    # doubling loop, then drop sharply at high=4 to trigger the
    # monotonicity check.
    def estimate(n: int) -> int:
        if n <= 2:
            return n
        return -100

    with pytest.raises(
        RuntimeError, match="Binary search function violates monotonicity"
    ):
        resolve_size_by_monotonic_search(
            1_000_000, estimate_value=estimate, initial_guess=2
        )


def test_resolve_size_target_zero_returns_one_when_min_value_is_zero() -> None:
    # estimate(1) = 0 <= target = 0, then high = 2 with estimate(2) = 0,
    # never exceeds target, so the upper-bound search fails.
    with pytest.raises(
        RuntimeError, match="Unable to find upper bound for binary search"
    ):
        resolve_size_by_monotonic_search(
            0, estimate_value=lambda n: 0, initial_guess=2
        )


def test_resolve_size_initial_guess_one_uses_minimum_high_of_two() -> None:
    # initial_guess=1 should be promoted to 2 internally (max(2, 1)).
    result = resolve_size_by_monotonic_search(
        50, estimate_value=lambda n: n, initial_guess=1
    )
    assert result == 50


# ---------------------------------------------------------------------------
# nthroot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "p", "expected"),
    [
        (8, 3, 2),
        (27, 3, 3),
        (1000, 3, 10),
        (100, 2, 10),
        (10_000, 2, 100),
        (16, 4, 2),
        (1, 5, 1),
    ],
)
def test_nthroot_exact_powers(x: int, p: int, expected: int) -> None:
    assert nthroot(x, p) == expected


def test_nthroot_first_root_is_identity() -> None:
    assert nthroot(123, 1) == 123


@pytest.mark.parametrize(
    ("x", "p", "expected"),
    [
        (10, 2, 3),  # sqrt(10) ~ 3.16 -> 3
        (9999, 2, 99),  # sqrt(9999) ~ 99.99 -> 99
        (26, 3, 2),  # cbrt(26) ~ 2.96 -> 2
        (1023, 10, 1),  # 2**10 == 1024 -> floor below 2
    ],
)
def test_nthroot_floors_non_exact(x: int, p: int, expected: int) -> None:
    assert nthroot(x, p) == expected


def test_nthroot_respects_lower_bound() -> None:
    # floor(8 ** (1/3)) == 2, but the lower bound dominates.
    assert nthroot(8, 3, lower_bound=5) == 5


def test_nthroot_lower_bound_not_applied_when_root_larger() -> None:
    assert nthroot(1000, 3, lower_bound=5) == 10


def test_nthroot_default_lower_bound_one() -> None:
    # floor(0 ** (1/2)) == 0, clamped up to the default lower bound of 1.
    assert nthroot(0, 2) == 1


def test_nthroot_returns_int() -> None:
    result = nthroot(1000, 3)
    assert isinstance(result, int)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
