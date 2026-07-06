from __future__ import annotations

import argparse
import math
import re

from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass, field

DEFAULT_PROBLEM_SIZE = 10_000_000

_MEMORY_SIZE_PATTERN = re.compile(
    r"^(?P<value>[0-9]+)(?P<unit>B|KiB|MiB|GiB|TiB)$"
)
_MEMORY_SIZE_UNITS = {
    "B": 1,
    "KiB": 1 << 10,
    "MiB": 1 << 20,
    "GiB": 1 << 30,
    "TiB": 1 << 40,
}


def parse_memory_size(value: str) -> int:
    match = _MEMORY_SIZE_PATTERN.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "memory size must use <integer><unit> with "
            "B, KiB, MiB, GiB, or TiB"
        )

    amount = int(match.group("value"))
    if amount <= 0:
        raise argparse.ArgumentTypeError("memory size must be positive")

    unit = match.group("unit")
    return amount * _MEMORY_SIZE_UNITS[unit]


def parse_work_scale(value: str) -> float:
    try:
        amount = float(value)
    except ValueError as ex:
        raise argparse.ArgumentTypeError(
            "work scale must be a positive finite number"
        ) from ex

    if not math.isfinite(amount) or amount <= 0.0:
        raise argparse.ArgumentTypeError(
            "work scale must be a positive finite number"
        )
    return amount


def add_size_request_parser_group(parser: ArgumentParser) -> None:
    group = parser.add_argument_group()
    group.add_argument(
        "--rescale-by-work",
        dest="rescale_by_work",
        metavar="WORK",
        type=parse_work_scale,
        nargs="+",
        default=argparse.SUPPRESS,
        help="Relative work factors to apply to each base size",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--size",
        type=int,
        default=argparse.SUPPRESS,
        nargs="+",
        help=(
            "Exact benchmark size with suite-specific semantics "
            f"(default: {DEFAULT_PROBLEM_SIZE:,} when neither sizing flag "
            "is provided)"
        ),
    )
    group.add_argument(
        "--memory-size",
        dest="memory_size",
        metavar="SIZE",
        type=parse_memory_size,
        default=argparse.SUPPRESS,
        nargs="+",
        help=(
            "Approximate benchmark working-set target using binary units "
            "(B, KiB, MiB, GiB, TiB)"
        ),
    )


@dataclass(frozen=True)
class SizeRequest:
    exact_size: list[int] | None = None
    memory_target_bytes: list[int] | None = None
    rescale_by_work: list[float] = field(default_factory=lambda: [1.0])

    def __post_init__(self) -> None:
        if not self.rescale_by_work:
            raise RuntimeError(
                "--rescale-by-work must specify at least one value"
            )
        if any(
            (not math.isfinite(work_scale)) or work_scale <= 0.0
            for work_scale in self.rescale_by_work
        ):
            raise RuntimeError("--rescale-by-work values must be positive")

    @classmethod
    def from_namespace(
        cls,
        args: Namespace,
        *,
        default_exact_size: int | None = DEFAULT_PROBLEM_SIZE,
    ) -> SizeRequest:
        exact_size = getattr(args, "size", None)
        memory_target_bytes = getattr(args, "memory_size", None)
        rescale_by_work = getattr(args, "rescale_by_work", [1.0])

        if exact_size is None and memory_target_bytes is None:
            if default_exact_size is None:
                raise RuntimeError("size request must specify a sizing mode")
            exact_size = [default_exact_size]

        if exact_size is not None and memory_target_bytes is not None:
            raise RuntimeError(
                "--size and --memory-size must be mutually exclusive"
            )
        return cls(
            exact_size=exact_size,
            memory_target_bytes=memory_target_bytes,
            rescale_by_work=rescale_by_work,
        )

    @property
    def uses_work_rescale(self) -> bool:
        return len(self.rescale_by_work) != 1 or self.rescale_by_work[0] != 1.0

    def config_lines(self) -> list[str]:
        if self.exact_size is not None:
            sizes = ", ".join([f"{e:,}" for e in self.exact_size])
            lines = ["Sizing: exact (--size)", f"Suite-defined size: {sizes}"]
            if self.uses_work_rescale:
                scales = ", ".join(str(e) for e in self.rescale_by_work)
                lines.append(f"Work rescale factors: {scales}")
            return lines

        assert self.memory_target_bytes is not None
        sizes = ", ".join([f"{e:,}" for e in self.memory_target_bytes])
        lines = [
            "Sizing: working-set target (--memory-size)",
            (f"Approximate working-set target (bytes): {sizes}"),
        ]
        if self.uses_work_rescale:
            scales = ", ".join(str(e) for e in self.rescale_by_work)
            lines.append(f"Work rescale factors: {scales}")
        return lines


def resolve_size_by_monotonic_search(
    target: int | float,
    *,
    estimate_value: Callable[[int], int | float],
    initial_guess: int,
) -> int:
    """
    Find the largest size whose monotonic estimate is less than or equal to
    a target. If even the minimum size exceeds the target, return the minimum
    size and let the caller decide whether to warn or fail.
    """
    if target < 0:
        raise ValueError("target must be non-negative")

    low = 1
    low_value = estimate_value(low)
    if low_value > target:
        return low

    high = max(2, initial_guess)
    high_value = estimate_value(high)
    if high_value < low_value:
        raise RuntimeError("Binary search function violates monotonicity")

    for _ in range(64):
        if high_value > target:
            break
        low = high
        low_value = high_value
        high *= 2
        new_high_value = estimate_value(high)
        if new_high_value < high_value:
            raise RuntimeError("Binary search function violates monotonicity")
        high_value = new_high_value
    else:
        raise RuntimeError("Unable to find upper bound for binary search")

    while low + 1 < high:
        mid = low + (high - low) // 2
        mid_value = estimate_value(mid)
        if mid_value < low_value or mid_value > high_value:
            raise RuntimeError("Binary search function violates monotonicity")
        if mid_value <= target:
            low = mid
            low_value = mid_value
        else:
            high = mid
            high_value = mid_value
    return low


def nthroot(x: int, p: int, lower_bound: int = 1) -> int:
    """Return integer approximation of `p`th root of `x`.

    Has optional `lower_bound`.
    """
    float_root = x ** (1 / p)
    root_round = round(float_root)
    root_approx: int
    if root_round**p == x:
        root_approx = root_round
    else:
        root_approx = math.floor(float_root)
    return max(lower_bound, root_approx)
