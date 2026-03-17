from __future__ import annotations

import argparse
import re
import warnings

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable
from dataclasses import dataclass

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


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def add_size_request_parser_group(parser: ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--size",
        type=int,
        default=argparse.SUPPRESS,
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
        help=(
            "Approximate benchmark working-set target using binary units "
            "(B, KiB, MiB, GiB, TiB)"
        ),
    )


@dataclass(frozen=True)
class SizeResolution:
    resolved_size: int
    requested_memory_target_bytes: int
    estimated_working_set_bytes: int
    detail_lines: tuple[str, ...] = ()

    def panel_lines(self) -> list[str]:
        lines = [
            f"resolved_size: {self.resolved_size:,}",
            (
                "requested_memory_target: "
                f"{self.requested_memory_target_bytes:,} bytes"
            ),
            (
                "estimated_working_set: "
                f"{self.estimated_working_set_bytes:,}"
                " bytes"
            ),
        ]
        lines.extend(self.detail_lines)
        return lines


@dataclass(frozen=True)
class SizeRequest:
    exact_size: int | None = None
    memory_target_bytes: int | None = None

    @classmethod
    def from_namespace(
        cls,
        args: Namespace,
        *,
        default_exact_size: int | None = DEFAULT_PROBLEM_SIZE,
    ) -> SizeRequest:
        exact_size = getattr(args, "size", None)
        memory_target_bytes = getattr(args, "memory_size", None)

        if exact_size is None and memory_target_bytes is None:
            if default_exact_size is None:
                raise RuntimeError("size request must specify a sizing mode")
            exact_size = default_exact_size

        if exact_size is not None and memory_target_bytes is not None:
            raise RuntimeError(
                "--size and --memory-size must be mutually exclusive"
            )
        return cls(
            exact_size=exact_size, memory_target_bytes=memory_target_bytes
        )

    def config_lines(self) -> list[str]:
        if self.exact_size is not None:
            return [
                "Sizing: exact (--size)",
                f"Suite-defined size: {self.exact_size:,}",
            ]

        assert self.memory_target_bytes is not None
        return [
            "Sizing: heuristic target (--memory-size)",
            (
                "Approximate working-set target: "
                f"{self.memory_target_bytes:,} bytes"
            ),
        ]


def resolve_suite_size(
    size_request: SizeRequest,
    *,
    resolve_from_target: Callable[[int], int],
    estimate_working_set_bytes: Callable[[int], int],
    describe_size: Callable[[int], Iterable[str]] | None = None,
) -> tuple[int, SizeResolution | None]:
    """
    Resolve an exact suite size from an explicit size or memory target.

    Parameters
    ----------
    size_request : SizeRequest
        User-provided sizing mode and value.
    resolve_from_target : Callable[[int], int]
        Maps a target working-set size in bytes to a suite-specific size.
    estimate_working_set_bytes : Callable[[int], int]
        Estimates the suite working set for a resolved size.
    describe_size : Callable[[int], Iterable[str]] | None, optional
        Produces additional human-readable sizing details.
    """
    if size_request.exact_size is not None:
        return size_request.exact_size, None

    target_bytes = size_request.memory_target_bytes
    assert target_bytes is not None
    resolved_size = resolve_from_target(target_bytes)
    detail_lines: Iterable[str] = ()
    if describe_size is not None:
        detail_lines = describe_size(resolved_size)
    estimated_working_set_bytes = estimate_working_set_bytes(resolved_size)
    resolution = SizeResolution(
        resolved_size=resolved_size,
        requested_memory_target_bytes=target_bytes,
        estimated_working_set_bytes=estimated_working_set_bytes,
        detail_lines=tuple(detail_lines),
    )
    if estimated_working_set_bytes > target_bytes:
        warnings.warn(
            "memory target is smaller than estimated working set: "
            f"estimated={estimated_working_set_bytes:,} bytes, "
            f"target={target_bytes:,} bytes",
            RuntimeWarning,
            stacklevel=2,
        )
    return resolved_size, resolution


def resolve_size_by_binary_search(
    target_bytes: int,
    *,
    estimate_working_set_bytes: Callable[[int], int],
    initial_guess: int,
) -> int:
    low = 1
    high = max(1, initial_guess)
    while estimate_working_set_bytes(high) <= target_bytes:
        low = high
        high *= 2

    while low < high:
        mid = (low + high + 1) // 2
        if estimate_working_set_bytes(mid) <= target_bytes:
            low = mid
        else:
            high = mid - 1

    return low


def resolve_linear_suite_size(
    size_request: SizeRequest,
    *,
    bytes_per_element: int,
    describe_size: Callable[[int], Iterable[str]] | None = None,
) -> tuple[int, SizeResolution | None]:
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")

    return resolve_suite_size(
        size_request,
        resolve_from_target=lambda target_bytes: max(
            1, target_bytes // bytes_per_element
        ),
        estimate_working_set_bytes=lambda resolved_size: (
            bytes_per_element * resolved_size
        ),
        describe_size=describe_size,
    )
