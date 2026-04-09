from __future__ import annotations

import argparse
import importlib
import math
import sys

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_MICROBENCHMARKS_DIR = _EXAMPLES_DIR / "microbenchmarks"


def _ensure_example_paths() -> None:
    for path in (str(_MICROBENCHMARKS_DIR), str(_EXAMPLES_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)


@lru_cache
def _module(name: str):
    _ensure_example_paths()
    return importlib.import_module(name)


def _sizing():
    return _module("_benchmark.sizing")


def _build_size_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _sizing().add_size_request_parser_group(parser)
    return parser


def _next_size_with_larger_estimate(estimate, size: int) -> int:
    current = estimate(size)
    limit = size + max(1024, 2 * math.isqrt(max(size, 1)) + 16)
    for candidate in range(size + 1, limit + 1):
        if estimate(candidate) > current:
            return candidate
    raise AssertionError("failed to find a larger estimated working set")


def _assert_target_resolution(resolve, estimate, target_bytes: int) -> None:
    size = resolve(target_bytes)
    assert estimate(size) <= target_bytes
    larger_size = _next_size_with_larger_estimate(estimate, size)
    assert estimate(larger_size) > target_bytes


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1B", 1),
        ("2KiB", 2 << 10),
        ("3MiB", 3 << 20),
        ("4GiB", 4 << 30),
        ("5TiB", 5 << 40),
    ],
)
def test_parse_memory_size_valid(value: str, expected: int) -> None:
    assert _sizing().parse_memory_size(value) == expected


@pytest.mark.parametrize("value", ["0B", "1KB", "1Mi", "GiB", "-1MiB"])
def test_parse_memory_size_invalid(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _sizing().parse_memory_size(value)


def test_size_request_defaults_to_legacy_exact_size() -> None:
    args = _build_size_parser().parse_args([])
    benchmark = _sizing()
    request = benchmark.SizeRequest.from_namespace(args)

    assert request.exact_size == [benchmark.DEFAULT_PROBLEM_SIZE]
    assert request.memory_target_bytes is None
    assert request.config_lines() == [
        "Sizing: exact (--size)",
        f"Suite-defined size: {benchmark.DEFAULT_PROBLEM_SIZE:,}",
    ]


def test_size_request_help_mentions_legacy_default() -> None:
    help_text = _build_size_parser().format_help()

    assert "default: 10,000,000" in help_text
    assert "default: None" not in help_text


def test_size_request_parses_memory_target_mode() -> None:
    args = _build_size_parser().parse_args(["--memory-size", "2GiB"])
    request = _sizing().SizeRequest.from_namespace(args)

    assert request.exact_size is None
    assert request.memory_target_bytes == [2 << 30]


def test_size_request_requires_mode_when_default_is_disabled() -> None:
    args = _build_size_parser().parse_args([])
    with pytest.raises(RuntimeError, match="size request must specify"):
        _sizing().SizeRequest.from_namespace(args, default_exact_size=None)


def test_size_request_parser_rejects_conflicting_modes() -> None:
    parser = _build_size_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--size", "32", "--memory-size", "1MiB"])


def _make_recording_suite(
    *, forbid_resolution: bool = False
) -> SimpleNamespace:
    calls: list[tuple[str, tuple[object, ...]]] = []
    resolutions: list[object] = []

    def print_size_resolution(resolution: object) -> None:
        if forbid_resolution:
            raise AssertionError(
                "exact-size path should not print a resolution"
            )
        resolutions.append(resolution)

    def run_timed(func, *args, **kwargs) -> None:
        del kwargs
        calls.append((func.__name__, args))

    def run_timed_with_info(info, func, *args, **kwargs) -> None:
        del info, kwargs
        calls.append((func.__name__, args))

    def run_timed_with_generator(info, func, gen, **kwargs) -> None:
        del info, kwargs
        for args in gen:
            calls.append((func.__name__, args))

    return SimpleNamespace(
        np=np,
        timer=object(),
        runs=1,
        warmup=0,
        calls=calls,
        resolutions=resolutions,
        print_size_resolution=print_size_resolution,
        run_timed=run_timed,
        run_timed_with_info=run_timed_with_info,
        run_timed_with_generator=run_timed_with_generator,
    )


@pytest.mark.parametrize(
    ("module_name", "runner_kwargs", "expected_name", "size_index"),
    [
        ("general_astype_bench", {}, "astype", 2),
        ("general_random_bench", {}, "randint", 1),
        ("general_nanred_bench", {}, "nan_red", 3),
        ("general_scalared_bench", {}, "scalar_red", 3),
        ("ufunc_bench", {"perform_check": False}, "unary_exp", 1),
        ("general_indexing_bench", {}, "boolean_get", 1),
        ("fast_advanced_indexing_bench", {}, "putmask_scalar", 1),
    ],
)
def test_linear_suites_resolve_memory_target_in_run_benchmarks(
    module_name: str,
    runner_kwargs: dict[str, object],
    expected_name: str,
    size_index: int,
) -> None:
    module = _module(module_name)
    sizing = _sizing()
    suite = _make_recording_suite()
    request = sizing.SizeRequest(memory_target_bytes=[100])

    module.run_benchmarks(suite, request, **runner_kwargs)

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    resolution = suite.resolutions[0][0]
    assert resolution.requested_memory_target_bytes == 100
    assert resolution.estimated_working_set_bytes <= 100
    assert suite.calls
    call_name, call_args = suite.calls[0]
    assert call_name == expected_name
    assert call_args[size_index] == [resolution.resolved_size]


@pytest.mark.parametrize(
    ("module_name", "expected_name", "precision"),
    [("sort_bench", "sort", "all"), ("solve_bench", "solve", "all")],
)
def test_non_linear_suites_resolve_memory_target_in_run_benchmarks(
    module_name: str, expected_name: str, precision: str
) -> None:
    module = _module(module_name)
    suite = _make_recording_suite()
    target_bytes = 1 << 20
    request = _sizing().SizeRequest(memory_target_bytes=[target_bytes])

    module.run_benchmarks(suite, request, variant="all", precision=precision)

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    resolution = suite.resolutions[0][0]
    assert resolution.requested_memory_target_bytes == target_bytes
    assert resolution.estimated_working_set_bytes <= target_bytes
    assert suite.calls
    call_name, call_args = suite.calls[0]
    assert call_name == expected_name
    assert call_args[2] == [resolution.resolved_size]

    def estimate(size: int) -> int:
        return module._estimate_working_set_bytes("all", precision, size)

    larger_size = _next_size_with_larger_estimate(
        estimate, resolution.resolved_size
    )
    assert estimate(larger_size) > target_bytes


def test_axis_sum_suite_resolves_memory_target_in_run_benchmarks() -> None:
    axis_sum = _module("axis_sum_bench")
    suite = _make_recording_suite()
    target_bytes = 1 << 20
    request = _sizing().SizeRequest(memory_target_bytes=[target_bytes])

    axis_sum.run_benchmarks(suite, request, case="all", perform_check=False)

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    resolution = suite.resolutions[0][0]
    assert resolution.requested_memory_target_bytes == target_bytes
    assert resolution.estimated_working_set_bytes <= target_bytes
    assert suite.calls
    call_name, call_args = suite.calls[0]
    assert call_name == "axis_sum"
    assert call_args[6] == resolution.resolved_size

    def estimate(size: int) -> int:
        return axis_sum._estimate_working_set_bytes("all", size)

    larger_size = _next_size_with_larger_estimate(
        estimate, resolution.resolved_size
    )
    assert estimate(larger_size) > target_bytes


def test_batched_fft_suite_resolves_memory_target_in_run_benchmarks() -> None:
    batched_fft = _module("batched_fft_bench")
    suite = _make_recording_suite()
    target_bytes = 1 << 20
    request = _sizing().SizeRequest(memory_target_bytes=[target_bytes])

    batched_fft.run_benchmarks(suite, request)

    assert len(suite.resolutions) == 1
    assert len(suite.resolutions[0]) == 1
    resolution = suite.resolutions[0][0]
    assert resolution.requested_memory_target_bytes == target_bytes
    assert resolution.estimated_working_set_bytes <= target_bytes
    assert len(suite.calls) == len(batched_fft._CASES)
    call_name, call_args = suite.calls[0]
    assert call_name == "batched_fft"
    expected_shape = batched_fft._case_shape(
        resolution.resolved_size, call_args[1]
    )
    assert call_args[3] == expected_shape[0]
    assert call_args[4] == expected_shape[1]


def test_resolve_linear_suite_size_reports_resolution_details() -> None:
    sizing = _sizing()
    sizes, resolutions = sizing.resolve_linear_suite_size(
        sizing.SizeRequest(memory_target_bytes=[100]),
        bytes_per_element=9,
        describe_size=lambda resolved_size: [f"shape: {resolved_size}"],
    )

    assert sizes == [11]
    assert resolutions is not None
    assert len(resolutions) == 1
    assert resolutions[0].requested_memory_target_bytes == 100
    assert resolutions[0].estimated_working_set_bytes == 99
    assert resolutions[0].detail_lines == ("shape: 11",)


def test_resolve_size_by_binary_search_finds_largest_fitting_size() -> None:
    sizing = _sizing()

    def estimate(size: int) -> int:
        return size * size + size

    _assert_target_resolution(
        lambda target_bytes: sizing.resolve_size_by_binary_search(
            target_bytes,
            estimate_working_set_bytes=estimate,
            initial_guess=target_bytes // 4,
        ),
        estimate,
        target_bytes=1_000,
    )


def test_stream_target_resolution_for_noncontiguous_layout() -> None:
    stream_bench = _module("stream_bench")
    target_bytes = 1 << 20
    size = stream_bench._resolve_size_from_memory_target(
        "all", "all", target_bytes
    )

    side = math.isqrt(size)
    assert side * side == size
    assert stream_bench.get_noncontiguous_shape(size) == (side, side)
    assert (
        stream_bench._estimate_working_set_bytes("all", size) <= target_bytes
    )

    next_size = (side + 1) * (side + 1)
    assert (
        stream_bench._estimate_working_set_bytes("all", next_size)
        > target_bytes
    )


def test_gemm_target_resolution_for_all_variants() -> None:
    gemm_gemv = _module("gemm_gemv_bench")

    def estimate(size: int) -> int:
        return gemm_gemv._estimate_working_set_bytes("all", "all", size)

    _assert_target_resolution(
        lambda target_bytes: gemm_gemv._resolve_size_from_memory_target(
            "all", "all", target_bytes
        ),
        estimate,
        target_bytes=8 << 20,
    )


def test_gemm_initializers_normalize_in_place() -> None:
    gemm_gemv = _module("gemm_gemv_bench")
    matrix = MagicMock(name="matrix")
    matrix.reshape.return_value = matrix
    matrix.__itruediv__.return_value = matrix
    vector = MagicMock(name="vector")
    vector.__itruediv__.return_value = vector
    tracker = SimpleNamespace(
        float64=object(), arange=Mock(side_effect=[matrix, vector])
    )

    gemm_gemv._make_matrix(tracker, 8, 16, tracker.float64, 1)
    gemm_gemv._make_vector(tracker, 16, tracker.float64, 2)

    matrix.__itruediv__.assert_called_once()
    matrix.__truediv__.assert_not_called()
    vector.__itruediv__.assert_called_once()
    vector.__truediv__.assert_not_called()


def test_undersized_stream_target_raises() -> None:
    sizing = _sizing()
    stream_bench = _module("stream_bench")
    with pytest.warns(
        RuntimeWarning,
        match="memory target is smaller than estimated working set",
    ):
        _, resolutions = sizing.resolve_suite_size(
            sizing.SizeRequest(memory_target_bytes=[1]),
            resolve_from_target=lambda target_bytes: (
                stream_bench._resolve_size_from_memory_target(
                    "all", "false", target_bytes
                )
            ),
            estimate_working_set_bytes=lambda size: (
                stream_bench._estimate_working_set_bytes("all", size)
            ),
        )
    assert resolutions is not None
    assert len(resolutions) == 1
    assert resolutions[0].estimated_working_set_bytes > 1


def test_undersized_gemm_target_warns() -> None:
    sizing = _sizing()
    gemm_gemv = _module("gemm_gemv_bench")
    with pytest.warns(
        RuntimeWarning,
        match="memory target is smaller than estimated working set",
    ):
        _, resolutions = sizing.resolve_suite_size(
            sizing.SizeRequest(memory_target_bytes=[1]),
            resolve_from_target=lambda target_bytes: (
                gemm_gemv._resolve_size_from_memory_target(
                    "skinny_gemm", "64", target_bytes
                )
            ),
            estimate_working_set_bytes=lambda size: (
                gemm_gemv._estimate_working_set_bytes(
                    "skinny_gemm", "64", size
                )
            ),
        )
    assert resolutions is not None
    assert len(resolutions) == 1
    assert resolutions[0].estimated_working_set_bytes > 1


def test_fast_advanced_indexing_uses_square_size_for_2d_cases() -> None:
    fast_advanced_indexing = _module("fast_advanced_indexing_bench")
    suite = _make_recording_suite(forbid_resolution=True)

    fast_advanced_indexing.run_benchmarks(
        suite, _sizing().SizeRequest(exact_size=[10_000])
    )

    call_map = {name: args for name, args in suite.calls}
    assert call_map["putmask_scalar"][1] == [10_000]
    assert call_map["take_1d"][1] == 10_000
    for name in ("einsum_2d", "take_2d", "take_along_axis"):
        assert call_map[name][1] == 100


def test_fast_advanced_indexing_clamps_small_targets_to_nonzero_indices() -> (
    None
):
    fast_advanced_indexing = _module("fast_advanced_indexing_bench")
    suite = _make_recording_suite()

    fast_advanced_indexing.run_benchmarks(
        suite, _sizing().SizeRequest(memory_target_bytes=[100])
    )

    call_map = {name: args for name, args in suite.calls}
    assert call_map["take_1d"][2] == 1
    for name in ("einsum_2d", "take_2d", "take_along_axis"):
        assert call_map[name][2] == 1


def test_general_indexing_clamps_small_targets_to_nonzero_indices() -> None:
    general_indexing = _module("general_indexing_bench")
    suite = _make_recording_suite()

    general_indexing.run_benchmarks(
        suite, _sizing().SizeRequest(memory_target_bytes=[100])
    )

    call_map = {name: args for name, args in suite.calls}
    for name in (
        "mixed_indexing",
        "non_contiguous_indexing",
        "array_get_1d",
        "array_set_1d",
        "row_select_2d",
        "scalar_list_set_2d",
    ):
        assert call_map[name][2] == 1


def test_axis_sum_normalizes_negative_axes_for_output_shape() -> None:
    axis_sum = _module("axis_sum_bench")

    assert axis_sum._normalized_axes(-1, 3) == (2,)
    assert axis_sum._normalized_axes((0, -1), 3) == (0, 2)


def test_main_dispatches_memory_target_request(mocker) -> None:
    main = _module("main")
    requests = []

    class FakeSuite:
        name = "fake"

        @staticmethod
        def add_suite_parser_group(parser) -> None:
            del parser

        def __init__(self, config, args) -> None:
            del config, args
            self.benchmark_count = 1

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> None:
            del exc_info

        def run_suite(self, size_request) -> None:
            requests.append(size_request)

    config = SimpleNamespace(
        summarize=None,
        summarize_flush=main.SummarizeFlush.NEVER,
        repeat=0,
        package="numpy",
        runs=1,
        warmup=0,
        print_panel=lambda *args, **kwargs: None,
    )
    mocker.patch.object(main.MicrobenchmarkConfig, "add_parser_group")
    mocker.patch.object(
        main.MicrobenchmarkConfig, "from_args", return_value=config
    )
    mocker.patch.object(main, "SUITE_CLASSES", [FakeSuite])

    assert main.main(["--suite", "fake", "--memory-size", "64MiB"]) == 0
    assert len(requests) == 1
    assert requests[0].memory_target_bytes == [64 << 20]


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
