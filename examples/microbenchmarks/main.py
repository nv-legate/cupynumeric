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
"""
Unified entry point for all microbenchmarks.

Examples:
    # Run with cupynumeric (default)
    python main.py --suite advanced_indexing

    # Compare with numpy backend
    python main.py --suite advanced_indexing --package numpy

    # Benchmark mode with structured logging (5 samples)
    python main.py --suite all --benchmark 5

    # Large problem size
    python main.py --suite advanced_indexing --size 100000000

    # Multi-GPU
    legate --gpus 4 main.py --suite all --benchmark 10
"""

import argparse
import sys
import traceback

from dataclasses import replace
from pathlib import Path

# Add parent directory to path to import benchmark.py
# (Relative imports don't work for scripts run directly)
sys.path.insert(0, str(Path(__file__).parent.parent))

from _benchmark import (
    MicrobenchmarkConfig,
    MicrobenchmarkSuite,
    SummarizeFlush,
)

# Import benchmark suites
#
# from general_scalared_bench import run_benchmarks as run_general_scalared

from axis_sum_bench import AxisSumSuite
from fast_advanced_indexing_bench import FastAdvancedIndexingSuite
from general_indexing_bench import GeneralIndexingSuite
from general_astype_bench import AsTypeSuite
from general_nanred_bench import NanRedSuite
from general_random_bench import RandomSuite
from general_scalared_bench import ScalarRedSuite
from gemm_gemv_bench import GemmSuite
from solve_bench import SolveSuite
from sort_bench import SortSuite
from stream_bench import StreamSuite
from sync_bench import SyncSuite
from ufunc_bench import UfuncSuite

SUITE_CLASSES: list[type[MicrobenchmarkSuite]] = [
    FastAdvancedIndexingSuite,
    AxisSumSuite,
    GeneralIndexingSuite,
    RandomSuite,
    GemmSuite,
    SolveSuite,
    SortSuite,
    StreamSuite,
    UfuncSuite,
    NanRedSuite,
    ScalarRedSuite,
    SyncSuite,
    AsTypeSuite,
]

# =============================================================================
# MAIN
# =============================================================================


class Formatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def main():
    parser = argparse.ArgumentParser(
        description="CuPyNumeric Microbenchmark Suite",
        formatter_class=Formatter,
        epilog=__doc__,
    )

    suite_names = [suite.name for suite in SUITE_CLASSES]

    # main options
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", *suite_names],
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10_000_000,
        help="Base problem size in elements",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit immediately after the first suite failure",
    )

    MicrobenchmarkConfig.add_parser_group(parser, "microbenchmark harness")

    for suite_class in SUITE_CLASSES:
        suite_class.add_suite_parser_group(parser)

    args = parser.parse_known_args()[0]

    # general configuration that affects all suites
    config = MicrobenchmarkConfig.from_args(args)

    # handle joint summary of all suites
    summarize = config.summarize
    orig_flush = config.summarize_flush
    if config.summarize is not None:
        # Flush the summary at the end of all the suite
        config = replace(config, summarize_flush=SummarizeFlush.NEVER)

    suites: dict[str, MicrobenchmarkSuite] = {}
    for suite_class in SUITE_CLASSES:
        suite = suite_class(config, args)
        suites[suite.name] = suite

    config_msg = [
        f"Backend: {config.package}",
        f"Suite: {args.suite}",
        f"Size: {args.size:,} elements",
        f"Runs: {config.runs}",
        f"Warmup: {config.warmup}",
    ]
    if args.fail_fast:
        config_msg.append("Fail fast: enabled")
    if config.repeat > 0:
        config_msg.append(
            f"Benchmark samples: {config.repeat} (structured logging enabled)"
        )
    config.print_panel(config_msg, "CuPyNumeric Microbenchmark Suite")

    # Run selected suite(s)
    completed = []
    failures = []

    if args.suite == "all":
        for suite_name, suite in suites.items():
            try:
                with suite as s:
                    s.run_suite(args.size)
                completed.append(suite)
            except Exception as exc:
                print(f"\nError in {suite_name} suite: {exc}")
                traceback.print_exc()
                failures.append(suite_name)
                if args.fail_fast:
                    return 1
    else:
        if args.suite in suites:
            suite = suites[args.suite]
            with suite as s:
                s.run_suite(args.size)
                completed.append(s)
        else:
            print(f"Unknown suite: {args.suite}")
            return 1

    if summarize and orig_flush != SummarizeFlush.NEVER:
        summarize.flush(title="Microbenchmarks")

    if len(completed) > 1:
        total_benchmarks = sum(s.benchmark_count for s in completed)
        final_msg = [
            f"Total suites run: {len(completed)}",
            f"Total benchmarks: {total_benchmarks}",
        ]
        config.print_panel(final_msg, "OVERAL SUMMARY")

    if failures:
        failure_msg = []
        for suite_name in failures:
            failure_msg.append(f"- {suite_name}")
        config.print_panel(failure_msg, "FAILED SUITES")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
