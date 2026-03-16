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

from pathlib import Path
from dataclasses import replace

# Add parent directory to path to import benchmark.py
# (Relative imports don't work for scripts run directly)
sys.path.insert(0, str(Path(__file__).parent.parent))

from _benchmark import (
    MicrobenchmarkHarness,
    MicrobenchmarkSuite,
    SummarizeFlush,
)

# Import benchmark suites
#
# from general_scalared_bench import run_benchmarks as run_general_scalared

from fast_advanced_indexing_bench import FastAdvancedIndexingSuite
from general_indexing_bench import GeneralIndexingSuite
from general_random_bench import RandomSuite
from gemm_gemv_bench import GemmSuite
from solve_bench import SolveSuite
from sort_bench import SortSuite
from stream_bench import StreamSuite
from ufunc_bench import UfuncSuite
from general_nanred_bench import NanRedSuite
from general_scalared_bench import ScalarRedSuite
from general_astype_bench import AsTypeSuite

SUITE_CLASSES: list[MicrobenchmarkSuite] = [
    FastAdvancedIndexingSuite,
    GeneralIndexingSuite,
    RandomSuite,
    GemmSuite,
    SolveSuite,
    SortSuite,
    StreamSuite,
    UfuncSuite,
    NanRedSuite,
    ScalarRedSuite,
    AsTypeSuite,
]

# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CuPyNumeric Microbenchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    suite_names = [suite.name for suite in SUITE_CLASSES]

    # main options
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", *suite_names],
        help="Benchmark suite to run (default: all)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10_000_000,
        help="Base problem size in elements (default: 10M)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit immediately after the first suite failure",
    )

    MicrobenchmarkHarness.add_parser_group(parser)

    for suite_class in SUITE_CLASSES:
        suite_class.add_suite_parser_group(parser)

    args = parser.parse_known_args()[0]

    # general configuration that affects all suites
    config = MicrobenchmarkHarness.config(args)

    # handle joint summary of all suites
    summarize = config.harness_config.summarize
    orig_flush = config.harness_config.summarize_flush
    if config.harness_config.summarize is not None:
        # Flush the summary at the end of all the suite
        new_harness_config = replace(
            config.harness_config, summarize_flush=SummarizeFlush.NEVER
        )
        config = replace(config, harness_config=new_harness_config)
        assert summarize == config.harness_config.summarize
    harness_config = config.harness_config

    suites: dict[str, MicrobenchmarkSuite] = {}
    for suite_class in SUITE_CLASSES:
        suite = suite_class(config, args)
        suites[suite.name] = suite

    config_msg = [
        f"Backend: {harness_config.package}",
        f"Suite: {args.suite}",
        f"Size: {args.size:,} elements",
        f"Runs: {config.runs}",
        f"Warmup: {config.warmup}",
    ]
    if args.fail_fast:
        config_msg.append("Fail fast: enabled")
    if harness_config.repeat > 0:
        config_msg.append(
            f"Benchmark samples: {harness_config.repeat} (structured logging enabled)"
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
            with suites[args.suite] as s:
                s.run_suite(args.size)
            completed.append(suite)
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
