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
CuPyNumeric Microbenchmark Suite

Unified entry point for all microbenchmarks.

Usage:
    python main.py [--suite SUITE] [--size SIZE] [--runs RUNS] [options]

Available Suites:
    all               - Run all available benchmarks
    advanced_indexing - Optimized indexing paths (putmask, einsum, take_task)
    gemm_gemv         - GEMM/GEMV microbenchmarks
    general_indexing  - General indexing (ADVANCED_INDEXING task + Copy)
    general_random    - General random generation
    general_nanred    - General nansum(), nanmean()
    scalar_red        - Scalar reductions: sum, prod, min, max, argmin, argmax
    stream            - STREAM-style bandwidth microbenchmarks

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

Standard Benchmark Options (from examples/benchmark.py):
    -b/--benchmark N    : Run N benchmark samples and create structured log
    --package           : Backend to use (legate|numpy|cupy)
    --log-conda-list    : Include conda environment in metadata
    --log-metadata-extra: Add custom metadata (key=value pairs)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import benchmark.py
# (Relative imports don't work for scripts run directly)
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import parse_args

# Import shared utilities
from microbenchmark_utilities import MicrobenchmarkSuite

# Import benchmark suites
from fast_advanced_indexing_bench import (
    run_benchmarks as run_advanced_indexing,
)
from gemm_gemv_bench import run_benchmarks as run_gemm_gemv
from general_indexing_bench import run_benchmarks as run_general_indexing

from general_random_bench import run_benchmarks as run_general_random
from stream_bench import run_benchmarks as run_stream

from general_nanred_bench import run_benchmarks as run_general_nanred

from general_scalared_bench import run_benchmarks as run_general_scalared

# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CuPyNumeric Microbenchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Suite-specific arguments
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=[
            "all",
            "advanced_indexing",
            "gemm_gemv",
            "general_indexing",
            "general_random",
            "general_nanred",
            "scalar_red",
            "stream",
        ],
        help="Benchmark suite to run (default: all)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10_000_000,
        help="Base problem size in elements (default: 10M)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timing runs per benchmark (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup runs (default: 2)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Print unified summary table at the end (parses all generated CSV files)",
    )
    parser.add_argument(
        "--gemm-gemv-variant",
        type=str,
        default="all",
        choices=["skinny_gemm", "square_gemm", "gemv", "all"],
        help="GEMM/GEMV variant to run (default: all)",
    )
    parser.add_argument(
        "--gemm-gemv-precision",
        type=str,
        default="32",
        choices=["32", "64", "all"],
        help="GEMM/GEMV precision in bits (default: 32)",
    )
    parser.add_argument(
        "--gemm-gemv-check",
        action="store_true",
        help="Validate GEMM/GEMV results after each timed sample",
    )
    parser.add_argument(
        "--stream-operation",
        type=str,
        default="all",
        choices=["copy", "mul", "scale", "add", "all"],
        help="STREAM operation to run (default: all)",
    )
    parser.add_argument(
        "--stream-precision",
        type=str,
        default="32",
        choices=["32", "64", "all"],
        help="STREAM precision in bits (default: 32)",
    )
    parser.add_argument(
        "--stream-contiguous",
        type=str,
        default="all",
        choices=["true", "false", "all"],
        help=(
            "STREAM layout to run; 'false' uses transpose-based "
            "non-contiguous views (default: all)"
        ),
    )
    parser.add_argument(
        "--stream-check",
        action="store_true",
        help="Validate STREAM results after each timed sample",
    )

    # Parse using standard infrastructure (adds --benchmark, --package, etc.)
    args, np, timer = parse_args(parser)

    print("=" * 80)
    print("CuPyNumeric Microbenchmark Suite")
    print("=" * 80)
    print("Configuration:")
    print(f"  Backend: {args.package}")
    print(f"  Suite: {args.suite}")
    print(f"  Size: {args.size:,} elements")
    print(f"  Runs: {args.runs}")
    print(f"  Warmup: {args.warmup}")
    if args.benchmark > 0:
        print(
            f"  Benchmark samples: {args.benchmark} (structured logging enabled)"
        )
    if args.suite in ("all", "gemm_gemv"):
        print(f"  GEMM/GEMV variant: {args.gemm_gemv_variant}")
        print(f"  GEMM/GEMV precision: {args.gemm_gemv_precision}")
        print(f"  GEMM/GEMV check: {args.gemm_gemv_check}")
    if args.suite in ("all", "stream"):
        print(f"  Stream operation: {args.stream_operation}")
        print(f"  Stream precision: {args.stream_precision}")
        print(f"  Stream contiguous: {args.stream_contiguous}")
        print(f"  Stream check: {args.stream_check}")
    print("=" * 80)

    # Available benchmark suites
    suites = {
        "advanced_indexing": run_advanced_indexing,
        "gemm_gemv": run_gemm_gemv,
        "general_indexing": run_general_indexing,
        "general_random": run_general_random,
        "general_nanred": run_general_nanred,
        "scalar_red": run_general_scalared,
        "stream": run_stream,
    }

    # Helper function to run a single suite
    def run_suite(suite_name):
        """Create and run a benchmark suite."""
        suite = MicrobenchmarkSuite(
            suite_name=suite_name, args=args, np_module=np, timer=timer
        )
        if suite_name == "gemm_gemv":
            suites[suite_name](
                suite,
                args.size,
                variant=args.gemm_gemv_variant,
                precision=args.gemm_gemv_precision,
                perform_check=args.gemm_gemv_check,
            )
        elif suite_name == "stream":
            suites[suite_name](
                suite,
                args.size,
                operation=args.stream_operation,
                precision=args.stream_precision,
                contiguous=args.stream_contiguous,
                perform_check=args.stream_check,
            )
        else:
            suites[suite_name](suite, args.size)
        return suite

    # Run selected suite(s)
    suite_coordinators = []

    if args.suite == "all":
        for suite_name in suites.keys():
            try:
                suite_coordinators.append(run_suite(suite_name))
            except Exception as e:
                print(f"\nError in {suite_name} suite: {e}")
                import traceback

                traceback.print_exc()
    else:
        if args.suite in suites:
            suite_coordinators.append(run_suite(args.suite))
        else:
            print(f"Unknown suite: {args.suite}")
            return 1

    # Print suite-level summaries
    for suite in suite_coordinators:
        suite.print_suite_summary()

    # Print unified summary table if requested (parses generated CSV files)
    if args.summarize and args.benchmark > 0 and suite_coordinators:
        print_unified_summary_table(suite_coordinators)

    # Print overall summary if multiple suites
    if len(suite_coordinators) > 1:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        total_benchmarks = sum(s.benchmark_count for s in suite_coordinators)
        print(f"Total suites run: {len(suite_coordinators)}")
        print(f"Total benchmarks: {total_benchmarks}")
        print("=" * 80)

    return 0


def print_unified_summary_table(suite_coordinators):
    """
    Parse generated CSV files and print a unified summary table.

    This reads all the separate CSV files created by run_benchmark() and
    consolidates them into a single summary table.
    """
    import os

    # Get output directory from environment variable
    output_dir = os.environ.get("LEGATE_BENCHMARK_OUT", None)

    if output_dir and output_dir != "stdout":
        # CSV files are in a directory - parse them
        all_data = parse_csv_files_from_directory(
            suite_coordinators, output_dir
        )
    else:
        # CSV data was printed to stdout - we can't parse it retroactively
        # User should redirect output to file and use visualize_benchmarks.py
        print("\n" + "=" * 80)
        print("UNIFIED SUMMARY TABLE")
        print("=" * 80)
        print(
            "Note: To generate a unified summary table, use directory output mode:"
        )
        print(
            "  LEGATE_BENCHMARK_OUT=./results python main.py --benchmark 5 --summarize"
        )
        print("")
        print(
            "This will create separate CSV files (one per benchmark) and print"
        )
        print("a unified summary table at the end.")
        print("=" * 80)
        return

    if not all_data:
        print("\nNo CSV data found to summarize.")
        return

    # Print unified table
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="\n🔬 All Benchmarks - Unified Summary")

        table.add_column("Suite", style="blue")
        table.add_column("Benchmark", style="magenta")
        table.add_column("Samples", justify="right", style="cyan")
        table.add_column("Avg (ms)", justify="right", style="yellow")
        table.add_column("Min (ms)", justify="right", style="green")
        table.add_column("Max (ms)", justify="right", style="red")

        for row in all_data:
            table.add_row(
                row["suite"],
                row["benchmark"],
                str(row["samples"]),
                f"{row['avg']:.3f}",
                f"{row['min']:.3f}",
                f"{row['max']:.3f}",
            )

        console.print(table)

    except ImportError:
        # Fallback to simple text table
        print("\n" + "=" * 90)
        print("ALL BENCHMARKS - UNIFIED SUMMARY")
        print("=" * 90)
        print(
            f"{'Suite':<20} {'Benchmark':<30} {'Samples':>7} {'Avg (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}"
        )
        print("-" * 90)

        for row in all_data:
            print(
                f"{row['suite']:<20} "
                f"{row['benchmark']:<30} "
                f"{row['samples']:>7} "
                f"{row['avg']:>10.3f} "
                f"{row['min']:>10.3f} "
                f"{row['max']:>10.3f}"
            )

        print("=" * 90)


def parse_csv_files_from_directory(suite_coordinators, output_dir):
    """Parse all CSV files generated by benchmarks in the output directory."""
    import csv
    from glob import glob

    all_data = []

    for suite in suite_coordinators:
        for bench_name in suite.get_benchmark_names():
            # Find CSV files matching this benchmark name
            pattern = f"{output_dir}/{bench_name}_*.csv"
            csv_files = glob(pattern)

            if not csv_files:
                continue

            # Parse the CSV file (use first match if multiple)
            csv_file = csv_files[0]
            try:
                with open(csv_file, "r") as f:
                    # Skip metadata header lines (start with #)
                    lines = []
                    for line in f:
                        if not line.startswith("#"):
                            lines.append(line)

                    if len(lines) < 2:  # Need header + at least one data row
                        continue

                    # Parse CSV
                    reader = csv.DictReader(lines)
                    rows = list(reader)

                    if not rows:
                        continue
                    # Extract timing data
                    times = [
                        float(row.get("time (milliseconds)", 0))
                        for row in rows
                    ]

                    # Compute statistics (with outlier removal like run_benchmark does)
                    filtered_times = times.copy()
                    if len(times) >= 3:
                        filtered_times.remove(max(filtered_times))
                    if len(times) >= 2:
                        filtered_times.remove(min(filtered_times))

                    avg = (
                        sum(filtered_times) / len(filtered_times)
                        if filtered_times
                        else 0
                    )

                    # Use the known suite name instead of splitting on "_".
                    suite_name = suite.suite_name
                    prefix = f"{suite_name}_"
                    if bench_name.startswith(prefix):
                        benchmark_name = bench_name[len(prefix) :]
                    else:
                        benchmark_name = bench_name

                    all_data.append(
                        {
                            "suite": suite_name,
                            "benchmark": benchmark_name,
                            "samples": len(times),
                            "avg": avg,
                            "min": min(times),
                            "max": max(times),
                        }
                    )

            except (IOError, ValueError, KeyError) as e:
                print(f"Warning: Could not parse {csv_file}: {e}")
                continue

    return all_data


if __name__ == "__main__":
    sys.exit(main())
