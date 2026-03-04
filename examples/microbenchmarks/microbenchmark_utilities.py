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
Shared benchmark utilities for microbenchmark suite.

This module provides a lightweight coordination layer for running multiple
benchmarks within a suite using the standard benchmark.py infrastructure.

Key responsibilities:
1. MicrobenchmarkSuite: Coordinates benchmark execution
   - Each benchmark uses run_benchmark() to create separate CSV files
   - Tracks benchmark count for suite summary

2. create_benchmark_function: Helper to create benchmark functions that follow
   the standard pattern (warmup + timed runs)

Design:
- Each benchmark calls run_benchmark() which creates a separate CSV file
- This preserves compatibility with existing visualization tools
- Optional unified summary table can be generated at the end by parsing CSV files
"""

import sys
from pathlib import Path

# Add parent directory to path to import benchmark.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import run_benchmark


class MicrobenchmarkSuite:
    """
    Coordination layer for running multiple microbenchmarks in a suite.

    Each benchmark calls run_benchmark() which creates a separate CSV file.
    This class provides suite-level coordination and summary.
    """

    def __init__(self, suite_name, args, np_module, timer):
        """
        Initialize microbenchmark suite coordinator.

        Parameters
        ----------
        suite_name : str
            Name of the benchmark suite
        args : argparse.Namespace
            Parsed command-line arguments from parse_args()
        np_module : module
            NumPy-compatible module (cupynumeric, cupy, or numpy)
        timer : Timer
            Backend-appropriate timer instance
        """
        self.suite_name = suite_name
        self.args = args
        self.np = np_module
        self.timer = timer
        self.runs = getattr(args, "runs", 5)
        self.warmup = getattr(args, "warmup", 2)
        self.benchmark_samples = args.benchmark
        self.benchmark_count = 0
        self._benchmark_names = []  # Track benchmark names for summary

    def run_single_benchmark(self, name, bench_func, size_params=None):
        """
        Run a single benchmark using standard run_benchmark() infrastructure.

        This creates a separate CSV file for the benchmark (standard behavior).

        Parameters
        ----------
        name : str
            Benchmark name (without suite prefix)
        bench_func : callable
            Function that runs the benchmark and returns TOTAL time in ms
            Signature: func() -> float (total_time_ms)
        size_params : dict, optional
            Size-related parameters to include in CSV (e.g., {"size": 10000, "n": 100})
        """
        full_name = f"{self.suite_name}_{name}"

        # Wrapper to accept and ignore arguments (run_benchmark expects this)
        def wrapper(*args, **kwargs):
            return bench_func()

        # Build input columns: size params + run config
        inputs = []
        if size_params:
            inputs.extend([(k, v) for k, v in size_params.items()])
        inputs.extend([("runs", self.runs), ("warmup", self.warmup)])
        # Use run_benchmark - creates separate CSV file
        run_benchmark(
            wrapper,
            self.benchmark_samples,
            full_name,
            inputs,
            ["time (milliseconds)"],
        )

        self.benchmark_count += 1
        self._benchmark_names.append(full_name)

    def get_benchmark_names(self):
        """Get list of all benchmark names run by this suite."""
        return self._benchmark_names

    def print_suite_summary(self):
        """Print suite-level summary."""
        if self.benchmark_count > 0:
            print("\n" + "=" * 80)
            print(f"SUITE COMPLETE: {self.suite_name}")
            print(f"Total benchmarks run: {self.benchmark_count}")
            print("=" * 80)


def create_benchmark_function(np, timer, operation, runs, warmup):
    """
    Create a benchmark function compatible with run_benchmark().

    Follows the standard pattern from examples/gemm.py and examples/einsum.py:
    - Do warmup internally
    - Time all iterations together as one block
    - Return TOTAL time for all iterations

    Parameters
    ----------
    np : module
        NumPy-compatible module
    timer : Timer
        Backend-appropriate timer
    operation : callable
        Operation to benchmark (should be a lambda or function)
    runs : int
        Number of timing runs (iterations)
    warmup : int
        Number of warmup runs

    Returns
    -------
    callable
        Function that returns total time in ms for all runs
    """

    def bench_func():
        # Run warmup + iterations together (like gemm.py)
        for idx in range(runs + warmup):
            if idx == warmup:
                timer.start()  # Start timing AFTER warmup
            operation()
        total = timer.stop()

        # Return TOTAL time (run_benchmark will compute average across samples)
        return total

    return bench_func
