#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation
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

import math
from functools import reduce
from typing import Protocol


class Timer(Protocol):
    def start(self):
        ...

    def stop(self):
        """
        Blocks execution until everything before it has completed. Returns the
        duration since the last call to start(), in milliseconds.
        """
        ...


class CuPyNumericTimer(Timer):
    def __init__(self):
        self._start_time = None

    def start(self):
        from legate.timing import time

        self._start_time = time("us")

    def stop(self):
        from legate.timing import time

        end_future = time("us")
        return (end_future - self._start_time) / 1000.0


class CuPyTimer(Timer):
    def __init__(self):
        self._start_event = None

    def start(self):
        from cupy import cuda

        self._start_event = cuda.Event()
        self._start_event.record()

    def stop(self):
        from cupy import cuda

        end_event = cuda.Event()
        end_event.record()
        end_event.synchronize()
        return cuda.get_elapsed_time(self._start_event, end_event)


class NumPyTimer(Timer):
    def __init__(self):
        self._start_time = None

    def start(self):
        from time import perf_counter_ns

        self._start_time = perf_counter_ns() / 1000.0

    def stop(self):
        from time import perf_counter_ns

        end_time = perf_counter_ns() / 1000.0
        return (end_time - self._start_time) / 1000.0


# Add common arguments and parse
def parse_args(parser):
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 - "
        "normal execution)",
    )
    parser.add_argument(
        "--package",
        dest="package",
        choices=["legate", "numpy", "cupy"],
        type=str,
        default="legate",
        help="NumPy package to use",
    )
    parser.add_argument(
        "--cupy-allocator",
        dest="cupy_allocator",
        choices=["default", "off", "managed"],
        type=str,
        default="default",
        help="cupy allocator to use",
    )
    args, _ = parser.parse_known_args()
    if args.package == "legate":
        import cupynumeric as np

        timer = CuPyNumericTimer()
    elif args.package == "cupy":
        import cupy as np

        if args.cupy_allocator == "off":
            np.cuda.set_allocator(None)
            print("Turning off memory pool")
        elif args.cupy_allocator == "managed":
            np.cuda.set_allocator(
                np.cuda.MemoryPool(np.cuda.malloc_managed).malloc
            )
            print("Using managed memory pool")
        timer = CuPyTimer()
    elif args.package == "numpy":
        import numpy as np

        timer = NumPyTimer()
    return args, np, timer


# A helper method for benchmarking applications
def run_benchmark(f, samples, name, args):
    if samples > 1:
        results = [f(*args) for s in range(samples)]
        # Remove the largest and the smallest ones
        if samples >= 3:
            results.remove(max(results))
        if samples >= 2:
            results.remove(min(results))
        mean = sum(results) / len(results)
        variance = sum(map(lambda x: (x - mean) ** 2, results)) / len(results)
        stddev = math.sqrt(variance)
        print("-----------------------------------------------")
        print("BENCHMARK RESULTS: " + name)
        print("Total Samples: " + str(samples))
        print("Average Time: " + str(mean) + " ms")
        print("Variance: " + str(variance) + " ms")
        print("Stddev: " + str(stddev) + " ms")
        print(
            "All Results: "
            + reduce(lambda x, y: x + y, map(lambda x: str(x) + ", ", results))
        )
        print("-----------------------------------------------")
    else:
        # Just run the application like normal
        f(*args)
