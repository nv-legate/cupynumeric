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

import json
import math
from functools import cache, reduce
from typing import Protocol
from itertools import product
from subprocess import CalledProcessError, check_output

from legate.core import get_machine
from legate.util.benchmark import BenchmarkLog, benchmark_log
from legate.util.info import info as legate_info
from legate.util.settings import SettingBase
from cupynumeric.settings import settings as cupynumeric_settings

FAILED_TO_DETECT = "(failed to detect)"


@cache
def _conda_list():
    try:
        if out := check_output(["conda", "list", "--json"]):
            info = json.loads(out.decode("utf-8"))
            names = [pkg["name"] for pkg in info]
            versions = [pkg["version"] for pkg in info]
            channels = [pkg["channel"] for pkg in info]
            builds = [pkg["build_string"] for pkg in info]
            version_len = max([len(v) for v in versions])
            build_len = max([len(b) for b in builds])
            entries = [
                f"{v:{version_len}}  {b:{build_len}}  {c}"
                for v, b, c in zip(versions, builds, channels)
            ]
            return dict(zip(names, entries))

    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(conda missing)"
    else:
        return FAILED_TO_DETECT


_metadata = {}

_backend = None


def _cupynumeric_settings_info():
    out = {}
    for att_name in dir(cupynumeric_settings):
        att = getattr(cupynumeric_settings, att_name)
        if isinstance(att, SettingBase):
            out[att.name] = att()
    return out


class Timer(Protocol):
    def start(self): ...

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
    """
    Parse arguments from `parser`, adding common command-line arguments for all examples
    that use the `run_benchmark()` utility.

    Parameters
    ----------
    parser:
        `argparse` parser that may have arguments already added to it.
        Additional arguments (specified below) will be added to it.

    Command-Line Arguments
    ----------------------
    -b/--benchmark: int (default: 0)
        Number of times to benchmark an example.  `0` corresponds to running
        the example without creating a log of benchmark performance data.
        Anything greater than 0 will create a log from running the example
        that many times.

    --package: 'legate', 'numpy', or 'cupy' (default: 'legate')
        Specify the package that will provide the `np` implementation
        used by the example.

    --cupy-allocator: 'default', 'off', or 'managed' (default: 'default')
        cupy allocator to use

    --log-conda-list: bool
        Add the output of `conda list` to the benchmark log's metadata

    --log-metadata-extra [key1=value1 [key2=value2 ...]]
        Additional data to added to the benchmark log's metadata

    Returns
    -------
    (args, np, timer)

    args:
        parsed arguments

    np:
        module implementing numpy-like operations (it is expected that
        `parse_args()` will be called from global scope, so that this `np` will
        be useable by all functions in the example)

    timer:
        a timer that matches the particular `np` implementation
    """
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=0,
        dest="benchmark",
        help="number of times to benchmark this application (default 0 - "
        "execute without benchmarking)",
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
    parser.add_argument(
        "--log-conda-list",
        dest="log_conda_list",
        action="store_true",
        help="add `conda list` to the benchmark metadata log",
    )
    parser.add_argument(
        "--log-metadata-extra",
        dest="metadata_extra",
        type=str,
        nargs="+",
        default=[],
        help="additional strings to add to benchmark log metadata, in key=value format",
    )
    args, _ = parser.parse_known_args()
    global _backend
    if args.package == "legate":
        import cupynumeric as np

        timer = CuPyNumericTimer()
        force_thunk = cupynumeric_settings.force_thunk()
        target = get_machine().preferred_target.name
        if force_thunk == "eager":
            _backend = "cupynumeric.eager"
        elif force_thunk == "deferred":
            _backend = f"cupynumeric.{target}_deferred"
        else:
            _backend = f"cupynumeric.{target}"
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
        _backend = "cupy"
    elif args.package == "numpy":
        import numpy as np

        timer = NumPyTimer()
        _backend = "numpy"
    if args.benchmark > 0:
        global _metadata
        _metadata.update(legate_info())
        _metadata["CuPyNumeric settings"] = _cupynumeric_settings_info()
        if args.log_conda_list:
            _metadata["Conda list"] = _conda_list()
        for s in args.metadata_extra:
            tokens = s.split("=")
            key = tokens[0]
            value = tokens[1]
            _metadata[key] = value
    return args, np, timer


class BenchmarkLogNull(BenchmarkLog):
    def _log_metadata(self, _metadata: str) -> None:
        pass

    def _log_columns(self, _columns: list[str]) -> None:
        pass

    def _log_row(self, _row: list[str]) -> None:
        pass


# A helper method for benchmarking applications
def run_benchmark(f, samples, name, inputs, output_columns, **kwargs):
    """
    Run and benchmark an example function

    Parameters
    ----------
    f: Callable
        The function to benchmark. `f` can have positional arguments and
        keyword arguments.  It is expected that each positional argument should
        generate a column in the table of benchmark data and keyword arguments
        should not.  It is also expected that each return value of `f` should
        generate a column in the table.
    samples: int
        Number of times to benchmark `f`. `0` indicates running `f` without
        generating a benchmark log; otherwise `f` a benchmark log will be generated
        with this many instances
    name: str
        The name of the benchmark
    inputs: list[tuple(str, Any)]
        Names and values for each of the positional arguments of `f`: names become column
        names in the data table
    output_columns: list[str]
        Names for each of the values returned by `f`
    kwargs:
        All keyword arguments are forwarded to `f`
    """
    bmark = None
    input_columns = [inp[0] for inp in inputs]
    input_args = [inp[1] for inp in inputs]
    columns = ["array package"] + input_columns + list(output_columns)

    if samples == 0:
        import os

        bmark = BenchmarkLogNull(name, 0, columns, os.devnull, {})
    else:
        bmark = benchmark_log(name, columns, metadata=_metadata)
    samples = max(1, samples)

    arg_lists = []

    for arg in input_args:
        if not isinstance(arg, list):
            arg_lists.append([arg])
        else:
            arg_lists.append(arg)

    plan = product(*arg_lists)

    with bmark as b:
        for args in plan:
            input_dict = {"array package": _backend}
            input_dict.update(dict(zip(input_columns, args)))
            times = []
            for _i in range(samples):
                output_vals = f(*args, **kwargs)
                if not isinstance(output_vals, tuple):
                    output_vals = (output_vals,)
                output_dict = dict(zip(output_columns, output_vals))
                b.log(**{**input_dict, **output_dict})
                times.append(output_vals[0])
            if samples > 1:
                # Remove the largest and the smallest ones
                if samples >= 3:
                    times.remove(max(times))
                if samples >= 2:
                    times.remove(min(times))
                mean = sum(times) / len(times)
                variance = sum(map(lambda x: (x - mean) ** 2, times)) / len(
                    times
                )
                stddev = math.sqrt(variance)
                print("-----------------------------------------------")
                print("BENCHMARK RESULTS: " + name)
                print("Total Samples: " + str(samples))
                print("Average Time: " + str(mean) + " ms")
                print("Variance: " + str(variance) + " ms")
                print("Stddev: " + str(stddev) + " ms")
                print(
                    "All Results: "
                    + reduce(
                        lambda x, y: x + y, map(lambda x: str(x) + ", ", times)
                    )
                )
                print("-----------------------------------------------")
