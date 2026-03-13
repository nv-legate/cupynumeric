#!/usr/bin/env python

# Copyright 2025 NVIDIA Corporation
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

import argparse
import itertools

from _benchmark import benchmark_info, parse_with_harness


@benchmark_info(name="take pairs")
def take_pairs(
    np, M: int, N: int, *, timer, print_timing: bool = True
) -> float:
    timer.start()
    data_array = np.arange(M * N, dtype=np.int32).reshape((M, N))
    indices = np.array(list(a for a in itertools.combinations(range(N), 2)))
    result = data_array.take(indices, axis=1)
    value = result.sum()
    print(f"result sum: {value}")
    total = timer.stop()
    if print_timing:
        print(f"Elapsed Time: {total} ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--num-rows",
        type=int,
        nargs="+",
        default=[10000],
        dest="M",
        help="number of rows in test matrix",
    )
    parser.add_argument(
        "-n",
        "--num-cols",
        type=int,
        nargs="+",
        default=[32],
        dest="N",
        help="number of columns in test matrix",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )

    args, harness = parse_with_harness(parser)

    harness.run_timed(
        take_pairs,
        harness.np,
        args.M,
        args.N,
        timer=harness.timer,
        print_timing=args.timing,
    )
