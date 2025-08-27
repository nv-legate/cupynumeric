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

from benchmark import parse_args, run_benchmark


def take_pairs(M: int, N: int) -> None:
    data_array = np.arange(M * N, dtype=np.int32).reshape((M, N))
    indices = np.array(list(a for a in itertools.combinations(range(N), 2)))
    data_array.take(indices, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--num-rows",
        type=int,
        default=10000,
        dest="M",
        help="number of rows in test matrix",
    )
    parser.add_argument(
        "-n",
        "--num-cols",
        type=int,
        default=32,
        dest="N",
        help="number of columns in test matrix",
    )

    args, np, timer = parse_args(parser)

    run_benchmark(take_pairs, args.benchmark, "take pairs", (args.M, args.N))
