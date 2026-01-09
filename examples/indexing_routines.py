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

import argparse
import gc
import math

from benchmark import parse_args, run_benchmark


def compute_diagonal(steps, N, timing, warmup):
    A1 = np.ones((N,), dtype=int)
    print("measuring diagonal")
    for step in range(steps + warmup):
        if step == warmup:
            timer.start()
        A2 = np.diag(A1)
        A1 = np.diag(A2)
    total = timer.stop()
    if timing:
        space = (N * N + N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for diagonal: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "bandwidth:               "
            + str(space * 1000.0 / (average))
            + " GB/s"
        )
    return total


def compute_choose(steps, N, timing, warmup):
    print("measuring choose")
    A = tuple(np.full((N,), k) for k in range(10))
    C1 = np.arange(N, dtype=int) % 10
    for step in range(steps + warmup):
        if step == warmup:
            timer.start()
        C1 = np.choose(C1, A, mode="wrap")
    total = timer.stop()
    if timing:
        space = N * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for choose: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def compute_repeat(steps, N, timing, warmup):
    A2 = np.ones((N, N), dtype=int)
    R = np.repeat(int(1), N)
    print("measuring repeat")
    for step in range(steps + warmup):
        if step == warmup:
            timer.start()
        A2 = np.repeat(A2, R, axis=1)
    total = timer.stop()
    if timing:
        space = (N * N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for repeat: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def compute_advanced_indexing_1d(steps, N, timing, warmup):
    A1 = np.ones((N,), dtype=int)
    B = np.arange(N, dtype=int)
    print("measuring advanced_indexing 1D")
    indx = B % 10
    indx_bool = (B % 2).astype(bool)
    for step in range(steps + warmup):
        if step == warmup:
            timer.start()
        A1[indx] = 10  # 1 copy
        A1[indx_bool] = 12  # 1 AI and 1 copy
    total = timer.stop()
    if timing:
        space = (3 * N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for advanced_indexing: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def compute_advanced_indexing_2d(steps, N, timing, warmup):
    A2 = np.ones((N, N), dtype=int)
    B = np.arange(N, dtype=int)
    print("measuring advanced_indexing 2D")
    indx = B % 10
    indx_bool = (B % 2).astype(bool)
    indx2d_bool = (A2 % 2).astype(bool)
    for step in range(steps + warmup):
        if step == warmup:
            timer.start()
        A2[indx_bool, indx_bool] = 11  # one ZIP and 1 copy = N+N*N
        A2[:, indx] = 12  # one ZIP and 3 copies = N+3*N*N
        A2[indx2d_bool] = 13  # 1 copy and one AI task = 2* N*N
    total = timer.stop()
    if timing:
        space = (6 * N * N + 2 * N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for advanced_indexing: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def compute_advanced_indexing_3d(steps, N, timing, warmup):
    A3 = np.ones((N, int(N / 100), 100), dtype=int)
    B = np.arange(N, dtype=int)
    print("measuring advanced_indexing_3d")
    indx = B % 10
    indx3d_bool = (A3 % 2).astype(bool)
    for step in range(steps + warmup):
        if step == warmup:
            timer.start()
        A3[indx, :, indx] = 15  # 1 ZIP and 3 copy = N+3N*N
        A3[indx3d_bool] = 16  # 1 copy and 1 AI task = 2*N*N
    total = timer.stop()
    if timing:
        space = (5 * N * N + N) * np.dtype(int).itemsize / 1073741824
        print("Total Size:       " + str(space) + " GB")
        print("Elapsed Time for advanced_indexing: " + str(total) + " ms")
        average = total / steps
        print("Average            :     " + str(average) + " ms")
        print(
            "bandwidth:               "
            + str(space * 1000 / (average))
            + " GB/s"
        )
    return total


def run_indexing_routines(
    N, steps, warmup, routine, *, print_timing=False, verbose=False
):
    # simple operation to warm up the library
    assert not math.isnan(np.sum(np.zeros((N, N)).dot(np.zeros((N,)))))
    gc.collect()
    time = 0
    if routine == "diagonal" or routine == "all":
        time += compute_diagonal(steps, N, print_timing, warmup)
    if routine == "choose" or routine == "all":
        time += compute_choose(steps, N, print_timing, warmup)
    if routine == "repeat" or routine == "all":
        time += compute_repeat(steps, N, print_timing, warmup)
    if routine == "ai1" or routine == "all":
        time += compute_advanced_indexing_1d(steps, N, print_timing, warmup)
    if routine == "ai2" or routine == "all":
        time += compute_advanced_indexing_2d(steps, N, print_timing, warmup)
    if routine == "ai3" or routine == "all":
        time += compute_advanced_indexing_3d(steps, N, print_timing, warmup)
    if print_timing:
        print("Total Elapsed Time: " + str(time) + " ms")
    return time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100,
        dest="I",
        help="number of iterations to run the algorithm for",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=10,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="print verbose output",
    )
    parser.add_argument(
        "-r",
        "--routine",
        default="all",
        dest="routine",
        choices=["diagonal", "choose", "repeat", "ai1", "ai2", "ai3", "all"],
        help="name of the index routine to test",
    )

    args, np, timer = parse_args(parser)

    run_benchmark(
        run_indexing_routines,
        args.benchmark,
        "Indexing Routines",
        [
            ("problem size", args.N),
            ("iterations", args.I),
            ("warmup iterations", args.warmup),
            ("routine", args.routine),
        ],
        ["time (milliseconds)"],
        print_timing=args.timing,
        verbose=args.verbose,
    )
