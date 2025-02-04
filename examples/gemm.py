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

from benchmark import parse_args, run_benchmark


def initialize(M, N, K, ft):
    A = np.random.uniform(size=(N, N), dtype=ft)
    B = np.random.uniform(size=(N, N), dtype=ft)
    C = np.zeros((N, N), dtype=ft)
    return A, B, C


def total_flops(M, N, K):
    return M * N * (2 * K - 1)


def total_space(M, N, K, ft):
    return (M * N + M * K + K * N) * np.dtype(ft).itemsize


def run_gemm(N, I, warmup, ft):  # noqa: E741
    print("Problem Size:     M=" + str(N) + " N=" + str(N) + " K=" + str(N))
    print("Total Iterations: " + str(I))
    flops = total_flops(N, N, N)
    print("Total Flops:      " + str(flops / 1e9) + " GFLOPS/iter")
    space = total_space(N, N, N, ft)
    print("Total Size:       " + str(space / 1e6) + " MB")
    A, B, C = initialize(N, N, N, ft)

    timer.start()
    # Run for as many iterations as was requested
    for idx in range(I + warmup):
        if idx == warmup:
            timer.start()
        np.dot(A, B, out=C)
        # We need to rotate the matrices to keep Legate honest
        # about moving data so it can't just duplicate A and B
        # on the first iteration and reuse them, this means
        # that A, B, C all need to be square
        A, B, C = B, C, A
    total = timer.stop()

    print("Elapsed Time:     " + str(total) + " ms")
    average = total / I
    print("Average GEMM:     " + str(average) + " ms")
    print("FLOPS/s:          " + str(flops / (average * 1e6)) + " GFLOPS/s")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
        dest="I",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=5,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        dest="P",
        help="number of bits of precision to use for the gemm computation "
        "(16,32,64)",
    )

    args, np, timer = parse_args(parser)

    if args.P == 16:
        run_benchmark(
            run_gemm,
            args.benchmark,
            "HGEMM",
            (args.N, args.I, args.warmup, np.float16),
        )
    elif args.P == 32:
        run_benchmark(
            run_gemm,
            args.benchmark,
            "SGEMM",
            (args.N, args.I, args.warmup, np.float32),
        )
    elif args.P == 64:
        run_benchmark(
            run_gemm,
            args.benchmark,
            "DGEMM",
            (args.N, args.I, args.warmup, np.float64),
        )
    else:
        raise TypeError("Precision must be one of 16, 32, or 64")
