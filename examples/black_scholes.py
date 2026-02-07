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


def generate_random(N, min, max, D):
    diff = D(max) - D(min)
    rands = np.random.random(N).astype(D)
    rands = rands * diff
    rands = rands + D(min)
    return rands


def initialize(N, D):
    S = generate_random(N, 5, 30, D)
    X = generate_random(N, 1, 100, D)
    T = generate_random(N, 0.25, 10, D)
    R = 0.02
    V = 0.3
    return S, X, T, R, V


def cnd(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438

    K = 1.0 / (1.0 + 0.2316419 * np.absolute(d))

    cnd = (
        RSQRT2PI
        * np.exp(-0.5 * d * d)
        * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    )

    return np.where(d > 0, 1.0 - cnd, cnd)


def black_scholes(S, X, T, R, V):
    sqrt_t = np.sqrt(T)
    d1 = np.log(S / X) + (R + 0.5 * V * V) * T / (V * sqrt_t)
    d2 = d1 - V * sqrt_t
    cnd_d1 = cnd(d1)
    cnd_d2 = cnd(d2)
    exp_rt = np.exp(-R * T)
    call_result = S * cnd_d1 - X * exp_rt * cnd_d2
    put_result = X * exp_rt * (1.0 - cnd_d2) - S * (1.0 - cnd_d1)
    return call_result, put_result


def run_black_scholes(N, D):
    print("Running black scholes on %dK options..." % N)
    N *= 1000
    timer.start()
    S, X, T, R, V = initialize(N, D)
    _, _ = black_scholes(S, X, T, R, V)
    total = timer.stop()
    print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        nargs="+",
        default=[10],
        dest="N",
        help="number of options to price in thousands",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        dest="P",
        help="precision of the computation in bits",
    )

    args, np, timer = parse_args(parser)

    dtype = None

    if args.P == 16:
        dtype = np.float16
    elif args.P == 32:
        dtype = np.float32
    elif args.P == 64:
        dtype = np.float64
    else:
        raise TypeError("Precision must be one of 16, 32, or 64")

    run_benchmark(
        run_black_scholes,
        args.benchmark,
        "Black Scholes",
        [("problem size", args.N), ("precision", dtype)],
        ["time (milliseconds)"],
    )
