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

import numpy as np
from benchmark import parse_args, run_benchmark


def initialize(shape, dt, axis):
    if dt == "int":
        A = num.random.randint(1000, size=shape, dtype=np.int32)
        if axis is None:
            B = num.zeros(shape=A.size, dtype=np.int32)
        else:
            B = num.zeros(shape=shape, dtype=np.int32)
    elif dt == "float":
        A = num.random.random(shape).astype(np.float32)
        # insert NAN at second element
        A[(1,) * len(shape)] = np.nan

        if axis is None:
            B = num.zeros(shape=A.size, dtype=np.float32)
        else:
            B = num.zeros(shape=shape, dtype=np.float32)
    else:
        A = (
            num.random.random(shape).astype(np.float32)
            + num.random.random(shape).astype(np.float32) * 1j
        )
        A[(1,) * len(shape)] = np.nan

        if axis is None:
            B = num.zeros(shape=A.size, dtype=np.complex64)
        else:
            B = num.zeros(shape=shape, dtype=np.complex64)

    return A, B


def check_scan(OP, A, B, ax):
    C = np.zeros(shape=B.shape, dtype=B.dtype)
    getattr(np, OP)(A, out=C, axis=ax)

    print("Checking result...")
    if np.allclose(B, C, equal_nan=True):
        print("PASS!")
    else:
        print("FAIL!")
        print(f"INPUT    : {A}")
        print(f"CUPYNUMERIC: {B}")
        print(f"NUMPY    : {C}")
        assert False


def run_scan(OP, shape, dt, ax, check):
    print(f"Problem Size:    shape={shape}")

    print(f"Problem Type:    OP={OP}")
    print(f"Axis:            axis={ax}")
    print(f"Data type:       dtype={dt}32")
    A, B = initialize(shape=shape, dt=dt, axis=ax)
    timer.start()

    # op handling
    getattr(num, OP)(A, out=B, axis=ax)

    total = timer.stop()
    print(f"Elapsed Time:  {total}ms")
    # error checking
    if check:
        check_scan(OP, A, B, ax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--shape",
        type=int,
        nargs="+",
        default=(10),
        dest="shape",
        help="array shape (default '(10)')",
    )
    parser.add_argument(
        "-t",
        "--datatype",
        default="int",
        choices=["int", "float", "complex"],
        dest="dt",
        help="data type (default int)",
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=int,
        default=None,
        dest="axis",
        help="scan axis (default None)",
    )
    parser.add_argument(
        "-o",
        "--operation",
        default="cumsum",
        choices=["cumsum", "cumprod", "nancumsum", "nancumprod"],
        dest="OP",
        help="operation, can be either cumsum (default), cumprod, "
        "nancumsum, nancumprod",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="check the result of the solve",
    )

    args, num, timer = parse_args(parser)

    run_benchmark(
        run_scan,
        args.benchmark,
        "Scan",
        (args.OP, args.shape, args.dt, args.axis, args.check),
    )
