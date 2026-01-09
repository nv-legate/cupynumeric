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


def check_quantiles(package, a, q, axis, str_m, q_out):
    eps = 1.0e-8
    if package == "cupy":
        arr = a.get()
        qs_arr = q.get()
    else:
        arr = a.__array__()
        qs_arr = q.__array__()

    np_q_out = np.quantile(arr, qs_arr, axis=axis, method=str_m)

    print("Checking result...")
    if num.allclose(np_q_out, q_out, atol=eps):
        print("PASS!")
    else:
        print("FAIL!")
        print("NUMPY    : " + str(np_q_out))
        print(package + ": " + str(q_out))
        assert False


def run_quantiles(
    shape,
    axis,
    datatype,
    lower,
    upper,
    str_method,
    *,
    perform_check=False,
    print_timing=False,
    package=None,
):
    num.random.seed(1729)
    newtype = np.dtype(datatype).type

    N = 1
    for e in shape:
        N *= e
    shape = tuple(shape)
    if np.issubdtype(newtype, np.integer):
        if lower is None:
            lower = 0
        if upper is None:
            upper = np.iinfo(newtype).max
        a = num.random.randint(low=lower, high=upper, size=N).astype(newtype)
        a = a.reshape(shape)
    elif np.issubdtype(newtype, np.floating):
        a = num.random.random(shape).astype(newtype)
    else:
        print("UNKNOWN type " + str(newtype))
        assert False

    q = np.array([0.0, 0.37, 0.42, 0.5, 0.67, 0.83, 0.99, 1.0])

    timer.start()
    q_out = num.quantile(a, q, axis=axis, method=str_method)
    total = timer.stop()

    if perform_check:
        check_quantiles(package, a, q, axis, str_method, q_out)
    else:
        # do we need to synchronize?
        assert True
    if print_timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":

    def tuple_of_ints(arg):
        return tuple(map(int, arg.split(",")))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="check the result of the solve",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=tuple_of_ints,
        default=(1000,),
        dest="shape",
        help="array reshape tuple (default '(1000,)')",
    )
    parser.add_argument(
        "-d",
        "--datatype",
        type=str,
        default="uint32",
        dest="datatype",
        help="data type (default np.uint32)",
    )
    parser.add_argument(
        "-l",
        "--lower",
        type=int,
        default=None,
        dest="lower",
        help="lower bound for integer based arrays (inclusive)",
    )
    parser.add_argument(
        "-u",
        "--upper",
        type=int,
        default=None,
        dest="upper",
        help="upper bound for integer based arrays (exclusive)",
    )
    parser.add_argument(
        "-a",
        "--axis",
        type=int,
        default=None,
        dest="axis",
        help="sort axis (default None)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="linear",
        dest="method",
        help="quantile interpolation method",
    )

    args, num, timer = parse_args(parser)

    run_benchmark(
        run_quantiles,
        args.benchmark,
        "Quantiles",
        [
            ("shape", args.shape),
            ("axis", args.axis),
            ("precision", args.datatype),
            ("lower", args.lower),
            ("upper", args.upper),
            ("method", args.method),
        ],
        ["time (milliseconds)"],
        perform_check=args.check,
        print_timing=args.timing,
        package=args.package,
    )
