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


def check_sorted(a, a_sorted, package, axis=-1):
    if package == "cupy":
        a_np = a.get()
    else:
        a_np = a.__array__()
    a_np_sorted = np.sort(a_np, axis)
    print("Checking result...")
    if num.allclose(a_np_sorted, a_sorted):
        print("PASS!")
    else:
        print("FAIL!")
        print("NUMPY    : " + str(a_np_sorted))
        print(package + ": " + str(a_sorted))
        assert False


def run_sort(
    shape,
    axis,
    argsort,
    datatype,
    lower,
    upper,
    *,
    perform_check=False,
    print_timing=False,
    package=None,
):
    num.random.seed(42)
    newtype = np.dtype(datatype).type

    N = 1
    for e in shape:
        N *= e
    shape = tuple(shape)
    if np.issubdtype(newtype, np.integer):
        if lower is None:
            lower = np.iinfo(newtype).min
        if upper is None:
            upper = np.iinfo(newtype).max
        a = num.random.randint(low=lower, high=upper, size=N).astype(newtype)
        a = a.reshape(shape)
    elif np.issubdtype(newtype, np.floating):
        a = num.random.random(shape).astype(newtype)
    elif np.issubdtype(newtype, np.complexfloating):
        a = num.array(
            num.random.random(shape) + num.random.random(shape) * 1j
        ).astype(newtype)
    else:
        print("UNKNOWN type " + str(newtype))
        assert False

    timer.start()
    if argsort:
        a_sorted = num.argsort(a, axis)
    else:
        a_sorted = num.sort(a, axis)
    total = timer.stop()

    if perform_check and not argsort:
        check_sorted(a, a_sorted, package, axis)
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
        default=(1000000,),
        dest="shape",
        help="array reshape tuple (default '(1000000,)')",
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
        default=-1,
        dest="axis",
        help="sort axis (default -1)",
    )
    parser.add_argument(
        "-g", "--arg", dest="argsort", action="store_true", help="use argsort"
    )

    args, num, timer = parse_args(parser)

    run_benchmark(
        run_sort,
        args.benchmark,
        "Sort",
        [
            ("shape", args.shape),
            ("axis", args.axis),
            ("argsort", args.argsort),
            ("datatype", args.datatype),
            ("lower", args.lower),
            ("upper", args.upper),
        ],
        ["time (milliseconds)"],
        perform_check=args.check,
        print_timing=args.timing,
        package=args.package,
    )
