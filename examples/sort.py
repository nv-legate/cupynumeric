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

import numpy
from _benchmark import benchmark_info, get_numpy, parse_with_harness


def check_sorted(np, a, a_sorted, axis=-1):
    a_numpy = get_numpy(a)
    a_numpy_sorted = numpy.sort(a_numpy, axis)
    print("Checking result...")
    if np.allclose(a_numpy_sorted, a_sorted):
        print("PASS!")
    else:
        print("FAIL!")
        print("NUMPY    : " + str(a_numpy_sorted))
        print(np.__name__ + ": " + str(a_sorted))
        assert False


@benchmark_info(name="Sort")
def run_sort(
    np,
    shape,
    axis,
    argsort,
    datatype,
    lower,
    upper,
    *,
    timer,
    perform_check=False,
    print_timing=False,
):
    np.random.seed(42)
    newtype = np.dtype(datatype).type

    N = 1
    for e in shape:
        N *= e
    shape = tuple(shape)
    if numpy.issubdtype(newtype, np.integer):
        if lower is None:
            lower = np.iinfo(newtype).min
        if upper is None:
            upper = np.iinfo(newtype).max
        a = np.random.randint(low=lower, high=upper, size=N).astype(newtype)
        a = a.reshape(shape)
    elif numpy.issubdtype(newtype, np.floating):
        a = np.random.random(shape).astype(newtype)
    elif numpy.issubdtype(newtype, np.complexfloating):
        a = np.array(
            np.random.random(shape) + np.random.random(shape) * 1j
        ).astype(newtype)
    else:
        print("UNKNOWN type " + str(newtype))
        assert False

    timer.start()
    if argsort:
        a_sorted = np.argsort(a, axis)
    else:
        a_sorted = np.sort(a, axis)
    total = timer.stop()

    if perform_check and not argsort:
        check_sorted(a, a_sorted, axis)
    else:
        # do we need to synchronize?
        assert True
    if print_timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":

    def tuple_of_ints(arg):
        return tuple(map(int, arg.strip("()").split(",")))

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
        nargs="+",
        default=[(1000000,)],
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

    args, harness = parse_with_harness(parser)

    harness.run_timed(
        run_sort,
        harness.np,
        args.shape,
        args.axis,
        args.argsort,
        args.datatype,
        args.lower,
        args.upper,
        timer=harness.timer,
        perform_check=args.check,
        print_timing=args.timing,
    )
