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


def check_quantiles(np, a, q, axis, str_m, q_out):
    eps = 1.0e-8
    arr = get_numpy(np, a)
    qs_arr = get_numpy(np, q)

    np_q_out = numpy.quantile(arr, qs_arr, axis=axis, method=str_m)

    print("Checking result...")
    if np.allclose(np_q_out, q_out, atol=eps):
        print("PASS!")
    else:
        print("FAIL!")
        print("NUMPY    : " + str(np_q_out))
        print(np.__name__ + ": " + str(q_out))
        assert False


@benchmark_info(name="Quantiles")
def run_quantiles(
    np,
    shape,
    axis,
    datatype,
    lower,
    upper,
    str_method,
    *,
    timer,
    perform_check=False,
    print_timing=False,
):
    np.random.seed(1729)
    newtype = numpy.dtype(datatype).type

    N = 1
    for e in shape:
        N *= e
    shape = tuple(shape)
    if numpy.issubdtype(newtype, numpy.integer):
        if lower is None:
            lower = 0
        if upper is None:
            upper = numpy.iinfo(newtype).max
        a = np.random.randint(low=lower, high=upper, size=N).astype(newtype)
        a = a.reshape(shape)
    elif numpy.issubdtype(newtype, numpy.floating):
        a = np.random.random(shape).astype(newtype)
    else:
        print("UNKNOWN type " + str(newtype))
        assert False

    q = numpy.array([0.0, 0.37, 0.42, 0.5, 0.67, 0.83, 0.99, 1.0])

    timer.start()
    q_out = np.quantile(a, q, axis=axis, method=str_method)
    total = timer.stop()

    if perform_check:
        check_quantiles(np, a, q, axis, str_method, q_out)
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
        default=[(1000,)],
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

    args, harness = parse_with_harness(parser)

    harness.run_timed(
        run_quantiles,
        harness.np,
        args.shape,
        args.axis,
        args.datatype,
        args.lower,
        args.upper,
        args.method,
        timer=harness.timer,
        perform_check=args.check,
        print_timing=args.timing,
    )
