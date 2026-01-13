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
from benchmark import parse_args, run_benchmark


def check_equal_numpy(input, output, package):
    if package == "cupy":
        out2 = numpy.linalg.cholesky(input.get())
    else:
        out2 = numpy.linalg.cholesky(input.__array__())

    print("Checking result...")
    if numpy.allclose(output, out2):
        print("PASS!")
    else:
        print("FAIL!")
        print("numpy     : " + str(out2))
        print(f"{package} : " + str(output))
        assert False


def cholesky(
    n, dtype, *, perform_check=False, print_timing=False, package=None
):
    input = num.eye(n, dtype=dtype)

    timer.start()
    out1 = num.linalg.cholesky(input)
    total = timer.stop()

    if perform_check:
        check_equal_numpy(input, out1, package)

    if print_timing:
        print(f"Elapsed Time: {total} ms")
        flops = (n**3) / 3 + 2 * n / 3
        print(f"{flops / total / 1000000} GOP/s")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=10,
        dest="n",
        help="number of rows/cols in the matrix",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        default="float64",
        choices=["float32", "float64", "complex64", "complex128"],
        dest="dtype",
        help="data type",
    )
    parser.add_argument(
        "--check",
        dest="check",
        action="store_true",
        help="compare result to numpy",
    )
    args, num, timer = parse_args(parser)

    run_benchmark(
        cholesky,
        args.benchmark,
        "Cholesky",
        [("problem size", args.n), ("precision", args.dtype)],
        ["time (milliseconds)"],
        perform_check=args.check,
        print_timing=args.timing,
        package=args.package,
    )
