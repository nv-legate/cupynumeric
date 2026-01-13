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
import re

from benchmark import parse_args, run_benchmark


def run_einsum(expr, N, iters, warmup, dtype, cupy_compatibility):
    # Parse contraction expression
    m = re.match(r"([a-zA-Z]*),([a-zA-Z]*)->([a-zA-Z]*)", expr)
    assert m is not None
    a_modes = list(m.group(1))
    b_modes = list(m.group(2))
    c_modes = list(m.group(3))

    # Sanity checks
    assert len(a_modes) == len(set(a_modes))
    assert len(b_modes) == len(set(b_modes))
    assert len(c_modes) == len(set(c_modes))
    assert all(m in a_modes or m in b_modes for m in c_modes)
    assert len(a_modes) == len(c_modes) or len(b_modes) == len(c_modes)
    swap_a_c = len(a_modes) == len(c_modes)

    # Count types of modes
    batch_modes = 0
    contracted_modes = 0
    other_modes = 0
    for m in a_modes:
        if m in b_modes:
            if m in c_modes:
                batch_modes += 1
            else:
                contracted_modes += 1
        else:
            assert m in c_modes
            other_modes += 1
    for m in b_modes:
        if m in a_modes:
            continue
        else:
            assert m in c_modes
            other_modes += 1

    # Print benchmark info
    print(f"Expression: {expr}")
    print(
        f"Modes: {batch_modes} batch, {contracted_modes} contracted, "
        f"{other_modes} other"
    )
    print(
        f"Problem Size: {N}**{len(a_modes)} x {N}**{len(b_modes)} "
        f"-> {N}**{len(c_modes)}"
    )
    print(f"Total Iterations: {iters}")
    print(f"Datatype: {dtype}")
    flops = N ** (batch_modes + other_modes) * (2 * N**contracted_modes - 1)
    print(f"Total FLOPS: {flops / 1e9:.3f} GFLOPS/iter")
    space = (
        N ** len(a_modes) + N ** len(b_modes) + N ** len(c_modes)
    ) * np.dtype(dtype).itemsize
    print(f"Total Size: {space / 1e6} MB")

    # Initialize arrays
    A = np.ones((N,) * len(a_modes), dtype=dtype)
    B = np.ones((N,) * len(b_modes), dtype=dtype)
    C = np.zeros((N,) * len(c_modes), dtype=dtype)

    # Run contraction
    timer.start()
    for idx in range(iters + warmup):
        if idx == warmup:
            timer.start()
        if cupy_compatibility:
            C = np.einsum(expr, A, B)
        else:
            # We use out= to avoid the use of intermediates, thus reducing
            # memory pressure and allowing us to use larger arrays.
            np.einsum(expr, A, B, out=C)
        # We make the output tensor an input to the next iteration, to
        # create a true data dependence between successive iterations.
        # This means that at least one of the input tensors needs to
        # have the same shape as the output tensor. Therefore, we can
        # only use expressions where the result has the same number of
        # modes as one of the arguments.
        if swap_a_c:
            A, C = C, A
        else:
            B, C = C, B
    total = timer.stop()

    # Print statistics
    average = total / iters
    print(f"Elapsed Time: {total:.3f} ms")
    print(f"Average Iteration: {average:.3f} ms")
    print(f"FLOPS/s: {flops / (average * 1e6):.3f} GFLOPS/s")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--expr",
        default="kac,kbc->kab",
        help="contraction to run, in np.einsum notation",
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
        "-i",
        "--iters",
        type=int,
        default=10,
        dest="iters",
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
        "-t",
        "--dtype",
        default="f32",
        choices=["f16", "f32", "f64", "c64", "c128"],
        dest="dtype",
        help="dtype for array elements",
    )
    parser.add_argument(
        "--cupy-compatibility",
        action="store_true",
        dest="cupy_compatibility",
        help="""einsum to call (if true, C = np.einsum(expr, A, B);
             else, use einsum(expr, A, B, out=C)""",
    )

    args, np, timer = parse_args(parser)

    cupy_compatibility = args.cupy_compatibility or args.package == "cupy"
    if cupy_compatibility:
        print("Use C = np.einsum(expr, A, B) for cupy compatibility")

    dtypes = {
        "f16": np.float16,
        "f32": np.float32,
        "f64": np.float64,
        "c64": np.complex64,
        "c128": np.complex128,
    }
    run_benchmark(
        run_einsum,
        args.benchmark,
        "Einsum",
        [
            ("expression", args.expr),
            ("problem size", args.N),
            ("iterations", args.iters),
            ("warmup iterations", args.warmup),
            ("precision", dtypes[args.dtype]),
            ("cupy compatibility", cupy_compatibility),
        ],
        ["time (milliseconds)"],
    )
