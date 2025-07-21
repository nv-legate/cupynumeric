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

# [code-start]

import argparse
import cupy
import numpy
import cupynumeric as cpn
import legate.core as lg
from legate.core import align, broadcast, VariantCode, TaskContext
from legate.core.task import InputStore, OutputStore, task
from legate.core.types import complex64
from legate.timing import time

# [fft-start]
@task(variants=(VariantCode.CPU, VariantCode.GPU),
    constraints=(align("dst", "src"), broadcast("src", (1, 2))),
)
def fft2d_batched_gpu(ctx: TaskContext, dst: OutputStore, src: InputStore):
    xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
    cp_src = xp.asarray(src)
    cp_dst = xp.asarray(dst)
    cp_dst[:] = xp.fft.fftn(cp_src, axes=(1, 2))  # Apply 2D FFT across axes 1 and 2

#[fft-end]

def main():
    parser = argparse.ArgumentParser(description="Run FFT operation")
    parser.add_argument(
        "--shape", type=str, default="128,256,256",
        help="Shape of the array in the format D1,D2,D3"
    )
    args = parser.parse_args()

    # [input-section]
    shape = tuple(map(int, args.shape.split(",")))

    A_cpn = cpn.zeros(shape, dtype=cpn.complex64)
    B_cpn = cpn.random.randint(1, 101, size=shape).astype(cpn.complex64)

    fft2d_batched_gpu(A_cpn, B_cpn)
    # [function-call]

    # benchmark run
    rt = lg.get_legate_runtime()
    rt.issue_execution_fence()
    start = time()

    fft2d_batched_gpu(A_cpn, B_cpn)

    rt.issue_execution_fence()
    end = time()

    print(f"\nTime elapsed for batched fft: {(end - start)/1000:.6f} milliseconds")


if __name__ == "__main__":
    main()

# [code-end]
