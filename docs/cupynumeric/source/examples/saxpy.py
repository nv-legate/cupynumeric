#!/usr/bin/env python
# Copyright 2024 Nihaal Chowdary
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

import cupy
import numpy
import argparse
import cupynumeric as cpn
import legate.core as lg
from legate.core import align, VariantCode, TaskContext
from legate.core.task import InputArray, OutputArray, task
from legate.timing import time


@task(variants=(VariantCode.CPU, VariantCode.GPU),
    constraints=(align("x", "y"), align("y", "z")),
)
def saxpy_task(ctx: TaskContext, x: InputArray, y: InputArray, z: OutputArray, a: float) -> None:
    xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
    x_local = xp.asarray(x)
    y_local = xp.asarray(y)
    z_local = xp.asarray(z)
    z_local[:] = a * x_local + y_local


def main():
    parser = argparse.ArgumentParser(description="Run SAXPY operation.")
    parser.add_argument("--size", type=int, default=1000, help="Size of input arrays")
    args = parser.parse_args()
    size = args.size

    x_global = cpn.arange(size, dtype=cpn.float32)
    y_global = cpn.ones(size, dtype=cpn.float32)
    z_global = cpn.zeros(size, dtype=cpn.float32)

    # Warm-up run
    saxpy_task(x_global, y_global, z_global, 2.0)
    
    rt = lg.get_legate_runtime()
    rt.issue_execution_fence()
    start = time()

    saxpy_task(x_global, y_global, z_global, 2.0)

    rt.issue_execution_fence()
    end = time()

    print(f"\nTime elapsed for saxpy: {(end - start) / 1000:.6f} milliseconds")


if __name__ == "__main__":
    main()
