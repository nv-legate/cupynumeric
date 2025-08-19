#!/usr/bin/env python
# Copyright 2024 NVIDIA Corporation
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
import cupy
import numpy
import argparse
import cupynumeric as cpn
import legate.core as lg
from legate.core import VariantCode, align, TaskContext
from legate.core.task import task, InputArray, ReductionArray, ADD
from legate.timing import time


# [matmul-start]
@task(
    variants=(VariantCode.CPU, VariantCode.GPU),
    constraints=(align("C", "A"), align("C", "B")),
)
def matmul_task(
    ctx: TaskContext, C: ReductionArray[ADD], A: InputArray, B: InputArray
) -> None:
    xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
    C = xp.asarray(C)[:, 0, :]
    A = xp.asarray(A)[:, :, 0]
    B = xp.asarray(B)[0, :, :]
    C += xp.matmul(A, B)


# [matmul-end]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Matrix multiplication operation"
    )
    parser.add_argument(
        "-m", type=int, default=50, help="Number of rows in matrix A and C"
    )
    parser.add_argument(
        "-k", type=int, default=75, help="Number of columns in A / rows in B"
    )
    parser.add_argument(
        "-n", type=int, default=100, help="Number of columns in matrix B and C"
    )
    args = parser.parse_args()

    # [input-section]
    m, k, n = args.m, args.k, args.n

    A_cpn = cpn.random.randint(1, 101, size=(m, k))
    B_cpn = cpn.random.randint(1, 101, size=(k, n))
    C_cpn = cpn.zeros((m, n))

    # Broadcasting to match Legate alignment requirements
    A_cpn = cpn.broadcast_to(
        A_cpn[:, :, cpn.newaxis], (m, k, n)
    )  # (m,k,1) → (m,k,n)
    B_cpn = cpn.broadcast_to(B_cpn[cpn.newaxis, :, :], (m, k, n))
    C_cpn = cpn.broadcast_to(C_cpn[:, cpn.newaxis, :], (m, k, n))

    matmul_task(C_cpn, A_cpn, B_cpn)
    # [function-call]

    # Benchmark run
    rt = lg.get_legate_runtime()
    rt.issue_execution_fence()
    start = time()

    matmul_task(C_cpn, A_cpn, B_cpn)

    rt.issue_execution_fence()
    end = time()

    print(f"\nTime elapsed for matmul: {(end - start) / 1000:.6f} seconds")


if __name__ == "__main__":
    main()
# [code-end]
