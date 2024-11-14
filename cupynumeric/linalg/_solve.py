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
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import legate.core.types as ty
from legate.core import broadcast, get_legate_runtime

from ..config import CuPyNumericOpCode
from ..runtime import runtime
from ._cholesky import transpose_copy_single
from ._exception import LinAlgError

if TYPE_CHECKING:
    from legate.core import Library, LogicalStore

    from .._thunk.deferred import DeferredArray


def solve_single(library: Library, a: LogicalStore, b: LogicalStore) -> None:
    task = get_legate_runtime().create_auto_task(
        library, CuPyNumericOpCode.SOLVE
    )
    task.throws_exception(LinAlgError)
    p_a = task.add_input(a)
    p_b = task.add_input(b)
    task.add_output(a, p_a)
    task.add_output(b, p_b)

    task.add_constraint(broadcast(p_a))
    task.add_constraint(broadcast(p_b))

    task.execute()


MIN_SOLVE_TILE_SIZE = 512
MIN_SOLVE_MATRIX_SIZE = 2048


def mp_solve(
    library: Library,
    n: int,
    nrhs: int,
    nb: int,
    a: LogicalStore,
    b: LogicalStore,
    output: LogicalStore,
) -> None:
    task = get_legate_runtime().create_auto_task(
        library, CuPyNumericOpCode.MP_SOLVE
    )
    task.throws_exception(LinAlgError)
    task.add_input(a)
    task.add_input(b)
    task.add_output(output)
    task.add_alignment(output, b)
    task.add_scalar_arg(n, ty.int64)
    task.add_scalar_arg(nrhs, ty.int64)
    task.add_scalar_arg(nb, ty.int64)
    task.add_nccl_communicator()  # for repartitioning
    task.add_cal_communicator()
    task.execute()


def solve_deferred(
    output: DeferredArray, a: DeferredArray, b: DeferredArray
) -> None:
    from .._thunk.deferred import DeferredArray

    library = output.library

    if (
        runtime.has_cusolvermp
        and runtime.num_gpus > 1
        and a.base.shape[0] >= MIN_SOLVE_MATRIX_SIZE
    ):
        n = a.base.shape[0]
        nrhs = b.base.shape[1]
        mp_solve(
            library, n, nrhs, MIN_SOLVE_TILE_SIZE, a.base, b.base, output.base
        )
        return

    a_copy = cast(
        DeferredArray,
        runtime.create_empty_thunk(a.shape, dtype=a.base.type, inputs=(a,)),
    )
    transpose_copy_single(library, a.base, a_copy.base)

    if b.ndim > 1:
        transpose_copy_single(library, b.base, output.base)
    else:
        output.copy(b)

    solve_single(library, a_copy.base, output.base)
