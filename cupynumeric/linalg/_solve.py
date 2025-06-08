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
from legate.settings import settings

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


MIN_SOLVE_TILE_SIZE = 16 if settings.test() else 512
MIN_SOLVE_MATRIX_SIZE = 32 if settings.test() else 2048


def mp_solve(
    library: Library,
    n: int,
    nrhs: int,
    nb: int,
    a: LogicalStore,
    b: LogicalStore,
    output: LogicalStore,
) -> None:
    # coloring via num_procs to get utilization
    initial_color_shape_x = runtime.num_gpus
    tilesize_x = (n + initial_color_shape_x - 1) // initial_color_shape_x
    color_shape_x = (n + tilesize_x - 1) // tilesize_x

    task = get_legate_runtime().create_manual_task(
        library, CuPyNumericOpCode.MP_SOLVE, (color_shape_x, 1)
    )
    task.throws_exception(LinAlgError)

    tiled_a = a.partition_by_tiling((tilesize_x, n))
    tiled_b = b.partition_by_tiling((tilesize_x, nrhs))
    tiled_output = output.partition_by_tiling((tilesize_x, nrhs))

    task.add_input(tiled_a)
    task.add_input(tiled_b)
    task.add_output(tiled_output)

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
