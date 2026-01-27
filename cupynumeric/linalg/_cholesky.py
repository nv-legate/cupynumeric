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

from typing import TYPE_CHECKING

from legate.core import (
    broadcast,
    constant,
    dimension,
    get_legate_runtime,
    types as ty,
)

from ..config import CuPyNumericOpCode
from ..runtime import runtime
from ..settings import settings
from ._eigen import prepare_manual_task_for_batched_matrices
from ._exception import LinAlgError
from ..types import TransposeMode

legate_runtime = get_legate_runtime()

if TYPE_CHECKING:
    from legate.core import Library, LogicalStore, LogicalStorePartition

    from .._thunk.deferred import DeferredArray
    from ..runtime import Runtime


def transpose_copy_single(
    library: Library, input: LogicalStore, output: LogicalStore
) -> None:
    task = legate_runtime.create_auto_task(
        library, CuPyNumericOpCode.TRANSPOSE_COPY_2D
    )
    p_out = task.add_output(output)
    p_in = task.add_input(input)
    # Output has the same shape as input, but is mapped
    # to a column major instance

    task.add_constraint(broadcast(p_out))
    task.add_constraint(broadcast(p_in))

    task.execute()


def transpose_copy(
    library: Library,
    launch_domain: tuple[int, ...],
    p_input: LogicalStorePartition,
    p_output: LogicalStorePartition,
) -> None:
    task = legate_runtime.create_manual_task(
        library, CuPyNumericOpCode.TRANSPOSE_COPY_2D, launch_domain
    )
    task.add_output(p_output)
    task.add_input(p_input)
    # Output has the same shape as input, but is mapped
    # to a column major instance

    task.execute()


def mp_potrf(
    library: Library,
    n: int,
    nb: int,
    input: LogicalStore,
    output: LogicalStore,
) -> None:
    task = legate_runtime.create_auto_task(library, CuPyNumericOpCode.MP_POTRF)
    task.throws_exception(LinAlgError)
    task.add_input(input)
    task.add_output(output)
    task.add_alignment(output, input)
    task.add_scalar_arg(n, ty.int64)
    task.add_scalar_arg(nb, ty.int64)
    task.add_nccl_communicator()  # for repartitioning
    task.execute()


def potrf_batched(
    library: Library,
    output: DeferredArray,
    input: DeferredArray,
    lower: bool,
    zeroout: bool,
) -> None:
    task = legate_runtime.create_auto_task(library, CuPyNumericOpCode.POTRF)
    task.add_output(output.base)
    task.add_input(input.base)
    task.add_scalar_arg(lower, ty.bool_)
    task.add_scalar_arg(zeroout, ty.bool_)
    ndim = input.base.ndim
    task.add_broadcast(input.base, (ndim - 2, ndim - 1))
    task.add_alignment(input.base, output.base)
    task.throws_exception(LinAlgError)
    task.execute()


def potrf(library: Library, p_output: LogicalStorePartition, i: int) -> None:
    task = legate_runtime.create_manual_task(
        library, CuPyNumericOpCode.POTRF, (i + 1, i + 1), lower_bounds=(i, i)
    )
    task.throws_exception(LinAlgError)
    task.add_output(p_output)
    task.add_input(p_output)
    task.add_scalar_arg(True, ty.bool_)  # lower triangular
    task.add_scalar_arg(False, ty.bool_)  # zero out upper triangle

    task.execute()


def potrs_batched(
    library: Library,
    x: LogicalStore,
    c: LogicalStore,
    c_shape: tuple[int, ...],
    lower: bool,
) -> None:
    nrhs = x.shape[-1]
    tilesize_c, color_shape = prepare_manual_task_for_batched_matrices(c_shape)
    tilesize_x = tuple(tilesize_c[:-1]) + (nrhs,)

    # partition defined py local batchsize
    tiled_c = c.partition_by_tiling(tilesize_c)
    tiled_x = x.partition_by_tiling(tilesize_x)

    task = get_legate_runtime().create_manual_task(
        library, CuPyNumericOpCode.POTRS, color_shape
    )
    task.throws_exception(LinAlgError)

    partition = tuple(dimension(i) for i in range(len(color_shape)))
    task.add_input(tiled_c, partition)
    task.add_input(tiled_x, partition)
    task.add_output(tiled_x, partition)

    task.add_scalar_arg(lower, ty.bool_)
    task.execute()


def trsm_batched(
    library: Library,
    x: LogicalStore,
    a: LogicalStore,
    b: LogicalStore,
    a_shape: tuple[int, ...],
    side: bool,
    lower: bool,
    transa: int,
    unit_diagonal: bool,
) -> None:
    nrhs = b.shape[-1]
    tilesize_a, color_shape = prepare_manual_task_for_batched_matrices(a_shape)
    tilesize_b = tuple(tilesize_a[:-1]) + (nrhs,)

    # partition defined py local batchsize
    tiled_a = a.partition_by_tiling(tilesize_a)
    tiled_b = b.partition_by_tiling(tilesize_b)
    tiled_x = x.partition_by_tiling(tilesize_b)

    task = get_legate_runtime().create_manual_task(
        library, CuPyNumericOpCode.TRSM, color_shape
    )
    task.throws_exception(LinAlgError)

    partition = tuple(dimension(i) for i in range(len(color_shape)))
    task.add_input(tiled_a, partition)
    task.add_input(tiled_b, partition)
    task.add_output(tiled_x, partition)

    task.add_scalar_arg(side, ty.bool_)
    task.add_scalar_arg(lower, ty.bool_)
    task.add_scalar_arg(transa, ty.int32)
    task.add_scalar_arg(unit_diagonal, ty.bool_)
    task.execute()


def trsm(
    library: Library, p_output: LogicalStorePartition, i: int, lo: int, hi: int
) -> None:
    if lo >= hi:
        return

    rhs = p_output.get_child_store(i, i)
    lhs = p_output

    task = legate_runtime.create_manual_task(
        library, CuPyNumericOpCode.TRSM, (hi, i + 1), lower_bounds=(lo, i)
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)

    task.add_scalar_arg(False, ty.bool_)  # side
    task.add_scalar_arg(True, ty.bool_)  # lower
    task.add_scalar_arg(2, ty.int32)  # transpose mode `C`
    task.add_scalar_arg(False, ty.bool_)  # unit diagonal

    task.execute()


def syrk(
    library: Library, p_output: LogicalStorePartition, k: int, i: int
) -> None:
    rhs = p_output.get_child_store(k, i)
    lhs = p_output

    task = legate_runtime.create_manual_task(
        library, CuPyNumericOpCode.SYRK, (k + 1, k + 1), lower_bounds=(k, k)
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)
    task.execute()


def gemm(
    library: Library,
    p_output: LogicalStorePartition,
    k: int,
    i: int,
    lo: int,
    hi: int,
) -> None:
    if lo >= hi:
        return

    rhs2 = p_output.get_child_store(k, i)
    lhs = p_output
    rhs1 = p_output

    task = legate_runtime.create_manual_task(
        library, CuPyNumericOpCode.GEMM, (hi, k + 1), lower_bounds=(lo, k)
    )
    task.add_output(lhs)
    task.add_input(rhs1, (dimension(0), constant(i)))
    task.add_input(rhs2)
    task.add_input(lhs)
    task.execute()


# TODO: We need a better cost model
def choose_color_shape(
    runtime: Runtime, shape: tuple[int, ...]
) -> tuple[int, ...]:
    extent = shape[0]

    # If there's only one processor or the matrix is too small,
    # don't even bother to partition it at all
    if runtime.num_procs == 1 or extent <= MIN_CHOLESKY_MATRIX_SIZE:
        return (1, 1)

    # If the matrix is big enough to warrant partitioning,
    # pick the granularity that the tile size is greater than a threshold
    num_tiles = runtime.num_procs
    max_num_tiles = runtime.num_procs * 4
    while (
        extent + num_tiles - 1
    ) // num_tiles > MIN_CHOLESKY_TILE_SIZE and num_tiles * 2 <= max_num_tiles:
        num_tiles *= 2

    return (num_tiles, num_tiles)


def tril_single(library: Library, output: LogicalStore) -> None:
    task = legate_runtime.create_auto_task(library, CuPyNumericOpCode.TRILU)
    task.add_output(output)
    task.add_input(output)
    task.add_scalar_arg(True, ty.bool_)
    task.add_scalar_arg(0, ty.int32)
    # Add a fake task argument to indicate that this is for Cholesky
    task.add_scalar_arg(True, ty.bool_)

    task.execute()


def tril(library: Library, p_output: LogicalStorePartition, n: int) -> None:
    task = legate_runtime.create_manual_task(
        library, CuPyNumericOpCode.TRILU, (n, n)
    )

    task.add_output(p_output)
    task.add_input(p_output)
    task.add_scalar_arg(True, ty.bool_)
    task.add_scalar_arg(0, ty.int32)
    # Add a fake task argument to indicate that this is for Cholesky
    task.add_scalar_arg(True, ty.bool_)

    task.execute()


def _rounding_divide(
    lhs: tuple[int, ...], rhs: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple((lh + rh - 1) // rh for (lh, rh) in zip(lhs, rhs))


def cho_solve_deferred(
    x: DeferredArray, c: DeferredArray, lower: bool
) -> None:
    library = runtime.library

    potrs_batched(library, x.base, c.base, c.shape, lower)


def solve_triangular_deferred(
    x: DeferredArray,
    a: DeferredArray,
    b: DeferredArray,
    trans: TransposeMode,
    lower: bool,
    unit_diagonal: bool,
) -> None:
    library = runtime.library
    transa: int

    # default to transpose for invalid input (mimic scipy)
    transa = 1

    match trans:
        case 0 | "N":
            transa = 0
        case 1 | "T":
            transa = 1
        case 2 | "C":
            transa = 2

    trsm_batched(
        library,
        x.base,
        a.base,
        b.base,
        a.shape,
        True,
        lower,
        transa,
        unit_diagonal,
    )


MIN_CHOLESKY_TILE_SIZE = 16 if settings.test() else 2048
MIN_CHOLESKY_MATRIX_SIZE = 32 if settings.test() else 8192


def cholesky_deferred(
    output: DeferredArray, input: DeferredArray, lower: bool, zeroout: bool
) -> None:
    library = runtime.library
    batched = len(input.base.shape) > 2
    if batched or not lower or output == input or runtime.num_procs == 1:
        size = input.base.shape[-1]
        if batched and size > 32768:
            # Choose 32768 as dimension cutoff for warning
            # so that for float64 anything larger than
            # 8 GiB produces a warning
            runtime.warn(
                "batched cholesky is only valid"
                " when the square submatrices fit"
                f" on a single proc, n > {size} may be too large",
                category=UserWarning,
            )
        potrf_batched(library, output, input, lower, zeroout)
        return

    # parallel implementation for individual matrices
    shape = tuple(output.base.shape)
    tile_shape: tuple[int, ...]
    if (
        runtime.has_cusolvermp
        and runtime.num_gpus > 1
        and shape[0] >= MIN_CHOLESKY_MATRIX_SIZE
    ):
        mp_potrf(
            library, shape[0], MIN_CHOLESKY_TILE_SIZE, input.base, output.base
        )
        if zeroout:
            tril_single(library, output.base)
    else:
        initial_color_shape = choose_color_shape(runtime, shape)
        tile_shape = _rounding_divide(shape, initial_color_shape)
        color_shape = _rounding_divide(shape, tile_shape)
        n = color_shape[0]

        p_input = input.base.partition_by_tiling(tile_shape)
        p_output = output.base.partition_by_tiling(tile_shape)
        transpose_copy(library, color_shape, p_input, p_output)

        for i in range(n):
            potrf(library, p_output, i)
            trsm(library, p_output, i, i + 1, n)
            for k in range(i + 1, n):
                syrk(library, p_output, k, i)
                gemm(library, p_output, k, i, k + 1, n)

        tril(library, p_output, n)
