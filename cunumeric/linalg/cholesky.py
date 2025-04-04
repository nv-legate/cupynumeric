# Copyright 2023 NVIDIA Corporation
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

from legate.core import Rect, types as ty
from legate.core.shape import Shape
from legate.settings import settings

from cunumeric.config import CuNumericOpCode

from .exception import LinAlgError

if TYPE_CHECKING:
    from legate.core.context import Context
    from legate.core.store import Store, StorePartition

    from ..deferred import DeferredArray
    from ..runtime import Runtime


def get_gpu_lower_triangular(point):
    assert len(point) == 2
    from legate.core import get_machine
    from legate.core.machine import ProcessorKind

    machine = get_machine()
    num_gpus = machine.count(ProcessorKind.GPU)
    gpus = machine.only(ProcessorKind.GPU)

    # The linearized block-cyclic lower-triangular-blocked decomposition
    # mapping of 2-d points to blocks is:
    def mapping(i, j):
        return (j + (i * (i + 1)) // 2) % num_gpus

    return gpus[mapping(point[0], point[1])]


def transpose_copy_single(
    context: Context, input: Store, output: Store
) -> None:
    task = context.create_auto_task(CuNumericOpCode.TRANSPOSE_COPY_2D)
    task.add_output(output)
    task.add_input(input)
    # Output has the same shape as input, but is mapped
    # to a column major instance

    task.add_broadcast(output)
    task.add_broadcast(input)

    task.execute()


def transpose_copy(
    context: Context,
    launch_domain: Rect,
    p_input: StorePartition,
    p_output: StorePartition,
) -> None:
    task = context.create_manual_task(
        CuNumericOpCode.TRANSPOSE_COPY_2D,
        launch_domain=launch_domain,
    )
    task.add_output(p_output)
    task.add_input(p_input)
    # Output has the same shape as input, but is mapped
    # to a column major instance

    task.execute()


def potrf_single(context: Context, output: Store) -> None:
    task = context.create_auto_task(CuNumericOpCode.POTRF)
    task.throws_exception(LinAlgError)
    task.add_output(output)
    task.add_input(output)
    task.execute()


def potrf(context: Context, p_output: StorePartition, i: int) -> None:
    launch_domain = Rect(lo=(i, i), hi=(i + 1, i + 1))
    task = context.create_manual_task(
        CuNumericOpCode.POTRF, launch_domain=launch_domain
    )
    task.throws_exception(LinAlgError)
    task.add_output(p_output)
    task.add_input(p_output)
    with get_gpu_lower_triangular((i, i)):
        task.execute()


def trsm(
    context: Context, p_output: StorePartition, i: int, lo: int, hi: int
) -> None:
    if lo >= hi:
        return

    rhs = p_output.get_child_store(i, i)
    lhs = p_output

    for point in Rect(lo=(lo, i), hi=(hi, i + 1)):
        task = context.create_manual_task(
            CuNumericOpCode.TRSM,
            launch_domain=Rect(lo=point, hi=point, exclusive=False),
        )
        task.add_output(lhs)
        task.add_input(rhs)
        task.add_input(lhs)
        with get_gpu_lower_triangular(point):
            task.execute()


def syrk(context: Context, p_output: StorePartition, k: int, i: int) -> None:
    rhs = p_output.get_child_store(k, i)
    lhs = p_output

    launch_domain = Rect(lo=(k, k), hi=(k + 1, k + 1))
    task = context.create_manual_task(
        CuNumericOpCode.SYRK, launch_domain=launch_domain
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)
    with get_gpu_lower_triangular((k, k)):
        task.execute()


def gemm(
    context: Context,
    p_output: StorePartition,
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

    for point in Rect(lo=(lo, k), hi=(hi, k + 1)):
        task = context.create_manual_task(
            CuNumericOpCode.GEMM,
            launch_domain=Rect(lo=point, hi=point, exclusive=False),
        )
        task.add_output(lhs)
        task.add_input(rhs1, proj=lambda p: (p[0], i))
        task.add_input(rhs2)
        task.add_input(lhs)
        with get_gpu_lower_triangular(point):
            task.execute()


MIN_CHOLESKY_TILE_SIZE = 4096
MIN_CHOLESKY_MATRIX_SIZE = 8192


# TODO: We need a better cost model
def choose_color_shape(runtime: Runtime, shape: Shape) -> Shape:
    if settings.test():
        num_tiles = runtime.num_procs * 2
        return Shape((num_tiles, num_tiles))

    extent = shape[0]
    # If there's only one processor or the matrix is too small,
    # don't even bother to partition it at all
    if runtime.num_procs == 1 or extent <= MIN_CHOLESKY_MATRIX_SIZE:
        return Shape((1, 1))

    # If the matrix is big enough to warrant partitioning,
    # pick the granularity that the tile size is greater than a threshold
    num_tiles = runtime.num_procs
    max_num_tiles = runtime.num_procs * 4
    while (
        (extent + num_tiles - 1) // num_tiles > MIN_CHOLESKY_TILE_SIZE
        and num_tiles * 2 <= max_num_tiles
    ):
        num_tiles *= 2

    return Shape((num_tiles, num_tiles))


def tril_single(context: Context, output: Store) -> None:
    task = context.create_auto_task(CuNumericOpCode.TRILU)
    task.add_output(output)
    task.add_input(output)
    task.add_scalar_arg(True, ty.bool_)
    task.add_scalar_arg(0, ty.int32)
    # Add a fake task argument to indicate that this is for Cholesky
    task.add_scalar_arg(True, ty.bool_)

    task.execute()


def tril(context: Context, p_output: StorePartition, n: int) -> None:
    launch_domain = Rect((n, n))
    task = context.create_manual_task(
        CuNumericOpCode.TRILU, launch_domain=launch_domain
    )

    task.add_output(p_output)
    task.add_input(p_output)
    task.add_scalar_arg(True, ty.bool_)
    task.add_scalar_arg(0, ty.int32)
    # Add a fake task argument to indicate that this is for Cholesky
    task.add_scalar_arg(True, ty.bool_)

    task.execute()


def _batched_cholesky(output: DeferredArray, input: DeferredArray) -> None:
    # the only feasible implementation for right now is that
    # each cholesky submatrix fits on a single proc. We will have
    # wildly varying memory available depending on the system.
    # Just use a fixed cutoff to provide some sensible warning.
    # TODO: find a better way to inform the user dims are too big
    context: Context = output.context  # type: ignore
    task = context.create_auto_task(CuNumericOpCode.BATCHED_CHOLESKY)
    task.add_input(input.base)
    task.add_output(output.base)
    ndim = input.base.ndim
    task.add_broadcast(input.base, (ndim - 2, ndim - 1))
    task.add_broadcast(output.base, (ndim - 2, ndim - 1))
    task.add_alignment(input.base, output.base)
    task.throws_exception(LinAlgError)
    task.execute()


def cholesky(
    output: DeferredArray, input: DeferredArray, no_tril: bool
) -> None:
    runtime = output.runtime
    context: Context = output.context
    if len(input.base.shape) > 2:
        if no_tril:
            raise NotImplementedError(
                "batched cholesky expects to only "
                "produce the lower triangular matrix"
            )
        size = input.base.shape[-1]
        # Choose 32768 as dimension cutoff for warning
        # so that for float64 anything larger than
        # 8 GiB produces a warning
        if size > 32768:
            runtime.warn(
                "batched cholesky is only valid"
                " when the square submatrices fit"
                f" on a single proc, n > {size} may be too large",
                category=UserWarning,
            )
        return _batched_cholesky(output, input)

    if runtime.num_procs == 1:
        transpose_copy_single(context, input.base, output.base)
        potrf_single(context, output.base)
        if not no_tril:
            tril_single(context, output.base)
        return

    shape = output.base.shape
    initial_color_shape = choose_color_shape(runtime, shape)
    tile_shape = (shape + initial_color_shape - 1) // initial_color_shape
    color_shape = (shape + tile_shape - 1) // tile_shape
    n = color_shape[0]

    p_input = input.base.partition_by_tiling(tile_shape)
    p_output = output.base.partition_by_tiling(tile_shape)
    transpose_copy(context, Rect(hi=color_shape), p_input, p_output)

    for i in range(n):
        potrf(context, p_output, i)
        trsm(context, p_output, i, i + 1, n)
        for k in range(i + 1, n):
            syrk(context, p_output, k, i)
            gemm(context, p_output, k, i, k + 1, n)

    if no_tril:
        return

    tril(context, p_output, n)
