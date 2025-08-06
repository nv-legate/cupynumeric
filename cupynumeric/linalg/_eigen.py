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

from typing import TYPE_CHECKING, Optional

import legate.core.types as ty
from legate.core import dimension, get_legate_runtime

from cupynumeric.config import CuPyNumericOpCode

from ..runtime import runtime
from ._exception import LinAlgError

if TYPE_CHECKING:
    from .._thunk.deferred import DeferredArray


def prepare_manual_task_for_batched_matrices(
    full_shape: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Generates a pair of tilesize and color_shape to distribute a store
    containing batched matrices (..., M, N) along the batched dimensions
    without cutting through the matrices themselves.
    The routine aims for a partition of 'runtime.num_procs'.

    Parameters
    ----------
    full_shape : tuple[int, ...]
        dimension shape of batched matrices, (..., M, N)

    Returns
    -------
    tilesize : tuple[int, ...]
        shape used to tile the full shape
    color_shape : tuple[int, ...]
        shape to be used as launch shape to cover the full store

    """

    def choose_nd_color_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        # start with 1D and re-balance by powers of 2
        # (don't worry about other primes)
        color_shape = [1 for i in shape]
        if len(shape) > 2:
            color_shape[0] = runtime.num_procs

            done = False
            while not done and color_shape[0] % 2 == 0:
                # find max idx
                # if large enough --> switch
                weight_per_dim = list(
                    map(lambda x, y: x / y, list(shape), color_shape)
                )[:-2]

                max_weight = max(weight_per_dim)
                idx = weight_per_dim.index(max_weight)

                if weight_per_dim[idx] > 2 * weight_per_dim[0]:
                    color_shape[0] = color_shape[0] // 2
                    color_shape[idx] = color_shape[idx] * 2
                else:
                    done = True

        return tuple(color_shape)

    # coloring via num_procs to get utilization
    initial_color_shape = choose_nd_color_shape(full_shape)
    tilesize = tuple(
        map(lambda x, y: (x + y - 1) // y, full_shape, initial_color_shape)
    )
    color_shape = tuple(
        map(lambda x, y: (x + y - 1) // y, full_shape, tilesize)
    )

    return tilesize, color_shape


def eig_deferred(
    a: DeferredArray, ew: DeferredArray, ev: Optional[DeferredArray] = None
) -> None:
    library = a.library

    m = a.shape[-1]

    if m == 0:
        raise ValueError("Input shape dimension 0 not allowed!")

    tilesize, color_shape = prepare_manual_task_for_batched_matrices(a.shape)

    # partition defined py local batchsize
    tiled_a = a.base.partition_by_tiling(tilesize)
    tiled_ew = ew.base.partition_by_tiling(tilesize[:-1])

    task = get_legate_runtime().create_manual_task(
        library, CuPyNumericOpCode.GEEV, color_shape
    )
    task.throws_exception(LinAlgError)
    partition = tuple(dimension(i) for i in range(len(color_shape)))
    task.add_input(tiled_a, partition)
    task.add_output(tiled_ew, partition[:-1])
    if ev is not None:
        tiled_ev = ev.base.partition_by_tiling(tilesize)
        task.add_output(tiled_ev, partition)
    task.execute()


def eigh_deferred(
    a: DeferredArray,
    uplo_l: bool,
    ew: DeferredArray,
    ev: Optional[DeferredArray] = None,
) -> None:
    library = a.library

    m = a.shape[-1]

    if m == 0:
        raise ValueError("Input shape dimension 0 not allowed!")

    tilesize, color_shape = prepare_manual_task_for_batched_matrices(a.shape)

    # partition defined py local batchsize
    tiled_a = a.base.partition_by_tiling(tilesize)
    tiled_ew = ew.base.partition_by_tiling(tilesize[:-1])

    task = get_legate_runtime().create_manual_task(
        library, CuPyNumericOpCode.SYEV, color_shape
    )
    task.throws_exception(LinAlgError)
    partition = tuple(dimension(i) for i in range(len(color_shape)))
    task.add_input(tiled_a, partition)
    task.add_output(tiled_ew, partition[:-1])
    if ev is not None:
        tiled_ev = ev.base.partition_by_tiling(tilesize)
        task.add_output(tiled_ev, partition)
    task.add_scalar_arg(uplo_l, ty.bool_)
    task.execute()
