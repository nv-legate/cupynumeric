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

import legate.core.types as ty
from legate.core import get_legate_runtime, dimension

from cupynumeric.config import CuPyNumericOpCode
from ..runtime import runtime
from ..settings import settings
from ._exception import LinAlgError

if TYPE_CHECKING:
    from legate.core import Library, LogicalStore

    from .._thunk.deferred import DeferredArray


def qr_single(
    library: Library, a: LogicalStore, q: LogicalStore, r: LogicalStore
) -> None:
    task = get_legate_runtime().create_auto_task(library, CuPyNumericOpCode.QR)
    task.throws_exception(LinAlgError)
    task.add_input(a)
    task.add_output(q)
    task.add_output(r)

    task.add_broadcast(a)
    task.add_broadcast(q)
    task.add_broadcast(r)

    task.execute()


QR_TILE_SIZE = 4 if settings.test() else 128
MIN_QR_MATRIX_SIZE = 1024 if settings.test() else 1048576


def mp_qr(
    library: Library,
    m: int,
    n: int,
    mb: int,
    nb: int,
    a: LogicalStore,
    q: LogicalStore,
    r: LogicalStore,
) -> None:
    initial_color_shape_x = runtime.num_gpus
    tilesize_x = (m + initial_color_shape_x - 1) // initial_color_shape_x
    color_shape_x = (m + tilesize_x - 1) // tilesize_x

    task = get_legate_runtime().create_manual_task(
        library, CuPyNumericOpCode.MP_QR, (color_shape_x, 1)
    )
    task.throws_exception(LinAlgError)

    tiled_a = a.partition_by_tiling((tilesize_x, n))
    tiled_q = q.partition_by_tiling((tilesize_x, n), (color_shape_x, 1))
    tiled_r = r.partition_by_tiling((tilesize_x, n), (color_shape_x, 1))

    task.add_input(tiled_a, (dimension(0), dimension(1)))
    task.add_output(tiled_q, (dimension(0), dimension(1)))
    task.add_output(tiled_r, (dimension(0), dimension(1)))

    task.add_scalar_arg(m, ty.int64)
    task.add_scalar_arg(n, ty.int64)
    task.add_scalar_arg(mb, ty.int64)
    task.add_scalar_arg(nb, ty.int64)

    task.add_nccl_communicator()  # for repartitioning

    task.execute()


def qr_deferred(a: DeferredArray, q: DeferredArray, r: DeferredArray) -> None:
    library = a.library

    m = a.base.shape[0]
    n = a.base.shape[1]

    # qr API requires squared tiles
    mb = QR_TILE_SIZE
    nb = QR_TILE_SIZE

    if (
        runtime.has_cusolvermp
        and runtime.num_gpus > 1
        and m * n >= MIN_QR_MATRIX_SIZE
    ):
        mp_qr(library, m, n, mb, nb, a.base, q.base, r.base)
    else:
        qr_single(library, a.base, q.base, r.base)
