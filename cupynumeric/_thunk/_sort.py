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

from legate.core import get_legate_runtime, types as ty

from ..lib.array_utils import normalize_axis_index
from ..config import CuPyNumericOpCode
from ..runtime import runtime
from cupynumeric._utils import is_np2

if is_np2:
    from numpy.lib.array_utils import normalize_axis_index
else:
    from numpy.core.multiarray import normalize_axis_index  # type: ignore

if TYPE_CHECKING:
    from .._thunk.deferred import DeferredArray


def sort_flattened(
    output: DeferredArray, input: DeferredArray, argsort: bool, stable: bool
) -> None:
    flattened = input.reshape((input.size,), order="C")

    # run sort flattened -- return 1D solution
    sort_result = runtime.create_deferred_thunk(
        flattened.shape, dtype=output.base.type
    )
    sort_deferred(sort_result, flattened, argsort, stable=stable)
    output.base = sort_result.base


def sort_swapped(
    output: DeferredArray,
    input: DeferredArray,
    argsort: bool,
    sort_axis: int,
    stable: bool,
) -> None:
    sort_axis = normalize_axis_index(sort_axis, input.ndim)

    # swap axes
    swapped = input.swapaxes(sort_axis, input.ndim - 1)

    swapped_copy = runtime.create_deferred_thunk(
        swapped.shape, dtype=input.base.type
    )
    swapped_copy.copy(swapped, deep=True)

    # run sort on last axis
    if argsort is True:
        sort_result = runtime.create_deferred_thunk(
            swapped_copy.shape, dtype=output.base.type
        )
        sort_deferred(sort_result, swapped_copy, argsort, stable=stable)
        output.base = sort_result.swapaxes(input.ndim - 1, sort_axis).base
    else:
        sort_deferred(swapped_copy, swapped_copy, argsort, stable=stable)
        output.base = swapped_copy.swapaxes(input.ndim - 1, sort_axis).base


def sort_task(
    output: DeferredArray, input: DeferredArray, argsort: bool, stable: bool
) -> None:
    legate_runtime = get_legate_runtime()
    task = legate_runtime.create_auto_task(
        output.library, CuPyNumericOpCode.SORT
    )

    uses_unbound_output = runtime.num_procs > 1 and input.ndim == 1

    if runtime.num_gpus > 1:
        task.add_nccl_communicator()
    elif runtime.num_gpus == 0 and runtime.num_procs > 1:
        task.add_cpu_communicator()

    task.add_scalar_arg(argsort, ty.bool_)  # return indices flag
    task.add_scalar_arg(input.base.shape, (ty.int64,))
    task.add_scalar_arg(stable, ty.bool_)
    task.add_input(input.base)

    if uses_unbound_output:
        unbound = runtime.create_unbound_thunk(dtype=output.base.type, ndim=1)
        task.add_output(unbound.base)
        task.execute()
        output.base = unbound.base
    else:
        task.add_output(output.base)
        task.add_alignment(output.base, input.base)
        task.execute()


def sort_deferred(
    output: DeferredArray,
    input: DeferredArray,
    argsort: bool,
    axis: int | None = -1,
    stable: bool = False,
) -> None:
    if axis is None and input.ndim > 1:
        sort_flattened(output, input, argsort, stable)
    else:
        if axis is None:
            computed_axis = 0
        else:
            computed_axis = normalize_axis_index(axis, input.ndim)

        if computed_axis == input.ndim - 1:
            sort_task(output, input, argsort, stable)
        else:
            sort_swapped(output, input, argsort, computed_axis, stable)
