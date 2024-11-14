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

from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate, convert_to_cupynumeric_ndarray
from .._utils import is_np2
from ..runtime import runtime
from .creation_shape import full

if is_np2:
    from numpy.lib.array_utils import normalize_axis_index  # type: ignore
else:
    from numpy.core.multiarray import normalize_axis_index  # type: ignore

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShape

if is_np2:
    from numpy.exceptions import AxisError  # type: ignore
else:
    from numpy import AxisError  # type: ignore

_builtin_max = max


@add_boilerplate("A")
def tile(
    A: ndarray, reps: int | Sequence[int] | npt.NDArray[np.int_]
) -> ndarray:
    """
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of ``max(d,
    A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Parameters
    ----------
    A : array_like
        The input array.
    reps : 1d array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    See Also
    --------
    numpy.tile

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    computed_reps: tuple[int, ...]
    if isinstance(reps, int):
        computed_reps = (reps,)
    else:
        if np.ndim(reps) > 1:
            raise TypeError("`reps` must be a 1d sequence")
        computed_reps = tuple(reps)
    # Figure out the shape of the destination array
    out_dims = _builtin_max(A.ndim, len(computed_reps))
    # Prepend ones until the dimensions match
    while len(computed_reps) < out_dims:
        computed_reps = (1,) + computed_reps
    out_shape: NdShape = ()
    # Prepend dimensions if necessary
    for dim in range(out_dims - A.ndim):
        out_shape += (computed_reps[dim],)
    offset = len(out_shape)
    for dim in range(A.ndim):
        out_shape += (A.shape[dim] * computed_reps[offset + dim],)
    assert len(out_shape) == out_dims
    result = ndarray(out_shape, dtype=A.dtype, inputs=(A,))
    result._thunk.tile(A._thunk, computed_reps)
    return result


def repeat(a: ndarray, repeats: Any, axis: int | None = None) -> ndarray:
    """
    Repeat elements of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    repeats : int or ndarray[int]
        The number of repetitions for each element. repeats is
        broadcasted to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values. By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as a, except along the
        given axis.

    Notes
    -----
    Currently, repeat operations supports only 1D arrays

    See Also
    --------
    numpy.repeat

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if repeats is None:
        raise TypeError(
            "int() argument must be a string, a bytes-like object or a number,"
            " not 'NoneType'"
        )

    if np.ndim(repeats) > 1:
        raise ValueError("`repeats` should be scalar or 1D array")

    # axes should be integer type
    if axis is not None and not isinstance(axis, int):
        raise TypeError("Axis should be of integer type")

    # when array is a scalar
    if np.ndim(a) == 0:
        if axis is not None and axis != 0 and axis != -1:
            raise AxisError(
                f"axis {axis} is out of bounds for array of dimension 0"
            )
        if np.ndim(repeats) == 0:
            if not isinstance(repeats, int):
                runtime.warn(
                    "converting repeats to an integer type",
                    category=UserWarning,
                )
            repeats = np.int64(repeats)
            return full((repeats,), cast(int | float, a))
        elif np.ndim(repeats) == 1 and len(repeats) == 1:
            if not isinstance(repeats, int):
                runtime.warn(
                    "converting repeats to an integer type",
                    category=UserWarning,
                )
            repeats = np.int64(repeats)
            return full((repeats[0],), cast(int | float, a))
        else:
            raise ValueError(
                "`repeat` with a scalar parameter `a` is only "
                "implemented for scalar values of the parameter `repeats`."
            )

    # array is an array
    array = convert_to_cupynumeric_ndarray(a)
    if np.ndim(repeats) == 1:
        repeats = convert_to_cupynumeric_ndarray(repeats)

    # if no axes specified, flatten array
    if axis is None:
        array = array.ravel()
        axis = 0

    axis_int: int = normalize_axis_index(axis, array.ndim)

    # If repeats is on a zero sized axis_int, then return the array.
    if array.shape[axis_int] == 0:
        return array.copy()

    if np.ndim(repeats) == 1:
        if repeats.shape[0] == 1 and repeats.shape[0] != array.shape[axis_int]:
            repeats = repeats[0]

    # repeats is a scalar.
    if np.ndim(repeats) == 0:
        # repeats is 0
        if repeats == 0:
            empty_shape = list(array.shape)
            empty_shape[axis_int] = 0
            return ndarray(shape=tuple(empty_shape), dtype=array.dtype)
        # repeats should be integer type
        if not isinstance(repeats, int):
            runtime.warn(
                "converting repeats to an integer type",
                category=UserWarning,
            )
        result = array._thunk.repeat(
            repeats=np.int64(repeats),
            axis=axis_int,
            scalar_repeats=True,
        )
    # repeats is an array
    else:
        # repeats should be integer type
        repeats = repeats._warn_and_convert(np.int64)
        if repeats.shape[0] != array.shape[axis_int]:
            raise ValueError("incorrect shape of repeats array")
        result = array._thunk.repeat(
            repeats=repeats._thunk, axis=axis_int, scalar_repeats=False
        )
    return ndarray(shape=result.shape, thunk=result)
