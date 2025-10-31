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

import itertools
from typing import TYPE_CHECKING, Sequence

from .._array.util import add_boilerplate
from ..lib.array_utils import normalize_axis_tuple
from .array_dimension import broadcast
from .array_transpose import transpose
from .creation_shape import empty_like

if TYPE_CHECKING:
    from .._array.array import ndarray
    from ..types import NdShapeLike


@add_boilerplate("m")
def flip(m: ndarray, axis: NdShapeLike | None = None) -> ndarray:
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : array_like
        Input array.
    axis : None or int or tuple[int], optional
         Axis or axes along which to flip over. The default, axis=None, will
         flip over all of the axes of the input array.  If axis is negative it
         counts from the last to the first axis.

         If axis is a tuple of ints, flipping is performed on all of the axes
         specified in the tuple.

    Returns
    -------
    out : array_like
        A new array that is constructed from `m` with the entries of axis
        reversed.

    See Also
    --------
    numpy.flip

    Availability
    --------
    Single GPU, Single CPU

    Notes
    -----
    cuPyNumeric implementation doesn't return a view, it returns a new array
    """
    return m.flip(axis=axis)


@add_boilerplate("m")
def flipud(m: ndarray) -> ndarray:
    """
    Reverse the order of elements along axis 0 (up/down).

    For a 2-D array, this flips the entries in each column in the up/down
    direction. Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array.

    Returns
    -------
    out : array_like
        A new array that is constructed from `m` with rows reversed.

    See Also
    --------
    numpy.flipud

    Availability
    --------
    Single GPU, Single CPU

    Notes
    -----
    cuPyNumeric implementation doesn't return a view, it returns a new array
    """
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return flip(m, axis=0)


@add_boilerplate("m")
def fliplr(m: ndarray) -> ndarray:
    """
    Reverse the order of elements along axis 1 (left/right).

    For a 2-D array, this flips the entries in each row in the left/right
    direction. Columns are preserved, but appear in a different order than
    before.

    Parameters
    ----------
    m : array_like
        Input array, must be at least 2-D.

    Returns
    -------
    f : ndarray
        A new array that is constructed from `m` with the columns reversed.

    See Also
    --------
    numpy.fliplr

    Availability
    --------
    Single GPU, Single CPU

    Notes
    -----
    cuPyNumeric implementation doesn't return a view, it returns a new array
    """
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return flip(m, axis=1)


@add_boilerplate("m")
def rot90(m: ndarray, k: int = 1, axes: Sequence[int] = (0, 1)) -> ndarray:
    """
    Rotate an array by 90 degrees in the plane specified by axes.

    Rotation direction is from the first towards the second axis.
    This means for a 2D array with the default `k` and `axes`, the
    rotation will be counterclockwise.

    Parameters
    ----------
    m : array_like
        Array of two or more dimensions.
    k : integer
        Number of times the array is rotated by 90 degrees.
    axes : (2,) array_like
        The array is rotated in the plane defined by the axes.
        Axes must be different.

    Returns
    -------
    y : ndarray
        A rotated copy of `m`.

    See Also
    --------
    flip : Reverse the order of elements in an array along the given axis.
    fliplr : Flip an array horizontally.
    flipud : Flip an array vertically.

        Availability
        --------
        Single GPU, Single CPU
    """
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == m.ndim:
        raise ValueError("Axes must be different.")

    if (
        axes[0] >= m.ndim
        or axes[0] < -m.ndim
        or axes[1] >= m.ndim
        or axes[1] < -m.ndim
    ):
        raise ValueError(
            f"Axes={axes} out of range for array of ndim={m.ndim}."
        )

    k %= 4

    if k == 0:
        return m.copy()
    if k == 2:
        return flip(flip(m, axes[0]), axes[1])

    axes_list = list(range(0, m.ndim))
    (axes_list[axes[0]], axes_list[axes[1]]) = (
        axes_list[axes[1]],
        axes_list[axes[0]],
    )

    if k == 1:
        return transpose(flip(m, axes[1]), axes_list)
    else:
        # k == 3
        return flip(transpose(m, axes_list), axes[1])


@add_boilerplate("a")
def roll(
    a: ndarray,
    shift: int | tuple[int, ...],
    axis: int | tuple[int, ...] | None = None,
) -> ndarray:
    """
    Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int or tuple of ints
        The number of places by which elements are shifted.  If a tuple,
        then `axis` must be a tuple of the same size, and each of the
        given axes is shifted by the corresponding number.  If an int
        while `axis` is a tuple of ints, then the same value is used for
        all given axes.
    axis : int or tuple of ints, optional
        Axis or axes along which elements are shifted.  By default, the
        array is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    Notes
    -----
    Supports rolling over multiple dimensions simultaneously.
    """
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)

    normalized_axis: tuple[int, ...] = normalize_axis_tuple(
        axis, a.ndim, allow_duplicate=True
    )
    broadcasted = broadcast(shift, normalized_axis)
    if broadcasted.ndim > 1:
        raise ValueError(
            "'shift' and 'axis' should be scalars or 1D sequences"
        )
    shifts = {ax: 0 for ax in range(a.ndim)}
    for sh, ax in broadcasted:
        shifts[ax] += sh

    rolls: list[tuple[tuple[slice, ...], ...]]
    rolls = [((slice(None), slice(None)),)] * a.ndim
    for ax, offset in shifts.items():
        offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
        if offset:
            # (original, result), (original, result)
            rolls[ax] = (
                (slice(None, -offset), slice(offset, None)),
                (slice(-offset, None), slice(None, offset)),
            )

    result = empty_like(a)
    for indices in itertools.product(*rolls):
        arr_index, res_index = zip(*indices)
        result[res_index] = a[arr_index]

    return result
