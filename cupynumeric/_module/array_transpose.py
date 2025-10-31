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

from typing import TYPE_CHECKING, Sequence

from .._array.util import add_boilerplate
from ..lib.array_utils import normalize_axis_tuple

if TYPE_CHECKING:
    from .._array.array import ndarray


@add_boilerplate("a")
def swapaxes(a: ndarray, axis1: int, axis2: int) -> ndarray:
    """

    Interchange two axes of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        If `a` is an ndarray, then a view of `a` is returned; otherwise a new
        array is created.

    See Also
    --------
    numpy.swapaxes

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.swapaxes(axis1, axis2)


@add_boilerplate("a")
def transpose(a: ndarray, axes: list[int] | None = None) -> ndarray:
    """

    Permute the dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : list[int], optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    Returns
    -------
    p : ndarray
        `a` with its axes permuted.  A view is returned whenever
        possible.

    See Also
    --------
    numpy.transpose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.transpose(axes=axes)


@add_boilerplate("a")
def moveaxis(
    a: ndarray, source: Sequence[int], destination: Sequence[int]
) -> ndarray:
    """
    Move axes of an array to new positions.
    Other axes remain in their original order.

    Parameters
    ----------
    a : ndarray
        The array whose axes should be reordered.
    source : int or Sequence[int]
        Original positions of the axes to move. These must be unique.
    destination : int or Sequence[int]
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    numpy.moveaxis

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    source = normalize_axis_tuple(source, a.ndim, "source")
    destination = normalize_axis_tuple(destination, a.ndim, "destination")
    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have the same number of elements"
        )
    order = [n for n in range(a.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return a.transpose(order)
