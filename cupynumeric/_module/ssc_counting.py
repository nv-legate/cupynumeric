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

from .._array.util import add_boilerplate

if TYPE_CHECKING:
    from .._array.array import ndarray


@add_boilerplate("a")
def count_nonzero(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> int | ndarray:
    """

    Counts the number of non-zero values in the array ``a``.

    Parameters
    ----------
    a : array_like
        The array for which to count non-zeros.
    axis : int or tuple, optional
        Axis or tuple of axes along which to count non-zeros.
        Default is None, meaning that non-zeros will be counted
        along a flattened version of ``a``.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        Default is False.
    Returns
    -------
    count : int or ndarray[int]
        Number of non-zero values in the array along a given axis.
        Otherwise, the total number of non-zero values in the array
        is returned.

    See Also
    --------
    numpy.count_nonzero

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a._count_nonzero(axis, keepdims=keepdims)
