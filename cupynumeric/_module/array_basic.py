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
    from ..types import NdShape


@add_boilerplate("a")
def ndim(a: ndarray) -> int:
    """

    Return the number of dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.  If it is not already an ndarray, a conversion is
        attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.  Scalars are zero-dimensional.

    See Also
    --------
    ndarray.ndim : equivalent method
    shape : dimensions of array
    ndarray.shape : dimensions of array

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return 0 if a is None else a.ndim


@add_boilerplate("a")
def shape(a: ndarray) -> NdShape:
    """

    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple[int, ...]
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    numpy.shape

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.shape
