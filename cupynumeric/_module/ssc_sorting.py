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

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate
from .._utils import is_np2

if TYPE_CHECKING:
    from ..types import SelectKind, SortType


@add_boilerplate("a")
def argsort(
    a: ndarray,
    axis: int | None = -1,
    kind: SortType = "quicksort",
    order: str | list[str] | None = None,
) -> ndarray:
    """

    Returns the indices that would sort an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis to sort. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'quicksort', 'mergesort', 'heapsort', 'stable'}``, optional
        Default is 'quicksort'. The underlying sort algorithm might vary.
        The code basically supports 'stable' or *not* 'stable'.
    order : str or list[str], optional
        Currently not supported

    Returns
    -------
    index_array : ndarray[int]
        Array of indices that sort a along the specified axis. It has the
        same shape as `a.shape` or is flattened in case of `axis` is None.

    See Also
    --------
    numpy.argsort

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    result = ndarray(a.shape, np.int64)
    result._thunk.sort(
        rhs=a._thunk, argsort=True, axis=axis, kind=kind, order=order
    )
    return result


if not is_np2:

    def msort(a: ndarray) -> ndarray:
        """

        Returns a sorted copy of an array sorted along the first axis.

        Parameters
        ----------
        a : array_like
            Input array.

        Returns
        -------
        out : ndarray
            Sorted array with same dtype and shape as `a`.

        See Also
        --------
        numpy.msort

        Availability
        --------
        Multiple GPUs, Multiple CPUs
        """
        return sort(a, axis=0)


@add_boilerplate("a")
def sort(
    a: ndarray,
    axis: int | None = -1,
    kind: SortType = "quicksort",
    order: str | list[str] | None = None,
) -> ndarray:
    """

    Returns a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis to sort. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'quicksort', 'mergesort', 'heapsort', 'stable'}``, optional
        Default is 'quicksort'. The underlying sort algorithm might vary.
        The code basically supports 'stable' or *not* 'stable'.
    order : str or list[str], optional
        Currently not supported

    Returns
    -------
    out : ndarray
        Sorted array with same dtype and shape as `a`. In case `axis` is
        None the result is flattened.


    See Also
    --------
    numpy.sort

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, a.dtype)
    result._thunk.sort(rhs=a._thunk, axis=axis, kind=kind, order=order)
    return result


@add_boilerplate("a")
def sort_complex(a: ndarray) -> ndarray:
    """

    Returns a sorted copy of an array sorted along the last axis. Sorts the
    real part first, the imaginary part second.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    out : ndarray, complex
        Sorted array with same shape as `a`.

    See Also
    --------
    numpy.sort_complex

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    result = sort(a)
    # force complex result upon return
    if np.issubdtype(result.dtype, np.complexfloating):
        return result
    elif (
        np.issubdtype(result.dtype, np.integer) and result.dtype.itemsize <= 2
    ):
        return result.astype(np.complex64, copy=True)
    else:
        return result.astype(np.complex128, copy=True)


# partition


@add_boilerplate("a")
def argpartition(
    a: ndarray,
    kth: int | Sequence[int],
    axis: int | None = -1,
    kind: SelectKind = "introselect",
    order: str | list[str] | None = None,
) -> ndarray:
    """

    Perform an indirect partition along the given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    kth : int or Sequence[int]
    axis : int or None, optional
        Axis to partition. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'introselect'}``, optional
        Currently not supported.
    order : str or list[str], optional
        Currently not supported.

    Returns
    -------
    out : ndarray[int]
        Array of indices that partitions a along the specified axis. It has the
        same shape as `a.shape` or is flattened in case of `axis` is None.


    Notes
    -----
    The current implementation falls back to `cupynumeric.argsort`.

    See Also
    --------
    numpy.argpartition

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, np.int64)
    result._thunk.partition(
        rhs=a._thunk,
        argpartition=True,
        kth=kth,
        axis=axis,
        kind=kind,
        order=order,
    )
    return result


@add_boilerplate("a")
def partition(
    a: ndarray,
    kth: int | Sequence[int],
    axis: int | None = -1,
    kind: SelectKind = "introselect",
    order: str | list[str] | None = None,
) -> ndarray:
    """

    Returns a partitioned copy of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    kth : int or Sequence[int]
    axis : int or None, optional
        Axis to partition. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'introselect'}``, optional
        Currently not supported.
    order : str or list[str], optional
        Currently not supported.

    Returns
    -------
    out : ndarray
        Partitioned array with same dtype and shape as `a`. In case `axis` is
        None the result is flattened.

    Notes
    -----
    The current implementation falls back to `cupynumeric.sort`.

    See Also
    --------
    numpy.partition

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, a.dtype)
    result._thunk.partition(
        rhs=a._thunk, kth=kth, axis=axis, kind=kind, order=order
    )
    return result
