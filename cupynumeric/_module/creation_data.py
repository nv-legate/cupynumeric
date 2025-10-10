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

from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._array.array import _warn_and_convert, ndarray
from .._array.util import add_boilerplate, broadcast_where, check_writeable
from ..runtime import runtime
from ..types import CastingKind
from .creation_shape import empty_like

from ..config import TransferType

casting_kinds: tuple[CastingKind, ...] = (
    "no",
    "equiv",
    "safe",
    "same_kind",
    "unsafe",
)

if TYPE_CHECKING:
    from ..types import OrderType


def array(
    obj: Any,
    dtype: np.dtype[Any] | None = None,
    copy: bool = True,
    order: OrderType | Literal["K"] = "K",
    subok: bool = False,
    ndmin: int = 0,
) -> ndarray:
    """
    array(object, dtype=None, copy=True)

    Create an array.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy will
        only be made if __array__ returns a copy, if obj is a nested sequence,
        or if a copy is needed to satisfy any of the other requirements
        (`dtype`, `order`, etc.).
    order : ``{'K', 'A', 'C', 'F'}``, optional
        Specify the memory layout of the array. If object is not an array, the
        newly created array will be in C order (row major) unless 'F' is
        specified, in which case it will be in Fortran order (column major).
        If object is an array the following holds.

        ===== ========= ===================================================
        order  no copy                     copy=True
        ===== ========= ===================================================
        'K'   unchanged F & C order preserved, otherwise most similar order
        'A'   unchanged F order if input is F and not C, otherwise C order
        'C'   C order   C order
        'F'   F order   F order
        ===== ========= ===================================================

        When ``copy=False`` and a copy is made for other reasons, the result is
        the same as if ``copy=True``, with some exceptions for 'A', see the
        Notes section. The default order is 'K'.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    See Also
    --------
    numpy.array

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if not isinstance(obj, ndarray):
        thunk = runtime.get_numpy_thunk(obj, share=(not copy), dtype=dtype)
        result = ndarray._from_thunk(thunk)
    else:
        result = obj
    if dtype is not None and result.dtype != dtype:
        result = result.astype(dtype)
    elif copy and obj is result:
        result = result.copy()
    if result.ndim < ndmin:
        shape = (1,) * (ndmin - result.ndim) + result.shape
        result = result.reshape(shape)
    return result


def asarray(a: Any, dtype: np.dtype[Any] | None = None) -> ndarray:
    """
    Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.

    Notes
    ------
    The input array will be copied it is a view of another NumPy ndarray or
    if its datatype and the requested datatype are different.

    See Also
    --------
    numpy.asarray

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if isinstance(a, np.ndarray):
        if a.base is None and (dtype is None or a.dtype == dtype):
            # The array is not a view, so we attach to this buffer.
            thunk = runtime.find_or_create_array_thunk(
                a, transfer=TransferType.SHARE, read_only=not a.flags["W"]
            )
            return ndarray._from_thunk(thunk, writeable=a.flags["W"])
        else:
            # Considering all the non-trivial operations like advanced
            # indexing, transpose, squeeze etc., finding the mapping between
            # the parent and child array when operations are done in
            # the NumPy land will be challenging.
            # A copy of the parent array is required to prevent
            # data races that can occur when attaching multiple
            # region-fields to the same backing memory pointed to
            # by the parent and child NumPy arraysts. This aliasing
            # of the physical memory under two different region-fields
            # for the parent and child without defining the
            # relationship between the two can lead to data races
            # in Legion. A copy of the parent array prevents
            # this from happening.
            a_dtype = dtype if dtype is not None else a.dtype
            return array(a, a_dtype, copy=True)

    if not isinstance(a, ndarray):
        thunk = runtime.get_numpy_thunk(a, share=True, dtype=dtype)
        writeable = a.flags.writeable if isinstance(a, np.ndarray) else True
        arr = ndarray._from_thunk(thunk, writeable=writeable)
    else:
        arr = a

    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


@add_boilerplate("a")
def copy(a: ndarray) -> ndarray:
    """

    Return an array copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    See Also
    --------
    numpy.copy

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = empty_like(a, dtype=a.dtype)
    result._thunk.copy(a._thunk, deep=True)
    return result


@add_boilerplate("dst", "src")
def copyto(
    dst: ndarray,
    src: ndarray,
    casting: CastingKind = "same_kind",
    where: ndarray | None = None,
) -> None:
    """
    Copies values from one array to another, broadcasting as necessary.

    Raises a TypeError if the casting rule is violated, and if
    where is provided, it selects which elements to copy.

    Parameters
    ----------
    dst : ndarray
        The array into which values are copied.
    src : array_like
        The array from which values are copied.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when copying.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions
        of `dst`, and selects elements to copy from `src` to `dst`
        wherever it contains the value True.

    Notes
    -----
    This function modifies the destination array in-place. If you need
    to preserve the original array, make a copy first.

    See Also
    --------
    numpy.copyto

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    # Circular imports
    from .array_dimension import broadcast_to
    from .indexing import putmask

    check_writeable(dst)

    # Validate casting parameter
    if casting not in casting_kinds:
        raise ValueError(
            "casting must be one of 'no', 'equiv', "
            "'safe', 'same_kind', or 'unsafe'"
        )

    # Check casting compatibility
    if not np.can_cast(src.dtype, dst.dtype, casting=casting):
        raise TypeError(
            f"Cannot cast array data from dtype('{src.dtype}') to dtype("
            f"according to the rule '{casting}'"
        )

    # Broadcast src to match dst shape
    if src.shape != dst.shape:
        if not _is_broadcastable_to(src.shape, dst.shape):
            raise ValueError(
                f"could not broadcast input array from shape "
                f"{src.shape} into shape {dst.shape}"
            )
        else:
            src = broadcast_to(src, dst.shape)

    # Handle the where parameter
    if where is not None:
        where = broadcast_where(where, dst.shape)
    if where is not None and where.dtype != bool:
        raise TypeError(
            f"Cannot cast array data from dtype('{where.dtype}') to "
            f"dtype('bool') according to the rule '{casting}'"
        )

    # Convert src to dst dtype if needed
    if src.dtype != dst.dtype:
        src = _warn_and_convert(src, dtype=dst.dtype)

    # Perform the copy operation
    if where is None:
        # Copy all elements
        dst._thunk.copy(src._thunk, deep=True)
    else:
        putmask(dst, where, src)


def _is_broadcastable_to(
    src_shape: tuple[int, ...], dst_shape: tuple[int, ...]
) -> bool:
    for s_dim, d_dim in zip_longest(
        reversed(src_shape), reversed(dst_shape), fillvalue=1
    ):
        if s_dim != 1 and s_dim != d_dim:
            return False
    return True
