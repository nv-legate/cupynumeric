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

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate
from ..runtime import runtime
from .creation_shape import empty_like

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
        result = ndarray(shape=None, thunk=thunk)
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
        Array interpretation of `a`.  No copy is performed if the input is
        already an ndarray with matching dtype.  If `a` is a subclass of
        ndarray, a base class ndarray is returned.

    See Also
    --------
    numpy.asarray

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if not isinstance(a, ndarray):
        thunk = runtime.get_numpy_thunk(a, share=True, dtype=dtype)
        writeable = a.flags.writeable if isinstance(a, np.ndarray) else True
        array = ndarray(shape=None, thunk=thunk, writeable=writeable)
    else:
        array = a
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return array


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
