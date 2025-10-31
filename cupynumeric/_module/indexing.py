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

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .._array.array import _warn_and_convert, ndarray
from .._array.util import (
    add_boilerplate,
    check_writeable,
    convert_to_cupynumeric_ndarray,
)
from .._module.array_dimension import broadcast_arrays
from .._utils.array import calculate_volume
from .._utils.coverage import is_implemented
from ..lib.array_utils import normalize_axis_index
from ..runtime import runtime
from ..types import NdShape
from .array_joining import hstack
from .array_shape import reshape
from .array_tiling import tile
from .creation_data import asarray
from .creation_matrices import tri
from .creation_ranges import arange
from .creation_shape import empty, ones, zeros_like
from .ssc_counting import count_nonzero
from .ssc_searching import nonzero

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

    from ..types import BoundsMode, OrderType

_builtin_min = min


@add_boilerplate("arr", "mask", "vals")
def place(arr: ndarray, mask: ndarray, vals: ndarray) -> None:
    """
    Change elements of an array based on conditional and input values.

    Parameters
    ----------
    arr : array_like
        Array to put data into.
    mask : array_like
        Mask array. Must have the same size as `arr`.
    vals : 1-D sequence
        Values to put into `arr`. Only the first N elements are used,
        where N is the number of True values in mask. If vals is smaller
        than N, it will be repeated, and if elements of a are to be masked,
        this sequence must be non-empty.

    See Also
    --------
    numpy.copyto, numpy.put, numpy.take, numpy.extract

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if arr.size == 0:
        return

    check_writeable(arr)

    if mask.size != arr.size:
        raise ValueError("arr array and condition array must be of same size")

    if vals.ndim != 1:
        raise ValueError("vals array has to be 1-dimensional")

    if mask.shape != arr.shape:
        mask_reshape = reshape(mask, arr.shape)
    else:
        mask_reshape = mask

    num_values = int(count_nonzero(mask_reshape))
    if num_values == 0:
        return

    if vals.size == 0:
        raise ValueError("vals array cannot be empty")

    if num_values != vals.size:
        reps = (num_values + vals.size - 1) // vals.size
        vals_resized = tile(A=vals, reps=reps) if reps > 1 else vals
        vals_resized = vals_resized[:num_values]
    else:
        vals_resized = vals

    if mask_reshape.dtype == bool:
        arr._thunk.set_item(mask_reshape._thunk, vals_resized._thunk)
    else:
        bool_mask = mask_reshape.astype(bool)
        arr._thunk.set_item(bool_mask._thunk, vals_resized._thunk)


# Indexing-like operations
def indices(
    dimensions: Sequence[int], dtype: npt.DTypeLike = int, sparse: bool = False
) -> ndarray | tuple[ndarray, ...]:
    """
    Return an array representing the indices of a grid.
    Compute an array where the subarrays contain index values 0, 1, ...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : Sequence[int]
        The shape of the grid.
    dtype : data-type, optional
        Data type of the result.
    sparse : bool, optional
        Return a sparse representation of the grid instead of a dense
        representation. Default is False.

    Returns
    -------
    grid : ndarray or tuple[ndarray, ...]
        If sparse is False returns one array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.
        If sparse is True returns a tuple of arrays, with
        ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` with
        dimensions[i] in the ith place

    See Also
    --------
    numpy.indices

    Notes
    -----
    The output shape in the dense case is obtained by prepending the number
    of dimensions in front of the tuple of dimensions, i.e. if `dimensions`
    is a tuple ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N, r0, ..., rN-1)``.
    The subarrays ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly:

        grid[k, i0, i1, ..., iN-1] = ik

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # implementation of indices routine is adapted from NumPy
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res_tuple: tuple[ndarray, ...] = ()
        for i, dim in enumerate(dimensions):
            idx = arange(dim, dtype=dtype).reshape(
                shape[:i] + (dim,) + shape[i + 1 :]
            )
            res_tuple += (idx,)
        return res_tuple
    else:
        out_shape = (N,) + dimensions
        res_array: ndarray = empty(out_shape, dtype=dtype)
        for i, dim in enumerate(dimensions):
            idx = arange(dim, dtype=dtype).reshape(
                shape[:i] + (dim,) + shape[i + 1 :]
            )
            res_array[i] = idx
        return res_array


def mask_indices(
    n: int, mask_func: Callable[[ndarray, int], ndarray], k: int = 0
) -> tuple[ndarray, ...]:
    """
    Return the indices to access (n, n) arrays, given a masking function.

    Assume `mask_func` is a function that, for a square array a of size
    ``(n, n)`` with a possible offset argument `k`, when called as
    ``mask_func(a, k)`` returns a new array with zeros in certain locations
    (functions like :func:`cupynumeric.triu` or :func:`cupynumeric.tril`
    do precisely this). Then this function returns the indices where
    the non-zero values would be located.

    Parameters
    ----------
    n : int
        The returned indices will be valid to access arrays of shape (n, n).
    mask_func : callable
        A function whose call signature is similar to that of
        :func:`cupynumeric.triu`, :func:`cupynumeric.tril`.
        That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.
        `k` is an optional argument to the function.
    k : scalar
        An optional argument which is passed through to `mask_func`. Functions
        like :func:`cupynumeric.triu`, :func:`cupynumeric,tril`
        take a second argument that is interpreted as an offset.

    Returns
    -------
    indices : tuple of arrays.
        The `n` arrays of indices corresponding to the locations where
        ``mask_func(np.ones((n, n)), k)`` is True.

    See Also
    --------
    numpy.mask_indices

    Notes
    -----
    WARNING: ``mask_indices`` expects ``mask_function`` to call cuPyNumeric
    functions for good performance. In case non-cuPyNumeric functions are
    called by ``mask_function``, cuPyNumeric will have to materialize all data
    on the host which might result in running out of system memory.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # this implementation is based on the Cupy
    a = ones((n, n), dtype=bool)
    if not is_implemented(mask_func):
        runtime.warn(
            "Calling non-cuPyNumeric functions in mask_func can result in bad "
            "performance",
            category=UserWarning,
        )
    return mask_func(a, k).nonzero()


@add_boilerplate("indices")
def unravel_index(
    indices: ndarray, shape: NdShape, order: OrderType = "C"
) -> tuple[ndarray, ...] | ndarray:
    """
    Converts a flat index or array of flat indices into a tuple
    of coordinate arrays.

    Parameters
    ----------
    indices : array_like
        An integer array whose elements are indices into the flattened
        version of an array of dimensions ``shape``.
    shape : tuple of ints
        The shape of the array to use for unraveling ``indices``.

    order : {'C', 'F'}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    unraveled_coords : tuple of ndarray
        Each array in the tuple has the same shape as the ``indices``
        array.

    See Also
    --------
    numpy.unravel_index

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if order not in ("F", "C"):
        raise ValueError("order is not understood")

    if indices is None or not np.can_cast(
        indices.dtype, np.int64, "same_kind"
    ):
        raise TypeError("only int indices permitted")

    size = calculate_volume(shape)

    if (indices < 0).any() or (indices > size).any():
        raise ValueError("indices have out-of-bounds value(s)")

    if indices.size == 0:
        unraveled_coords = tuple(
            empty(indices.shape, dtype=indices.dtype)
            for dim in range(len(shape))
        )
        return unraveled_coords

    unraveled_coords = tuple()
    for dim in shape[::-1] if order == "C" else shape:
        unraveled_coords = (
            (indices % dim,) + unraveled_coords
            if order == "C"
            else unraveled_coords + (indices % dim,)
        )
        indices = indices // dim
    return unraveled_coords


def diag_indices(n: int, ndim: int = 2) -> tuple[ndarray, ...]:
    """
    Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array a with a.ndim >= 2 dimensions and
    shape (n, n, …, n). For a.ndim = 2 this is the usual diagonal,
    for a.ndim > 2 this is the set of indices to
    access a[i, i, ..., i] for i = [0..n-1].

    Parameters
    ----------
    n : int
        The size, along each dimension, of the arrays for which the
        returned indices can be used.
    ndim : int, optional
        The number of dimensions.

    See Also
    --------
    numpy.diag_indices

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    idx = arange(n, dtype=int)
    return (idx,) * ndim


@add_boilerplate("arr")
def diag_indices_from(arr: ndarray) -> tuple[ndarray, ...]:
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    See diag_indices for full details.

    Parameters
    ----------
    arr : array_like
        at least 2-D

    See Also
    --------
    numpy.diag_indices_from, numpy.diag_indices

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    for i in range(1, arr.ndim):
        if arr.shape[i] != arr.shape[0]:
            raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(arr.shape[0], arr.ndim)


def tril_indices(
    n: int, k: int = 0, m: int | None = None
) -> tuple[ndarray, ...]:
    """
    Return the indices for the lower-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.
    k : int, optional
        Diagonal offset (see :func:`cupynumeric.tril` for details).
    m : int, optional
        The column dimension of the arrays for which the returned
        indices will be valid.
        By default `m` is taken equal to `n`.

    Returns
    -------
    inds : tuple of arrays
        The indices for the lower-triangle. The returned tuple contains two
        arrays, each with the indices along one dimension of the array.

    See also
    --------
    numpy.tril_indices

    Notes
    -----

    Availability
    ------------
    Multiple GPUs, Multiple CPUs
    """

    tri_ = tri(n, m, k=k, dtype=bool)
    return nonzero(tri_)


@add_boilerplate("arr")
def tril_indices_from(arr: ndarray, k: int = 0) -> tuple[ndarray, ...]:
    """
    Return the indices for the lower-triangle of arr.

    See :func:`cupynumeric.tril_indices` for full details.

    Parameters
    ----------
    arr : array_like
        The indices will be valid for arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see :func:`cupynumeric.tril` for details).

    Returns
    -------
    inds : tuple of arrays
        The indices for the lower-triangle. The returned tuple contains two
        arrays, each with the indices along one dimension of the array.

    See Also
    --------
    numpy.tril_indices_from

    Notes
    -----

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # this implementation is taken from numpy
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])


def triu_indices(
    n: int, k: int = 0, m: int | None = None
) -> tuple[ndarray, ...]:
    """
    Return the indices for the upper-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The size of the arrays for which the returned indices will
        be valid.
    k : int, optional
        Diagonal offset (see :func:`cupynumeric.triu` for details).
    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.

    Returns
    -------
    inds : tuple of arrays
        The indices for the upper-triangle. The returned tuple contains two
        arrays, each with the indices along one dimension of the array.

    See also
    --------
    numpy.triu_indices

    Notes
    -----

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    tri_ = ~tri(n, m, k=k - 1, dtype=bool)
    return nonzero(tri_)


@add_boilerplate("arr")
def triu_indices_from(arr: ndarray, k: int = 0) -> tuple[ndarray, ...]:
    """
    Return the indices for the upper-triangle of arr.

    See :func:`cupynumeric.triu_indices` for full details.

    Parameters
    ----------
    arr : ndarray, shape(N, N)
        The indices will be valid for arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see :func:`cupynumeric.triu` for details).

    Returns
    -------
    inds : tuple of arrays
        The indices for the upper-triangle. The returned tuple contains two
        arrays, each with the indices along one dimension of the array.

    See Also
    --------
    numpy.triu_indices_from

    Notes
    -----

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # this implementation is taken from numpy
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])


@add_boilerplate("a")
def take(
    a: ndarray,
    indices: ndarray,
    axis: int | None = None,
    out: ndarray | None = None,
    mode: BoundsMode = "raise",
) -> ndarray:
    """
    Take elements from an array along an axis.
    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier
    to use if you need elements along a given axis. A call such as
    `np.take(arr, indices, axis=3)` is equivalent to `arr[:,:,:,indices,...]`.

    Parameters
    ----------
    a : array_like `(Ni…, M, Nk…)`
        The source array.
    indices : array_like `(Nj…)`
        The indices of the values to extract.
        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened input
        array is used.
    out : ndarray, optional `(Ni…, Nj…, Nk…)`
        If provided, the result will be placed in this array. It should be of
        the appropriate shape and dtype.
    mode : ``{'raise', 'wrap', 'clip'}``, optional
        Specifies how out-of-bounds indices will behave.
        'raise' - raise an error (default)
        'wrap' - wrap around
        'clip' - clip to the range
        'clip' mode means that all indices that are too large are replaced by
        the index that addresses the last element along that axis.
        Note that this disables indexing with negative numbers.

    Returns
    -------
    out : ndarray `(Ni…, Nj…, Nk…)`
        The returned array has the same type as a.

    Raises
    ------

    See Also
    --------
    numpy.take

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.take(indices=indices, axis=axis, out=out, mode=mode)


def ix_(*args: Any) -> tuple[ndarray, ...]:
    """
    Construct an open mesh from multiple sequences.

    This function takes N 1-D sequences and returns N outputs with N
    dimensions each, such that the shape is 1 in all but one dimension
    and the dimension with the non-unit shape value cycles through all
    N dimensions.

    Using `ix_` one can quickly construct index arrays that will index
    the cross product. ``a[np.ix_([1,3],[2,5])]`` returns the array
    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.

    Parameters
    ----------
    args : 1-D sequences
        Each sequence should be of integer or boolean type.
        Boolean sequences will be interpreted as boolean masks for the
        corresponding dimension (equivalent to passing in
        ``np.nonzero(boolean_sequence)``).

    Returns
    -------
    out : tuple of ndarrays
        N arrays with N dimensions each, with N the number of input
        sequences. Together these arrays form an open mesh.

    See Also
    --------
    ogrid, mgrid, meshgrid

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    out = []
    nd = len(args)

    for k, new in enumerate(args):
        if not isinstance(new, ndarray):
            new = asarray(new)
            if new.size == 0:
                # Explicitly type empty arrays to avoid float default
                new = new.astype(np.intp)

        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")

        if np.issubdtype(new.dtype, bool):
            (new,) = new.nonzero()

        new = new.reshape((1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)

    return tuple(out)


def _fill_fancy_index_for_along_axis_routines(
    a_shape: NdShape, axis: int, indices: ndarray
) -> tuple[ndarray, ...]:
    # the logic below is base on the cupy implementation of
    # the *_along_axis routines
    ndim = len(a_shape)
    fancy_index = []
    for i, n in enumerate(a_shape):
        if i == axis:
            fancy_index.append(indices)
        else:
            ind_shape = (1,) * i + (-1,) + (1,) * (ndim - i - 1)
            fancy_index.append(arange(n).reshape(ind_shape))
    return tuple(fancy_index)


@add_boilerplate("a", "indices")
def take_along_axis(a: ndarray, indices: ndarray, axis: int | None) -> ndarray:
    """
    Take values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to look up values in the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like
    :func:`cupynumeric.argsort` and :func:`cupynumeric.argpartition`,
    produce suitable indices for this function.

    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        Source array
    indices : ndarray (Ni..., J, Nk...)
        Indices to take along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions Ni and Nj only need to broadcast
        against `arr`.
    axis : int
        The axis to take 1d slices along. If axis is None, the input array is
        treated as if it had first been flattened to 1d, for consistency with
        :func:`cupynumeric.sort` and :func:`cupynumeric.argsort`.

    Returns
    -------
    out: ndarray (Ni..., J, Nk...)
        The indexed result. It is going to be a view to `arr` for most cases,
        except the case when `axis=Null` and `arr.ndim>1`.

    See Also
    --------
    numpy.take_along_axis

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("`indices` must be an integer array")

    computed_axis = 0
    if axis is None:
        if indices.ndim != 1:
            raise ValueError("indices must be 1D if axis=None")
        if a.ndim > 1:
            a = a.ravel()
    else:
        computed_axis = normalize_axis_index(axis, a.ndim)

    if a.ndim != indices.ndim:
        raise ValueError(
            "`indices` and `a` must have the same number of dimensions"
        )

    # The TAKE task uses 4D accessors and reshaping logic that works well
    # for <= 4D arrays. For higher dimensions, fall back to fancy indexing.
    # Also check if broadcasting is needed - TAKE task doesn't support it.

    use_take_task = False
    if a.ndim <= 4:
        # Check if broadcasting is needed
        # TAKE task requires exact dimension matches (no broadcasting at all)
        # Note: np.prod() returns 1 for empty slices
        j_src = int(np.prod(a.shape[:computed_axis]))
        n_src = int(np.prod(a.shape[computed_axis + 1 :]))
        j_ind = int(np.prod(indices.shape[:computed_axis]))
        n_ind = int(np.prod(indices.shape[computed_axis + 1 :]))

        # Broadcasting is needed if ANY dimension differs
        # (even if one is 1, that's still broadcasting in NumPy semantics)
        needs_broadcasting = (j_src != j_ind) or (n_src != n_ind)

        if not needs_broadcasting:
            use_take_task = True

    if use_take_task:
        # Use the optimized TAKE task
        # Note: NumPy's take_along_axis doesn't have a mode parameter,
        # but we use "raise" to catch out-of-bounds indices
        result_thunk = a._thunk.take_along_axis(
            indices._thunk, computed_axis, out=None, mode="raise"
        )
        return ndarray._from_thunk(result_thunk)
    else:
        # Fall back to fancy indexing for:
        # - Arrays with >4 dimensions
        # - Arrays requiring broadcasting
        return a[
            _fill_fancy_index_for_along_axis_routines(
                a.shape, computed_axis, indices
            )
        ]


@add_boilerplate("a", "indices", "values")
def put_along_axis(
    a: ndarray, indices: ndarray, values: ndarray, axis: int | None
) -> None:
    """
    Put values into the destination array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to place values into the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like
    :func:`cupynumeric.argsort` and :func:`cupynumeric.argpartition`, produce
    suitable indices for this function.

    Parameters
    ----------
    a : ndarray (Ni..., M, Nk...)
        Destination array.
    indices : ndarray (Ni..., J, Nk...)
        Indices to change along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
        against `arr`.
    values : array_like (Ni..., J, Nk...)
        values to insert at those indices. Its shape and dimension are
        broadcast to match that of `indices`.
    axis : int
        The axis to take 1d slices along. If axis is None, the destination
        array is treated as if a flattened 1d view had been created of it.
        `axis=None` case is currently supported only for 1D input arrays.

    Note
    ----
    Having duplicate entries in `indices` will result in undefined behavior
    since operation performs asynchronous update of the `arr` entries.

    See Also
    --------
    numpy.put_along_axis

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """

    if a.size == 0:
        return

    check_writeable(a)

    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("`indices` must be an integer array")

    computed_axis = 0
    if axis is None:
        if indices.ndim != 1:
            raise ValueError("indices must be 1D if axis=None")
        if a.ndim > 1:
            # TODO call a=a.flat when flat is implemented
            raise ValueError("a.ndim>1 case is not supported when axis=None")
        if (indices.size == 0) or (values.size == 0):
            return
        if values.shape != indices.shape:
            values = values._wrap(indices.size)
    else:
        computed_axis = normalize_axis_index(axis, a.ndim)

    if a.ndim != indices.ndim:
        raise ValueError(
            "`indices` and `a` must have the same number of dimensions"
        )
    ind = _fill_fancy_index_for_along_axis_routines(
        a.shape, computed_axis, indices
    )
    a[ind] = values


@add_boilerplate("a")
def choose(
    a: ndarray,
    choices: Sequence[ndarray],
    out: ndarray | None = None,
    mode: BoundsMode = "raise",
) -> ndarray:
    """
    Construct an array from an index array and a list of arrays to choose from.

    Given an "index" array (`a`) of integers and a sequence of ``n`` arrays
    (`choices`), `a` and each choice array are first broadcast, as necessary,
    to arrays of a common shape; calling these *Ba* and *Bchoices[i], i =
    0,...,n-1* we have that, necessarily, ``Ba.shape == Bchoices[i].shape``
    for each ``i``.  Then, a new array with shape ``Ba.shape`` is created as
    follows:

    * if ``mode='raise'`` (the default), then, first of all, each element of
      ``a`` (and thus ``Ba``) must be in the range ``[0, n-1]``; now, suppose
      that ``i`` (in that range) is the value at the ``(j0, j1, ..., jm)``
      position in ``Ba`` - then the value at the same position in the new array
      is the value in ``Bchoices[i]`` at that same position;

    * if ``mode='wrap'``, values in `a` (and thus `Ba`) may be any (signed)
      integer; modular arithmetic is used to map integers outside the range
      `[0, n-1]` back into that range; and then the new array is constructed
      as above;

    * if ``mode='clip'``, values in `a` (and thus ``Ba``) may be any (signed)
      integer; negative integers are mapped to 0; values greater than ``n-1``
      are mapped to ``n-1``; and then the new array is constructed as above.

    Parameters
    ----------
    a : ndarray[int]
        This array must contain integers in ``[0, n-1]``, where ``n`` is the
        number of choices, unless ``mode=wrap`` or ``mode=clip``, in which
        cases any integers are permissible.
    choices : Sequence[ndarray]
        Choice arrays. `a` and all of the choices must be broadcastable to the
        same shape.  If `choices` is itself an array (not recommended), then
        its outermost dimension (i.e., the one corresponding to
        ``choices.shape[0]``) is taken as defining the "sequence".
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if ``mode='raise'``; use other modes for better performance.
    mode : ``{'raise', 'wrap', 'clip'}``, optional
        Specifies how indices outside ``[0, n-1]`` will be treated:

          * 'raise' : an exception is raised (default)
          * 'wrap' : value becomes value mod ``n``
          * 'clip' : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array : ndarray
        The merged result.

    Raises
    ------
    ValueError: shape mismatch
        If `a` and each choice array are not all broadcastable to the same
        shape.

    See Also
    --------
    numpy.choose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.choose(choices=choices, out=out, mode=mode)


def select(
    condlist: Sequence[npt.ArrayLike | ndarray],
    choicelist: Sequence[npt.ArrayLike | ndarray],
    default: Any = 0,
) -> ndarray:
    """
    Return an array drawn from elements in choicelist, depending on conditions.

    Parameters
    ----------
    condlist : list of bool ndarrays
        The list of conditions which determine from which array in `choicelist`
        the output elements are taken. When multiple conditions are satisfied,
        the first one encountered in `condlist` is used.
    choicelist : list of ndarrays
        The list of arrays from which the output elements are taken. It has
        to be of the same length as `condlist`.
    default : scalar, optional
        The element inserted in `output` when all conditions evaluate to False.

    Returns
    -------
    output : ndarray
        The output at position m is the m-th element of the array in
        `choicelist` where the m-th element of the corresponding array in
        `condlist` is True.

    See Also
    --------
    numpy.select

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if len(condlist) != len(choicelist):
        raise ValueError(
            "list of cases must be same length as list of conditions"
        )
    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    condlist_ = tuple(convert_to_cupynumeric_ndarray(c) for c in condlist)
    for i, c in enumerate(condlist_):
        if c.dtype != bool:
            raise TypeError(
                f"invalid entry {i} in condlist: should be boolean ndarray"
            )

    choicelist_ = tuple(convert_to_cupynumeric_ndarray(c) for c in choicelist)
    common_type = np.result_type(*choicelist_, default)
    args = condlist_ + choicelist_
    choicelist_ = tuple(
        c._maybe_convert(common_type, args) for c in choicelist_
    )
    default_ = np.array(default, dtype=common_type)

    out_shape = np.broadcast_shapes(
        *(c.shape for c in condlist_), *(c.shape for c in choicelist_)
    )
    out = ndarray._from_inputs(shape=out_shape, dtype=common_type, inputs=args)
    out._thunk.select(
        tuple(c._thunk for c in condlist_),
        tuple(c._thunk for c in choicelist_),
        default_,
    )
    return out


@add_boilerplate("condition", "a")
def compress(
    condition: ndarray,
    a: ndarray,
    axis: int | None = None,
    out: ndarray | None = None,
) -> ndarray:
    """
    Return selected slices of an array along given axis.

    When working along a given axis, a slice along that axis is returned
    in output for each index where condition evaluates to True.
    When working on a 1-D array, compress is equivalent to numpy.extract.

    Parameters
    ----------
    condition, 1-D array of bools
        Array that selects which entries to return. If `len(c)` is less than
        the size of a along the given axis, then output is truncated to the
        length of the condition array.

    a : array_like
        Array from which to extract a part.

    axis: int, optional
        Axis along which to take slices. If None (default),
        work on the flattened array.

    out : ndarray, optional
        Output array. Its type is preserved and it must be of the right
        shape to hold the output.

    Returns
    -------
    compressed_array : ndarray
        A copy of `a` without the slices along `axis` for which condition
        is false.

    Raises
    ------
    ValueError : dimension mismatch
        If condition is not 1D array
    ValueError : shape mismatch
        If condition contains entries that are out of bounds of array
    ValueError : shape mismatch
        If output array has a wrong shape

    See Also
    --------
    numpy.compress, numpy.extract

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    return a.compress(condition, axis=axis, out=out)


@add_boilerplate("a")
def diagonal(
    a: ndarray,
    offset: int = 0,
    axis1: int = 0,
    axis2: int = 1,
    extract: bool = True,
) -> ndarray:
    """
    diagonal(a: ndarray, offset=0, axis1=None, axis2=None)

    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.  If
    `a` has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal is
    returned.  The shape of the resulting array can be determined by
    removing `axis1` and `axis2` and appending an index to the right equal
    to the size of the resulting diagonals.

    Parameters
    ----------
    a : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).
    axis1 : int, optional
        Axis to be used as the first axis of the 2-D sub-arrays from
        which the diagonals should be taken.  Defaults to first axis (0).
    axis2 : int, optional
        Axis to be used as the second axis of the 2-D sub-arrays from
        which the diagonals should be taken. Defaults to second axis (1).

    Returns
    -------
    array_of_diagonals : ndarray
        If `a` is 2-D, then a 1-D array containing the diagonal and of the
        same type as `a` is returned unless `a` is a `matrix`, in which case
        a 1-D array rather than a (2-D) `matrix` is returned in order to
        maintain backward compatibility.

        If ``a.ndim > 2``, then the dimensions specified by `axis1` and `axis2`
        are removed, and a new axis inserted at the end corresponding to the
        diagonal.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    Notes
    -----
    Unlike NumPy's, the cuPyNumeric implementation always returns a copy

    See Also
    --------
    numpy.diagonal

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    if a is None or a.ndim < 1:
        raise ValueError("diag requires an array of at least two dimensions")
    return a.diagonal(offset=offset, axis1=axis1, axis2=axis2, extract=extract)


@add_boilerplate("a", "indices", "values")
def put(
    a: ndarray, indices: ndarray, values: ndarray, mode: str = "raise"
) -> None:
    """
    Replaces specified elements of an array with given values.
    The indexing works as if the target array is first flattened.

    Parameters
    ----------
    a : array_like
        Array to put data into
    indices : array_like
        Target indices, interpreted as integers.
        WARNING: In case there are repeated entries in the
        indices array, Legate doesn't guarantee the order in
        which values are updated.

    values : array_like
        Values to place in `a` at target indices. If values array is shorter
        than indices, it will be repeated as necessary.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.
        'raise' : raise an error.
        'wrap' : wrap around.
        'clip' : clip to the range.

    See Also
    --------
    numpy.put

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    a.put(indices=indices, values=values, mode=mode)


@add_boilerplate("a", "mask", "values")
def putmask(a: ndarray, mask: ndarray, values: ndarray) -> None:
    """
    putmask(a, mask, values)
    Changes elements of an array based on conditional and input values.
    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.
    If `values` is not the same size as `a` and `mask` then it will repeat.
    This gives behavior different from ``a[mask] = values``.

    Parameters
    ----------
    a : ndarray
        Target array.
    mask : array_like
        Boolean mask array. It has to be the same shape as `a`.
    values : array_like
        Values to put into `a` where `mask` is True. If `values` is smaller
        than `a` it will be repeated.

    See Also
    --------
    numpy.putmask

    Availability
    ------------
    Multiple GPUs, Multiple CPUs
    """
    if not a.shape == mask.shape:
        raise ValueError("mask and data must be the same size")

    check_writeable(a)

    mask = _warn_and_convert(mask, np.dtype(bool))

    if a.dtype != values.dtype:
        values = _warn_and_convert(values, a.dtype)

    try:
        np.broadcast_shapes(values.shape, a.shape)
    except ValueError:
        values = values._wrap(a.size)
        values = values.reshape(a.shape)

    a._thunk.putmask(mask._thunk, values._thunk)


@add_boilerplate("a", "val")
def fill_diagonal(a: ndarray, val: ndarray, wrap: bool = False) -> None:
    """
    Fill the main diagonal of the given array of any dimensionality.

    For an array a with a.ndim >= 2, the diagonal is the list of locations with
    indices a[i, ..., i] all identical. This function modifies the input
    array in-place, it does not return a value.

    Parameters
    ----------

    a : array, at least 2-D.
        Array whose diagonal is to be filled, it gets modified in-place.
    val : scalar or array_like
        Value(s) to write on the diagonal. If val is scalar, the value is
        written along the diagonal.
        If array-like, the flattened val is written along
        the diagonal, repeating if necessary to fill all diagonal entries.
    wrap : bool
        If true, the diagonal "wraps" after N columns, for tall 2d matrices.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    Notes
    -----

    See Also
    --------
    numpy.fill_diagonal

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    if val is None or np.isnan(val).any():
        raise ValueError("cannot convert float NaN to integer")

    if val.size == 0 or a.size == 0:
        return

    check_writeable(a)

    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")

    n = _builtin_min(a.shape)

    if a.ndim > 2:
        for s in a.shape:
            if s != n:
                raise ValueError(
                    "All dimensions of input must be of equal length"
                )

    len_val = n

    if a.ndim == 2 and wrap and a.shape[0] > a.shape[1]:
        len_val = a.shape[0] - (a.shape[0] // (a.shape[1] + 1))

    if (val.size != len_val and val.ndim > 0) or val.ndim > 1:
        val = val._wrap(len_val)

    if a.ndim == 2 and wrap and a.shape[0] > a.shape[1]:
        idx0_tmp = arange(a.shape[1], dtype=int)
        idx0 = idx0_tmp.copy()
        while idx0.size < len_val:
            idx0_tmp = idx0_tmp + (a.shape[1] + 1)
            idx0 = hstack((idx0, idx0_tmp))
        idx0 = idx0[0:len_val]
        idx1 = arange(len_val, dtype=int) % a.shape[1]
        a[idx0, idx1] = val
    else:
        idx = arange(n, dtype=int)
        indices = (idx,) * a.ndim

        a[indices] = val


def ravel_multi_index(
    multi_index: tuple[ndarray, ...] | ndarray,
    dims: NdShape,
    mode: BoundsMode | tuple[BoundsMode, ...] = "raise",
    order: OrderType = "C",
) -> ndarray:
    from .math_misc import clip

    """
    Converts a tuple of index arrays into an array of flat indices, applying
    boundary modes to the multi-index.

    Parameters
    ----------
    multi_index : tuple of array_like
        A tuple of integer arrays, one array for each dimension.
    dims : tuple of ints
        The shape of array into which the indices from `multi_index` apply.
    mode : {'raise', 'wrap', 'clip'} or tuple of {'raise', 'wrap', 'clip'},
    optional
        Specifies how out-of-bounds indices are handled. Can specify either
        one mode or a tuple of modes, one mode per index.
        * 'raise' - raise an error (default)
        * 'wrap' - wrap around
        * 'clip' - clip to the range
        In 'clip' mode, a negative index which would normally wrap will clip
        to 0 instead.
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    raveled_indices : ndarray
        An array of indices into the flattened version of an array of
        dimensions `dims`.

    See Also
    --------
    nump.ravel_multi_index

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """

    if isinstance(multi_index, tuple):
        multi_index = tuple(
            convert_to_cupynumeric_ndarray(a) for a in multi_index
        )
    else:
        multi_index = (convert_to_cupynumeric_ndarray(multi_index),)

    dims = tuple(dims)
    ndim = len(dims)

    for d in dims:
        if d == 0:
            raise ValueError(
                "cannot unravel if shape has zero entries (is empty)."
            )
        if not np.issubdtype(type(d), int):
            raise TypeError(
                f"'{type(d).__name__}' object cannot be interpreted as an integer"
            )

    if all(i == 0 for i in dims):
        return convert_to_cupynumeric_ndarray(0)

    for arr in multi_index:
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError("only int indices permitted")

    if len(multi_index) != ndim:
        raise ValueError(
            f"parameter multi_index must be a sequence of length {ndim}"
        )

    # Convert mode to tuple if it's a single string
    if isinstance(mode, str):
        mode = (mode,) * ndim
    elif len(mode) != ndim:
        raise ValueError(
            f"mode length ({len(mode)}) does not match dimensionality of "
            f"target array ({ndim})"
        )

    multi_index = tuple(broadcast_arrays(*multi_index))

    # Handle each index according to its mode
    indices = []
    for idx, dim, m in zip(multi_index, dims, mode):
        if m == "raise":
            if ((idx < 0) | (idx >= dim)).any():
                raise ValueError("invalid entry in coordinates array")
        elif m == "wrap":
            idx = idx % dim
        elif m == "clip":
            idx = clip(idx, 0, dim - 1)
        else:
            raise ValueError(f"invalid mode: {m}")
        indices.append(idx)

    # Calculate strides based on order
    if order == "C":
        strides = [1]
        for i in range(ndim - 1, 0, -1):
            strides.insert(0, strides[0] * dims[i])
    elif order == "F":
        strides = [1]
        for i in range(0, ndim - 1):
            strides.append(strides[-1] * dims[i])
    else:
        raise ValueError(f"only 'C' or 'F' order is permitted, not {order}")

    # Calculate raveled indices
    raveled = zeros_like(indices[0])
    for idx, stride in zip(indices, strides):
        raveled = raveled + idx * stride

    return raveled
