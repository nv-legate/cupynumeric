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

from itertools import chain
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from .._array.array import ndarray
from .._array.util import add_boilerplate, convert_to_cupynumeric_ndarray
from ..lib.array_utils import normalize_axis_index
from .array_dimension import _atleast_nd
from .array_transpose import moveaxis
from .creation_ranges import arange
from .creation_shape import empty, ones, zeros
from .ssc_searching import flatnonzero

if TYPE_CHECKING:
    import numpy.typing as npt

    from .._ufunc.ufunc import CastingKind
    from ..types import NdShape

_builtin_any = any
_builtin_max = max
_builtin_sum = sum

casting_kinds: tuple[CastingKind, ...] = (
    "no",
    "equiv",
    "safe",
    "same_kind",
    "unsafe",
)


class ArrayInfo:
    def __init__(
        self, ndim: int, shape: NdShape, dtype: np.dtype[Any]
    ) -> None:
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype


def convert_to_array_form(indices: Sequence[int]) -> str:
    return "".join(f"[{coord}]" for coord in indices)


def check_list_depth(arr: Any, prefix: NdShape = (0,)) -> int:
    if not isinstance(arr, list):
        return 0
    elif len(arr) == 0:
        raise ValueError(
            f"List at arrays{convert_to_array_form(prefix)} cannot be empty"
        )

    depths = list(
        check_list_depth(each, prefix + (idx,)) for idx, each in enumerate(arr)
    )

    if len(set(depths)) != 1:  # this should be one
        # If we're here elements don't have the same depth
        first_depth = depths[0]
        for idx, other_depth in enumerate(depths[1:]):
            if other_depth != first_depth:
                raise ValueError(
                    "List depths are mismatched. First element was at depth "
                    f"{first_depth}, but there is an element at"
                    f" depth {other_depth}, "
                    f"arrays{convert_to_array_form(prefix + (idx + 1,))}"
                )

    return depths[0] + 1


def check_shape_with_axis(
    inputs: list[ndarray], func_name: str, axis: int
) -> None:
    ndim = inputs[0].ndim
    shape = inputs[0].shape

    axis = normalize_axis_index(axis, ndim)
    if ndim >= 1:
        if _builtin_any(
            shape[:axis] != inp.shape[:axis]
            or shape[axis + 1 :] != inp.shape[axis + 1 :]
            for inp in inputs
        ):
            raise ValueError(
                f"All arguments to {func_name} "
                "must have the same "
                "dimension size in all dimensions "
                "except the target axis"
            )
    return


def check_shape_dtype_without_axis(
    inputs: Sequence[ndarray],
    func_name: str,
    dtype: npt.DTypeLike | None = None,
    casting: CastingKind = "same_kind",
) -> tuple[list[ndarray], ArrayInfo]:
    if len(inputs) == 0:
        raise ValueError("need at least one array to concatenate")

    inputs = list(convert_to_cupynumeric_ndarray(inp) for inp in inputs)
    ndim = inputs[0].ndim
    shape = inputs[0].shape

    if _builtin_any(ndim != inp.ndim for inp in inputs):
        raise ValueError(
            f"All arguments to {func_name} must have the same number of dimensions"
        )

    # Cast arrays with the passed arguments (dtype, casting)
    if dtype is None:
        dtype = np.result_type(*[inp.dtype for inp in inputs])
    else:
        dtype = np.dtype(dtype)

    converted = list(inp.astype(dtype, casting=casting) for inp in inputs)
    return converted, ArrayInfo(ndim, shape, dtype)


def _block_collect_slices(
    arr: ndarray | Sequence[ndarray], cur_depth: int, depth: int
) -> tuple[list[Any], NdShape, Sequence[ndarray]]:
    # collects slices for each array in `arr`
    # the outcome will be slices on every dimension of the output array
    # for each array in `arr`
    if cur_depth < depth:
        sublist_results = list(
            _block_collect_slices(each, cur_depth + 1, depth) for each in arr
        )
        # 'sublist_results' contains a list of 3-way tuples,
        # for arrays, out_shape of the sublist, and slices
        arrays, outshape_list, slices = zip(*sublist_results)
        max_ndim = _builtin_max(
            1 + (depth - cur_depth), *(len(each) for each in outshape_list)
        )
        outshape_list = list(
            ((1,) * (max_ndim - len(each)) + tuple(each))
            for each in outshape_list
        )
        leading_dim = _builtin_sum(
            each[-1 + (cur_depth - depth)] for each in outshape_list
        )
        # flatten array lists from sublists into a single list
        arrays = list(chain(*arrays))
        # prepares the out_shape of the current list
        shape = list(outshape_list[0])
        shape[-1 + cur_depth - depth] = leading_dim
        out_shape = tuple(shape)
        offset = 0
        updated_slices = []
        # update the dimension in each slice for the current axis
        for shape, slice_list in zip(outshape_list, slices):
            cur_dim = shape[-1 + cur_depth - depth]
            updated_slices.append(
                list(
                    (slice(offset, offset + cur_dim),) + each
                    for each in slice_list
                )
            )
            offset += cur_dim
        # flatten lists of slices into a single list
        slices = list(chain(*updated_slices))
    else:
        arrays = list(convert_to_cupynumeric_ndarray(inp) for inp in arr)
        common_shape = arrays[0].shape
        if len(arr) > 1:
            arrays, common_info = check_shape_dtype_without_axis(
                arrays, block.__name__
            )
            common_shape = common_info.shape
            check_shape_with_axis(arrays, block.__name__, axis=-1)
        # the initial slices for each arr on arr.shape[-1]
        out_shape, slices, arrays = _collect_outshape_slices(
            arrays, common_shape, axis=-1 + len(common_shape)
        )

    return arrays, out_shape, slices


def _block_slicing(arrays: Sequence[ndarray], depth: int) -> ndarray:
    # collects the final slices of input arrays and assign them at once
    arrays, out_shape, slices = _block_collect_slices(arrays, 1, depth)
    out_array = ndarray._from_inputs(shape=out_shape, inputs=arrays)

    for dest, inp in zip(slices, arrays):
        out_array[(Ellipsis,) + tuple(dest)] = inp

    return out_array


def _collect_outshape_slices(
    inputs: Sequence[ndarray], common_shape: NdShape, axis: int
) -> tuple[NdShape, list[tuple[slice, ...]], list[ndarray]]:
    leading_dim = _builtin_sum(arr.shape[axis] for arr in inputs)
    out_shape = list(common_shape)
    out_shape[axis] = leading_dim
    post_idx = (slice(None),) * len(out_shape[axis + 1 :])
    slices = []
    offset = 0
    # collect slices for arrays in `inputs`
    inputs = [inp for inp in inputs if inp.size > 0]
    for inp in inputs:
        slices.append((slice(offset, offset + inp.shape[axis]),) + post_idx)
        offset += inp.shape[axis]

    return tuple(out_shape), slices, inputs


def _concatenate(
    inputs: Sequence[ndarray],
    common_info: ArrayInfo,
    axis: int = 0,
    out: ndarray | None = None,
    dtype: npt.DTypeLike | None = None,
    casting: CastingKind = "same_kind",
) -> ndarray:
    if axis < 0:
        axis += len(common_info.shape)
    out_shape, slices, inputs = _collect_outshape_slices(
        inputs, common_info.shape, axis
    )

    if out is None:
        out_array = ndarray._from_inputs(
            shape=out_shape, dtype=common_info.dtype, inputs=inputs
        )
    else:
        out_array = convert_to_cupynumeric_ndarray(out)
        if out_array.shape != out_shape:
            raise ValueError(
                f"out.shape({out.shape}) is not matched "
                f"to the result shape of concatenation ({out_shape})"
            )

    for dest, src in zip(slices, inputs):
        out_array[(Ellipsis,) + dest] = src

    return out_array


def append(arr: ndarray, values: ndarray, axis: int | None = None) -> ndarray:
    """

    Append values to the end of an array.

    Parameters
    ----------
    arr :  array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of arr. It must be of the correct
        shape (the same shape as arr, excluding axis). If axis is not
        specified, values can be any shape and will be flattened before use.
    axis : int, optional
        The axis along which values are appended. If axis is not given, both
        `arr` and `values` are flattened before use.

    Returns
    -------
    res : ndarray
        A copy of arr with values appended to axis.

    See Also
    --------
    numpy.append

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # Check to see if we can build a new tuple of cuPyNumeric arrays
    inputs = list(convert_to_cupynumeric_ndarray(inp) for inp in [arr, values])
    return concatenate(inputs, axis)


def block(arrays: Sequence[Any]) -> ndarray:
    """
    Assemble an nd-array from nested lists of blocks.

    Blocks in the innermost lists are concatenated (see concatenate)
    along the last dimension (-1), then these are concatenated along
    the second-last dimension (-2), and so on until the outermost
    list is reached.

    Blocks can be of any dimension, but will not be broadcasted using
    the normal rules. Instead, leading axes of size 1 are inserted,
    to make block.ndim the same for all blocks. This is primarily useful
    for working with scalars, and means that code like np.block([v, 1])
    is valid, where v.ndim == 1.

    When the nested list is two levels deep, this allows block matrices
    to be constructed from their components.

    Parameters
    ----------
    arrays : nested list of array_like or scalars
        If passed a single ndarray or scalar (a nested list of depth 0),
        this is returned unmodified (and not copied).

        Elements shapes must match along the appropriate axes (without
        broadcasting), but leading 1s will be prepended to the shape as
        necessary to make the dimensions match.

    Returns
    -------
    block_array : ndarray
        The array assembled from the given blocks.
        The dimensionality of the output is equal to the greatest of: * the
        dimensionality of all the inputs * the depth to which the input list
        is nested

    Raises
    ------
    ValueError
        If list depths are mismatched - for instance, [[a, b], c] is
        illegal, and should be spelt [[a, b], [c]]
        If lists are empty - for instance, [[a, b], []]

    See Also
    --------
    numpy.block

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # arrays should concatenate from innermost subarrays
    # the 'arrays' should be balanced tree
    # check if the 'arrays' is a balanced tree
    depth = check_list_depth(arrays)

    result = _block_slicing(arrays, depth)
    return result


def _eager(x: Any) -> Any:
    if not hasattr(x, "_thunk"):
        return x
    from .._thunk.eager import EagerArray

    if isinstance(x._thunk, EagerArray):
        return x._thunk.array
    raise ValueError


def concatenate(
    inputs: Sequence[ndarray],
    axis: int | None = 0,
    out: ndarray | None = None,
    dtype: npt.DTypeLike | None = None,
    casting: CastingKind = "same_kind",
) -> ndarray:
    """

    concatenate((a1, a2, ...), axis=0, out=None, dtype=None,
    casting="same_kind")

    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : Sequence[array_like]
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    dtype : str or data-type
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.
    casting : ``{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}``, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

    Returns
    -------
    res : ndarray
        The concatenated array.

    See Also
    --------
    numpy.concatenate

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    # special case for when all inputs are eager, i.e not DeferredArray,
    # specifically -- fall back immediately to numpy instead
    try:
        eager_inputs = [_eager(x) for x in inputs]
        eager_out = _eager(out)
        result = np.concatenate(
            eager_inputs,
            axis=axis,
            out=eager_out,
            dtype=dtype,
            casting=casting,
        )
        return convert_to_cupynumeric_ndarray(result)
    except Exception:
        pass

    if dtype is not None and out is not None:
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an argument,"
            "but both were provided."
        )

    if casting not in casting_kinds:
        raise ValueError(
            "casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'"
        )

    # flatten arrays if axis == None and concatenate arrays on the first axis
    if axis is None:
        # Reshape arrays in the `array_list` to handle scalars
        reshaped = _atleast_nd(1, inputs)
        if not isinstance(reshaped, list):
            reshaped = [reshaped]
        inputs = list(inp.ravel() for inp in reshaped)
        axis = 0

    # Check to see if we can build a new tuple of cuPyNumeric arrays
    cupynumeric_inputs, common_info = check_shape_dtype_without_axis(
        inputs, concatenate.__name__, dtype, casting
    )
    check_shape_with_axis(cupynumeric_inputs, concatenate.__name__, axis)

    return _concatenate(
        cupynumeric_inputs, common_info, axis, out, dtype, casting
    )


def stack(
    arrays: Sequence[ndarray], axis: int = 0, out: ndarray | None = None
) -> ndarray:
    """

    Join a sequence of arrays along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    Parameters
    ----------
    arrays : Sequence[array_like]
        Each array must have the same shape.

    axis : int, optional
        The axis in the result array along which the input arrays are stacked.

    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no
        out argument were specified.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    See Also
    --------
    numpy.stack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if type(axis) is not int:
        raise TypeError("The target axis should be an integer")

    arrays, common_info = check_shape_dtype_without_axis(
        arrays, stack.__name__
    )
    shapes = {inp.shape for inp in arrays}
    if len(shapes) != 1:
        raise ValueError("all input arrays must have the same shape for stack")

    axis = normalize_axis_index(axis, common_info.ndim + 1)
    shape = common_info.shape[:axis] + (1,) + common_info.shape[axis:]
    arrays = [arr.reshape(shape) for arr in arrays]
    common_info.shape = shape
    return _concatenate(arrays, common_info, axis, out=out)


def vstack(tup: Sequence[ndarray]) -> ndarray:
    """

    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : Sequence[ndarray]
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    numpy.vstack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Reshape arrays in the `array_list` if needed before concatenation
    reshaped = _atleast_nd(2, tup)
    if not isinstance(reshaped, list):
        reshaped = [reshaped]
    tup, common_info = check_shape_dtype_without_axis(
        reshaped, vstack.__name__
    )
    check_shape_with_axis(tup, vstack.__name__, 0)
    return _concatenate(tup, common_info, axis=0, dtype=common_info.dtype)


def hstack(tup: Sequence[ndarray]) -> ndarray:
    """

    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : Sequence[ndarray]
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    numpy.hstack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Reshape arrays in the `array_list` to handle scalars
    reshaped = _atleast_nd(1, tup)
    if not isinstance(reshaped, list):
        reshaped = [reshaped]

    tup, common_info = check_shape_dtype_without_axis(
        reshaped, hstack.__name__
    )
    check_shape_with_axis(
        tup, hstack.__name__, axis=(0 if common_info.ndim == 1 else 1)
    )
    # When ndim == 1, hstack concatenates arrays along the first axis
    return _concatenate(
        tup,
        common_info,
        axis=(0 if common_info.ndim == 1 else 1),
        dtype=common_info.dtype,
    )


def dstack(tup: Sequence[ndarray]) -> ndarray:
    """

    Stack arrays in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : Sequence[ndarray]
        The arrays must have the same shape along all but the third axis.
        1-D or 2-D arrays must have the same shape.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 3-D.

    See Also
    --------
    numpy.dstack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Reshape arrays to (1,N,1) for ndim ==1 or (M,N,1) for ndim == 2:
    reshaped = _atleast_nd(3, tup)
    if not isinstance(reshaped, list):
        reshaped = [reshaped]
    tup, common_info = check_shape_dtype_without_axis(
        reshaped, dstack.__name__
    )
    check_shape_with_axis(tup, dstack.__name__, 2)
    return _concatenate(tup, common_info, axis=2, dtype=common_info.dtype)


def column_stack(tup: Sequence[ndarray]) -> ndarray:
    """

    Stack 1-D arrays as columns into a 2-D array.

    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.

    Parameters
    ----------
    tup : Sequence[ndarray]
        1-D or 2-D arrays to stack. All of them must have the same
        first dimension.

    Returns
    -------
    stacked : ndarray
        The 2-D array formed by stacking the given arrays.

    See Also
    --------
    numpy.column_stack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Reshape arrays in the `array_list` to handle scalars
    reshaped = _atleast_nd(1, tup)
    if not isinstance(reshaped, list):
        reshaped = [reshaped]

    tup, common_info = check_shape_dtype_without_axis(
        reshaped, column_stack.__name__
    )

    if common_info.ndim == 1:
        tup = list(inp.reshape((inp.shape[0], 1)) for inp in tup)
        common_info.shape = tup[0].shape
    check_shape_with_axis(tup, column_stack.__name__, 1)
    return _concatenate(tup, common_info, axis=1, dtype=common_info.dtype)


row_stack = vstack


@add_boilerplate("arr", "values")
def insert(
    arr: ndarray,
    obj: int | slice | Sequence[int] | ndarray,
    values: ndarray,
    axis: int | None = None,
) -> ndarray:
    """
    Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : int, slice or sequence of ints
        Object that defines the index or indices before which `values` is
        inserted.
    values : array_like
        Values to insert into `arr`. If the type of `values` is different from
        that of `arr`, `values` is converted to the type of `arr`.
        `values` should be shaped so that `arr[...,obj,...] = values` is legal.
    axis : int, optional
        Axis along which to insert `values`. If `axis` is None then `arr`
        is flattened first.

    Returns
    -------
    out : ndarray
        A copy of `arr` with `values` inserted. Note that `insert` does not
        occur in-place: a new array is returned. If `axis` is None, `out` is
        a flattened array.

    See Also
    --------
    np.append

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    ndim = arr.ndim

    if axis is None:
        arr = arr.ravel()
        ndim = 1
        axis = 0
    else:
        axis = normalize_axis_index(axis, ndim)
    slobj = [slice(None)] * ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    if isinstance(obj, slice):
        indices = arange(*obj.indices(N), dtype=np.intp)
    else:
        if isinstance(obj, ndarray) and obj.dtype == bool:
            if obj.ndim != 1:
                raise ValueError(
                    "boolean array argument obj to insert must be one dimensional"
                )
            indices = flatnonzero(obj)
        else:
            indices = convert_to_cupynumeric_ndarray(obj)
            if indices.ndim > 1:
                raise ValueError(
                    "index array argument obj to insert must be one "
                    "dimensional or scalar"
                )
    if indices.size == 1:
        index = indices.item()
        if index < -N or index > N:
            raise IndexError(
                f"index {obj} is out of bounds for axis {axis} with size {N}"
            )
        if index < 0:
            index += N

        values = convert_to_cupynumeric_ndarray(
            np.array(values, copy=None, ndmin=arr.ndim, dtype=arr.dtype)
        )
        if indices.ndim == 0:
            # Convert values to numpy for moveaxis, then back to cupynumeric
            values_np = moveaxis(
                convert_to_cupynumeric_ndarray(values), [0], [axis]
            )
            values = convert_to_cupynumeric_ndarray(values_np)
        numnew = values.shape[axis]
        newshape[axis] += numnew
        new = empty(tuple(newshape), arr.dtype)

        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(index, index + numnew)
        new[tuple(slobj)] = values
        slobj[axis] = slice(index + numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]

        return new

    elif indices.size == 0 and not isinstance(obj, ndarray):
        indices = indices.astype(np.intp)

    indices[indices < 0] += N

    numnew = len(indices)
    order = indices.argsort(kind="mergesort")
    indices[order] += arange(numnew)

    newshape[axis] += numnew
    old_mask = ones(newshape[axis], dtype=bool)
    false_mask = zeros(indices.shape, dtype=bool)
    old_mask[indices] = false_mask

    new = empty(tuple(newshape), arr.dtype)
    values_index_tuple = (
        (slice(None),) * axis + (indices,) + (slice(None),) * (ndim - axis - 1)
    )
    arr_index_tuple = (
        (slice(None),) * axis
        + (old_mask,)
        + (slice(None),) * (ndim - axis - 1)
    )

    new[values_index_tuple] = values
    new[arr_index_tuple] = arr

    return new
