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

import numbers
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import numpy as np

from .._array.array import _warn_and_convert, ndarray
from .._array.util import add_boilerplate, convert_to_cupynumeric_ndarray
from .._utils import is_np2
from ..lib.array_utils import normalize_axis_index
from ..runtime import runtime
from .array_rearrange import flip
from .array_transpose import moveaxis
from .creation_ranges import linspace
from .creation_shape import full, zeros
from .math_extrema import amax, amin
from .stats_avgs_vars import mean, median

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShape

if is_np2:
    from numpy.exceptions import AxisError
else:
    from numpy import AxisError  # type: ignore[no-redef,attr-defined]

_builtin_max = max

# Statistical functions for pad modes
_STAT_FUNCTIONS = {
    "mean": mean,
    "maximum": amax,
    "minimum": amin,
    "median": median,
}

# Statistical functions that return per-axis results (for Python fallback)
_STAT_FUNCTIONS_AXIS = {
    "mean": lambda x, axis: mean(x, axis=axis, keepdims=True),
    "maximum": lambda x, axis: amax(x, axis=axis, keepdims=True),
    "minimum": lambda x, axis: amin(x, axis=axis, keepdims=True),
    "median": lambda x, axis: median(x, axis=axis, keepdims=True),
}


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
    result = ndarray._from_inputs(out_shape, dtype=A.dtype, inputs=(A,))
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
                "converting repeats to an integer type", category=UserWarning
            )
        result = array._thunk.repeat(
            repeats=np.int64(repeats), axis=axis_int, scalar_repeats=True
        )
    # repeats is an array
    else:
        # repeats should be integer type
        repeats = _warn_and_convert(repeats, np.dtype(np.int64))
        if repeats.shape[0] != array.shape[axis_int]:
            raise ValueError("incorrect shape of repeats array")
        result = array._thunk.repeat(
            repeats=repeats._thunk, axis=axis_int, scalar_repeats=False
        )
    return ndarray._from_thunk(result)


@add_boilerplate("x")
def _as_pairs(
    x: Any, ndim: int, as_index: bool = False
) -> tuple[tuple[int, int], ...]:
    """
    Broadcast `x` to an array with the shape (`ndim`, 2).

    A helper function for `pad` that prepares and validates arguments like
    `pad_width` for iteration in pairs.
    """
    if as_index:
        x = x.astype(np.intp)

    if x.ndim == 0:
        # Scalar → same before/after padding on every axis
        return tuple((int(x), int(x)) for _ in range(ndim))
    if x.ndim == 1:
        if x.shape[0] == 1:
            # Length-1 sequence → broadcast identical pair to each axis
            return tuple((int(x[0]), int(x[0])) for _ in range(ndim))
        if x.shape[0] == ndim:
            # One entry per axis → treat as symmetric padding on that axis. Note that this
            # branch remains unambiguous even when ``ndim == 2``; we prioritize the
            # per-axis interpretation over the shared pair handled below.
            return tuple((int(val), int(val)) for val in x)
        if x.shape[0] == 2:
            # Two-element vector → shared (before, after) pair
            return tuple((int(x[0]), int(x[1])) for _ in range(ndim))
        raise ValueError("sequence argument must be of length 1, 2, or ndim")
    if x.ndim == 2:
        if x.shape[0] == ndim:
            # Explicit per-axis pairs
            return tuple((int(x[i, 0]), int(x[i, 1])) for i in range(ndim))
        if x.shape[0] == 1 and x.shape[1] == 2:
            # Single (before, after) row broadcast to all axes
            return tuple((int(x[0, 0]), int(x[0, 1])) for _ in range(ndim))
        raise ValueError(
            "sequence argument must have shape (ndim, 2) or (1, 2)"
        )

    raise ValueError("sequence argument must be 1- or 2-dimensional")


def _infer_constant_shape(
    values: ndarray, ndim: int
) -> tuple[int, int, ndarray]:
    """Normalize ``constant_values`` into a shape descriptor for C++.

    The return value is ``(rows, cols, flattened)`` where ``flattened`` is a
    1-D view of the original data, ``rows`` is either ``ndim`` or ``1``, and
    ``cols`` is ``1`` or ``2`` depending on whether separate before/after
    values are provided. The C++ kernels use this metadata to reconstruct the
    correct constant for each axis without doing any additional reshaping.
    """

    if values.ndim == 0:
        return 1, 1, values.reshape((1,))

    if values.ndim == 1:
        length = values.shape[0]
        if length == 1:
            return 1, 1, values.reshape((1,))
        if length == 2:
            # One before/after pair shared by all axes
            return 1, 2, values
        if length == ndim:
            # Per-axis values, identical on both sides
            return ndim, 1, values
        raise ValueError(
            "sequence argument must have length 1, 2, or match array.ndim"
        )

    if values.ndim == 2:
        rows, cols = values.shape
        if rows == 1 and cols == 1:
            # Flatten to 1-D so we can treat it like a scalar downstream
            return 1, 1, values.reshape((1,))
        if rows == 1 and cols == 2:
            # Collapse the single row into a flat array [before, after]
            return 1, 2, values.reshape((-1,))
        if rows == ndim and cols in (1, 2):
            # Flatten to contiguous storage; offset is derived from cols
            return rows, cols, values.reshape((-1,))
        raise ValueError(
            "constant_values argument must have shape (ndim, 2), (ndim, 1), or (1, 2)"
        )

    raise ValueError("constant_values must be 0, 1, or 2-dimensional")


def _pad_simple(
    array: ndarray, pad_width: tuple[tuple[int, int], ...], fill_value: Any = 0
) -> tuple[ndarray, tuple[slice, ...]]:
    """
    Pad an array by filling with a constant value.
    """
    # Compute the new shape
    new_shape = tuple(
        array.shape[i] + pad_width[i][0] + pad_width[i][1]
        for i in range(array.ndim)
    )

    # Create output array filled with fill_value
    padded = full(new_shape, fill_value, dtype=array.dtype)

    # Get slices for original area
    original_area_slice = tuple(
        slice(pad_width[i][0], pad_width[i][0] + array.shape[i])
        for i in range(array.ndim)
    )

    # Copy original array into the padded array
    padded[original_area_slice] = array

    return padded, original_area_slice


def _set_pad_area(
    roi: ndarray,
    axis: int,
    width_pair: tuple[int, int],
    value_pair: tuple[Any, Any],
) -> None:
    """
    Set pad values for a region of interest along an axis.
    """
    left_slice: list[Any] = [slice(None)] * roi.ndim
    left_slice[axis] = slice(0, width_pair[0])
    roi[tuple(left_slice)] = value_pair[0]

    if width_pair[1] > 0:
        right_slice: list[Any] = [slice(None)] * roi.ndim
        right_slice[axis] = slice(-width_pair[1], None)
        roi[tuple(right_slice)] = value_pair[1]


def _get_linear_ramps(
    padded: ndarray,
    axis: int,
    width_pair: tuple[int, int],
    end_value_pair: tuple[Any, Any],
    pad_width: tuple[tuple[int, int], ...],
) -> tuple[Any, Any]:
    """
    Construct linear ramps for linear_ramp mode.
    """
    # Calculate edge indices in padded array
    left_index = pad_width[axis][0]
    right_index = padded.shape[axis] - pad_width[axis][1] - 1

    # Get edge values from center region
    left_slice: list[Any] = [slice(None)] * padded.ndim
    left_slice[axis] = left_index
    left_start = padded[tuple(left_slice)]

    right_slice: list[Any] = [slice(None)] * padded.ndim
    right_slice[axis] = right_index
    right_start = padded[tuple(right_slice)]

    # Left ramp
    left_ramp = None
    if width_pair[0] > 0:
        # Create linear interpolation: walk from user-specified end value
        # towards the edge value inside the array.
        left_end = end_value_pair[0]  # type: ignore[assignment]
        if left_start.size == 1:
            left_start_value: Any = float(left_start.flat[0])
        else:
            left_start_value = left_end
        left_ramp_1d = cast(
            ndarray,
            linspace(
                cast(Any, left_end),
                cast(Any, left_start_value),
                width_pair[0],
                endpoint=False,
            ),
        )
        # Broadcast to match the shape
        shape_for_bc: list[int] = [1] * padded.ndim
        shape_for_bc[axis] = width_pair[0]
        left_ramp_1d = left_ramp_1d.reshape(shape_for_bc)
        left_ramp = left_ramp_1d

    # Right ramp
    right_ramp = None
    if width_pair[1] > 0:
        right_end = cast(Any, end_value_pair[1])
        if right_start.size == 1:
            right_start_value: Any = float(right_start.flat[0])
        else:
            right_start_value = right_end
        # Create linear space from edge value to end value
        right_ramp_1d = cast(
            ndarray,
            linspace(
                cast(Any, right_start_value),
                cast(Any, right_end),
                width_pair[1] + 1,
            ),
        )[1:]
        # Broadcast to match the shape
        shape_for_bc = [1] * padded.ndim
        shape_for_bc[axis] = width_pair[1]
        right_ramp_1d = right_ramp_1d.reshape(shape_for_bc)
        right_ramp = right_ramp_1d

    return left_ramp, right_ramp


def _get_stats(
    padded: ndarray,
    axis: int,
    width_pair: tuple[int, int],
    length_pair: tuple[int, int],
    stat_func: Callable[..., Any],
    pad_width: tuple[tuple[int, int], ...],
) -> tuple[Any, Any]:
    """
    Calculate statistic for stat modes (mean, maximum, minimum).
    """
    # Calculate slices in padded array
    left_slice = [slice(None)] * padded.ndim
    left_start = pad_width[axis][0]
    left_end = min(pad_width[axis][0] + length_pair[0], padded.shape[axis])
    left_slice[axis] = slice(left_start, left_end)

    # Only compute stat if we have data
    left_region = padded[tuple(left_slice)]
    if left_region.shape[axis] > 0:
        left_stat = stat_func(left_region, axis)
    else:
        # Empty region, use zero
        stat_shape = list(left_region.shape)
        stat_shape[axis] = 1
        left_stat = zeros(tuple(stat_shape), dtype=padded.dtype)

    right_slice = [slice(None)] * padded.ndim
    right_start = max(
        0, padded.shape[axis] - pad_width[axis][1] - length_pair[1]
    )
    right_end = padded.shape[axis] - pad_width[axis][1]
    right_slice[axis] = slice(right_start, right_end)

    # Only compute stat if we have data
    right_region = padded[tuple(right_slice)]
    if right_region.shape[axis] > 0:
        right_stat = stat_func(right_region, axis)
    else:
        # Empty region, use zero
        stat_shape = list(right_region.shape)
        stat_shape[axis] = 1
        right_stat = zeros(tuple(stat_shape), dtype=padded.dtype)

    return left_stat, right_stat


def _set_reflect_both(
    padded: ndarray,
    axis: int,
    width_pair: tuple[int, int],
    method: str,
    include_edge: bool,
    pad_width: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """
    Set reflected values for both sides of an axis.
    """
    left_pad, right_pad = width_pair

    # Calculate the CURRENT filled region bounds in padded array
    # Initially this is the center (original input), but expands with each iteration
    # The filled region starts at original center and grows as we reflect
    original_center_lo = pad_width[axis][0]
    original_center_hi = padded.shape[axis] - pad_width[axis][1] - 1

    # Current filled bounds (what we can reflect from)
    # On first iteration: filled_lo = center_lo, filled_hi = center_hi
    # On subsequent iterations: includes previously reflected regions
    filled_lo = original_center_lo - (pad_width[axis][0] - left_pad)
    filled_hi = original_center_hi + (pad_width[axis][1] - right_pad)
    filled_length = filled_hi - filled_lo + 1

    # Calculate how much we can reflect in this iteration
    if include_edge:
        # Symmetric mode - can reflect the full filled region
        max_reflect = filled_length
    else:
        # Reflect mode - need at least 1 value, exclude edge
        max_reflect = filled_length - 1 if filled_length > 1 else 0

    # Left reflection
    left_chunk = min(left_pad, max_reflect)
    if left_chunk > 0 and filled_lo + left_chunk <= padded.shape[axis]:
        # Source slice for reflection from filled region
        src_slice = [slice(None)] * padded.ndim
        if include_edge:
            src_slice[axis] = slice(filled_lo, filled_lo + left_chunk)
        else:
            src_slice[axis] = slice(
                filled_lo + 1,
                min(filled_lo + 1 + left_chunk, padded.shape[axis]),
            )

        # Get the data and flip it
        src_data = padded[tuple(src_slice)]
        if src_data.shape[axis] > 0:  # Only flip if we have data
            src_data = flip(src_data, axis=axis)
            if method == "odd":
                edge_slice = [slice(None)] * padded.ndim
                edge_slice[axis] = slice(filled_lo, filled_lo + 1)
                edge_vals = padded[tuple(edge_slice)]
                src_data = 2 * edge_vals - src_data

            # Destination slice
            dest_slice = [slice(None)] * padded.ndim
            dest_slice[axis] = slice(max(0, filled_lo - left_chunk), filled_lo)
            padded[tuple(dest_slice)] = src_data

    # Right reflection
    right_chunk = min(right_pad, max_reflect)
    if right_chunk > 0 and filled_hi >= 0:
        # Source slice for reflection from filled region
        src_slice = [slice(None)] * padded.ndim
        if include_edge:
            # Symmetric: include the edge, take from end
            src_slice[axis] = slice(
                max(0, filled_hi - right_chunk + 1), filled_hi + 1
            )
        else:
            # Reflect: exclude the edge, take one less from end
            src_slice[axis] = slice(max(0, filled_hi - right_chunk), filled_hi)

        # Get the data and flip it
        src_data = padded[tuple(src_slice)]
        if src_data.shape[axis] > 0:  # Only flip if we have data
            src_data = flip(src_data, axis=axis)
            if method == "odd":
                edge_slice = [slice(None)] * padded.ndim
                edge_slice[axis] = slice(filled_hi, filled_hi + 1)
                edge_vals = padded[tuple(edge_slice)]
                src_data = 2 * edge_vals - src_data

            # Destination slice
            dest_slice = [slice(None)] * padded.ndim
            dest_slice[axis] = slice(
                filled_hi + 1,
                min(filled_hi + 1 + right_chunk, padded.shape[axis]),
            )
            padded[tuple(dest_slice)] = src_data

    return left_pad - left_chunk, right_pad - right_chunk


def _set_wrap_both(
    padded: ndarray,
    axis: int,
    width_pair: tuple[int, int],
    pad_width: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    """
    Set wrapped values for both sides of an axis.
    """
    left_pad, right_pad = width_pair

    # Calculate the CURRENT filled region bounds (same as reflect)
    original_center_lo = pad_width[axis][0]
    original_center_hi = padded.shape[axis] - pad_width[axis][1] - 1

    # Current filled bounds expand with each iteration
    filled_lo = original_center_lo - (pad_width[axis][0] - left_pad)
    filled_hi = original_center_hi + (pad_width[axis][1] - right_pad)
    filled_length = filled_hi - filled_lo + 1

    # Calculate how much we can wrap in this iteration
    if filled_length == 0:
        # Can't wrap from empty center
        return 0, 0
    left_chunk = min(left_pad, filled_length)
    right_chunk = min(right_pad, filled_length)

    # Left wrap
    if left_chunk > 0:
        # Source: take from the right side of filled region
        src_slice = [slice(None)] * padded.ndim
        src_slice[axis] = slice(filled_hi - left_chunk + 1, filled_hi + 1)
        src_data = padded[tuple(src_slice)]

        # Destination: left padding area
        dest_slice = [slice(None)] * padded.ndim
        dest_slice[axis] = slice(filled_lo - left_chunk, filled_lo)
        padded[tuple(dest_slice)] = src_data

    # Right wrap
    if right_chunk > 0:
        # Source: take from the left side of filled region
        src_slice = [slice(None)] * padded.ndim
        src_slice[axis] = slice(filled_lo, filled_lo + right_chunk)
        src_data = padded[tuple(src_slice)]

        # Destination: right padding area
        dest_slice = [slice(None)] * padded.ndim
        dest_slice[axis] = slice(filled_hi + 1, filled_hi + 1 + right_chunk)
        padded[tuple(dest_slice)] = src_data

    return left_pad - left_chunk, right_pad - right_chunk


def _pad_cpp(
    array: ndarray,
    pad_width_tuple: tuple[tuple[int, int], ...],
    mode: str,
    constant_value: ndarray,
    constant_rows: int,
    constant_cols: int,
) -> ndarray:
    """Use C++ task for simple padding modes."""
    # Calculate output shape
    out_shape = tuple(
        array.shape[dim] + pad_width_tuple[dim][0] + pad_width_tuple[dim][1]
        for dim in range(array.ndim)
    )

    center_slice = tuple(
        slice(
            pad_width_tuple[dim][0], pad_width_tuple[dim][0] + array.shape[dim]
        )
        for dim in range(array.ndim)
    )

    if mode == "constant" and constant_value.size == 1:
        fill_scalar = convert_to_cupynumeric_ndarray(constant_value)
        output = full(out_shape, fill_scalar, dtype=array.dtype)
        output[center_slice] = array
        return output

    # Create output array filled with zeros (handles padding regions)
    output = zeros(out_shape, dtype=array.dtype)

    # Copy input to center region
    output[center_slice] = array

    # Call C++ task to fill only the padding regions based on mode
    # Assumes partitions always have access to center region for edge replication
    output._thunk.pad(
        pad_width_tuple,
        mode,
        constant_value._thunk,
        constant_rows,
        constant_cols,
    )
    return output


def _pad_python(
    array: ndarray,
    pad_width_tuple: tuple[tuple[int, int], ...],
    mode: str | Callable[..., Any],
    **kwargs: Any,
) -> ndarray:
    """Pure Python implementation for complex modes."""
    if callable(mode):
        # Use function-based padding with numpy.apply_along_axis
        function = mode
        # Create a new zero padded array
        padded, _ = _pad_simple(array, pad_width_tuple, fill_value=0)

        # Apply function along each axis
        for axis in range(padded.ndim):
            # Use numpy's apply_along_axis approach
            # Note: This is a simplified implementation
            # For full compatibility, we'd need to support the full API
            moved = moveaxis(padded, (axis,), (-1,))
            for base_index in np.ndindex(moved.shape[:-1]):
                function(
                    moved[base_index + (Ellipsis,)],
                    pad_width_tuple[axis],
                    axis,
                    kwargs,
                )
            padded = moveaxis(moved, (-1,), (axis,))

        return padded

    # Create array with final shape and original values (padded area is undefined)
    padded, original_area_slice = _pad_simple(array, pad_width_tuple)

    # Prepare for iteration over all dimensions
    axes = range(padded.ndim)

    # Check empty arrays first (before mode-specific processing)
    if array.size == 0:
        # Only modes 'constant' and 'empty' can extend empty axes, all other
        # modes depend on `array` not being empty
        # -> ensure every empty axis is only 'padded with 0'
        for axis, width_pair in zip(axes, pad_width_tuple):
            if array.shape[axis] == 0 and any(width_pair):
                raise ValueError(
                    f"can't extend empty axis {axis} using modes other than "
                    "'constant' or 'empty'"
                )
        # passed, don't need to do anything more as _pad_simple already
        # returned the correct result
        return padded

    elif mode == "linear_ramp":
        end_values = kwargs.get("end_values", 0)
        end_values_pairs = _as_pairs(end_values, padded.ndim)
        for axis, width_pair, value_pair in zip(
            axes, pad_width_tuple, end_values_pairs
        ):
            ramp_pair = _get_linear_ramps(
                padded, axis, width_pair, value_pair, pad_width_tuple
            )
            _set_pad_area(padded, axis, width_pair, ramp_pair)

    elif mode in _STAT_FUNCTIONS_AXIS:
        func = _STAT_FUNCTIONS_AXIS[mode]
        length = kwargs.get("stat_length", None)
        # Default stat_length is the full length of each axis
        if length is None:
            length = tuple(array.shape[i] for i in range(array.ndim))
        length_pairs = _as_pairs(length, padded.ndim, as_index=True)
        for axis, width_pair, length_pair in zip(
            axes, pad_width_tuple, length_pairs
        ):
            stat_pair = _get_stats(
                padded, axis, width_pair, length_pair, func, pad_width_tuple
            )
            _set_pad_area(padded, axis, width_pair, stat_pair)

    elif mode in {"reflect", "symmetric"}:
        method = kwargs.get("reflect_type", "even")
        if method not in {"even", "odd"}:
            raise ValueError("reflect_type must be 'even' or 'odd'")
        include_edge = True if mode == "symmetric" else False

        for axis, pad_pair in zip(axes, pad_width_tuple):
            left_index, right_index = pad_pair
            if array.shape[axis] == 1 and (left_index > 0 or right_index > 0):
                # Extending singleton dimension for 'reflect' is legacy
                # behavior; it really should raise an error.
                # Use edge mode for singleton dimensions
                if left_index > 0:
                    left_slice = [slice(None)] * padded.ndim
                    left_slice[axis] = slice(0, left_index)
                    edge_slice = [slice(None)] * padded.ndim
                    edge_slice[axis] = slice(
                        pad_width_tuple[axis][0], pad_width_tuple[axis][0] + 1
                    )
                    padded[tuple(left_slice)] = padded[tuple(edge_slice)]
                if right_index > 0:
                    right_slice = [slice(None)] * padded.ndim
                    right_slice[axis] = slice(-right_index, None)
                    edge_slice = [slice(None)] * padded.ndim
                    edge_slice[axis] = slice(
                        padded.shape[axis] - pad_width_tuple[axis][1] - 1,
                        padded.shape[axis] - pad_width_tuple[axis][1],
                    )
                    padded[tuple(right_slice)] = padded[tuple(edge_slice)]
                continue

            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with reflected
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_reflect_both(
                    padded,
                    axis,
                    (left_index, right_index),
                    method,
                    include_edge,
                    pad_width_tuple,
                )

    elif mode == "wrap":
        for axis, (left_index, right_index) in zip(axes, pad_width_tuple):
            while left_index > 0 or right_index > 0:
                # Iteratively pad until dimension is filled with wrapped
                # values. This is necessary if the pad area is larger than
                # the length of the original values in the current dimension.
                left_index, right_index = _set_wrap_both(
                    padded, axis, (left_index, right_index), pad_width_tuple
                )

    return padded


@add_boilerplate("array")
def pad(
    array: ndarray,
    pad_width: int | Sequence[int] | Sequence[tuple[int, int]],
    mode: str | Callable[..., Any] = "constant",
    **kwargs: Any,
) -> ndarray:
    """
    Pad an array.

    Parameters
    ----------
    array : array_like
        The array to pad.
    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ((before_1, after_1), ... (before_N, after_N)) unique pad widths
        for each axis.
        ((before, after),) yields same before and after pad for each axis.
        (pad,) or int is a shortcut for before = after = pad width for all
        axes.
    mode : str or function, optional
        One of the following string values or a user supplied function.

        'constant' (default)
            Pads with a constant value.
        'edge'
            Pads with the edge values of array.
        'linear_ramp'
            Pads with the linear ramp between end_value and the array edge
            value.
        'maximum'
            Pads with the maximum value of all or part of the vector along
            each axis.
        'mean'
            Pads with the mean value of all or part of the vector along
            each axis.
        'minimum'
            Pads with the minimum value of all or part of the vector along
            each axis.
        'reflect'
            Pads with the reflection of the vector mirrored on the first and
            last values of the vector along each axis.
        'symmetric'
            Pads with the reflection of the vector mirrored along the edge of
            the array.
        'wrap'
            Pads with the wrap of the vector along the axis. The first values
            are used to pad the end and the end values are used to pad the
            beginning.
        'empty'
            Pads with undefined values.

        <function>
            Padding function, see Notes.

    stat_length : sequence or int, optional
        Used in 'maximum', 'mean', 'minimum'.  Number of values at edge of
        each axis used to calculate the statistic value.

        ((before_1, after_1), ... (before_N, after_N)) unique statistic
        lengths for each axis.

        ((before, after),) yields same before and after statistic lengths
        for each axis.

        (stat_length,) or int is a shortcut for before = after = statistic
        length for all axes.

        Default is ``None``, to use the entire axis.

    constant_values : sequence or scalar, optional
        Used in 'constant'.  The values to set the padded values for each
        axis.

        ((before_1, after_1), ... (before_N, after_N)) unique pad constants
        for each axis.

        ((before, after),) yields same before and after constants for each
        axis.

        (constant,) or constant is a shortcut for before = after = constant
        for all axes.

        Default is 0.

    end_values : sequence or scalar, optional
        Used in 'linear_ramp'.  The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.

        ((before_1, after_1), ... (before_N, after_N)) unique end values
        for each axis.

        ((before, after),) yields same before and after end values for each
        axis.

        (constant,) or constant is a shortcut for before = after = constant
        for all axes.

        Default is 0.

    reflect_type : {'even', 'odd'}, optional
        Used in 'reflect', and 'symmetric'.  The 'even' style is the
        default with an unaltered reflection around the edge value.  For
        the 'odd' style, the extended part of the array is created by
        subtracting the reflected values from two times the edge value.

    Returns
    -------
    pad : ndarray
        Padded array of rank equal to `array` with shape increased
        according to `pad_width`.

    Notes
    -----
    This is a simplified implementation that supports the most common modes.
    Some advanced features may not be fully supported.

    See Also
    --------
    numpy.pad

    Availability
    ------------
    Multiple GPUs, Multiple CPUs
    """
    # Normalize pad_width to the standard format
    if isinstance(pad_width, numbers.Integral):
        pad_width_tuple = tuple(
            (int(pad_width), int(pad_width)) for _ in range(array.ndim)
        )
    else:
        pad_width_array = convert_to_cupynumeric_ndarray(pad_width)

        if not pad_width_array.dtype.kind == "i":
            raise TypeError("`pad_width` must be of integral type.")

        # Broadcast to shape (array.ndim, 2)
        pad_width_pairs = _as_pairs(pad_width_array, array.ndim, as_index=True)
        pad_width_tuple = pad_width_pairs

    # Determine if we can use C++ task for this mode
    CPP_MODES = {"constant", "edge"}
    STAT_MODES = {"mean", "maximum", "minimum", "median"}
    SUPPORTED_STRING_MODES = (
        CPP_MODES
        | STAT_MODES
        | {"wrap", "linear_ramp", "reflect", "symmetric", "empty"}
    )

    if isinstance(mode, str) and mode not in SUPPORTED_STRING_MODES:
        raise ValueError(f"mode '{mode}' is not supported")

    # Convert empty mode to constant with value 0 (they're equivalent)
    constant_value: ndarray | int | float = 0
    constant_rows = 0
    constant_cols = 0
    converted_from_stat = False

    if isinstance(mode, str) and mode == "empty":
        mode = "constant"
        constant_value = 0
        converted_from_stat = True  # Mark as converted so it uses C++
        # Will fall through to CPP_MODES check

    # Check statistical modes - can convert to constant mode for C++ (1D only for now)
    if isinstance(mode, str) and mode in STAT_MODES:
        if "stat_length" not in kwargs:
            # Statistical modes: pre-compute statistics and use constant mode in C++
            # This optimization only works for 1D arrays (global stat)
            # Multi-dimensional arrays need per-axis stats, so use Python fallback
            if array.ndim == 1:
                # Compute the statistic value using GPU-accelerated cupynumeric operations
                stat_func = _STAT_FUNCTIONS[mode]

                # Compute global stat (ignoring stat_length for 1D optimization)
                # TODO: Could use stat_length to compute from edge regions only
                stat_value_array = cast(
                    Callable[[ndarray], ndarray], stat_func
                )(array)

                # Keep as deferred array - will pass to C++ task as broadcast input
                # This avoids blocking evaluation
                constant_value = (
                    stat_value_array  # Pass as ndarray, not scalar
                )
                mode = "constant"  # Use constant mode in C++
                converted_from_stat = True
                # Will fall through to CPP_MODES check below

    if isinstance(mode, str):
        if mode == "constant":
            allowed_kwargs = {"constant_values"}
        elif mode == "edge":
            allowed_kwargs = set()
        elif mode in STAT_MODES:
            allowed_kwargs = {"stat_length"}
        elif mode == "linear_ramp":
            allowed_kwargs = {"end_values"}
        elif mode in {"reflect", "symmetric"}:
            allowed_kwargs = {"reflect_type"}
        elif mode == "wrap":
            allowed_kwargs = set()
        else:
            allowed_kwargs = set()

        unsupported_kwargs = set(kwargs) - allowed_kwargs
        if unsupported_kwargs:
            raise ValueError(
                f"unsupported keyword arguments for mode '{mode}': {unsupported_kwargs}"
            )

    # Check if mode can be handled by C++
    use_cpp = False
    if isinstance(mode, str) and mode in CPP_MODES:
        if mode == "constant":
            if not converted_from_stat and "constant_values" in kwargs:
                constant_value = kwargs["constant_values"]

            value_array = convert_to_cupynumeric_ndarray(constant_value)
            if value_array.dtype != array.dtype:
                value_array = value_array.astype(array.dtype, copy=False)

            constant_rows, constant_cols, value_array = _infer_constant_shape(
                value_array, array.ndim
            )
            constant_value = value_array
            use_cpp = True
        elif mode == "edge":
            use_cpp = True

    # Check for empty arrays before C++ path
    if use_cpp:
        # Validate that non-constant modes can't pad empty arrays
        if mode != "constant" and array.size == 0:
            for axis, width_pair in zip(range(array.ndim), pad_width_tuple):
                if array.shape[axis] == 0 and any(width_pair):
                    raise ValueError(
                        f"can't extend empty axis {axis} using modes other than "
                        "'constant' or 'empty'"
                    )

        if mode != "constant":
            constant_value = zeros((1,), dtype=array.dtype)

        return _pad_cpp(
            array,
            pad_width_tuple,
            cast(str, mode),
            cast(ndarray, constant_value),
            constant_rows,
            constant_cols,
        )

    # Use Python implementation (works correctly with all configurations)
    return _pad_python(array, pad_width_tuple, mode, **kwargs)
