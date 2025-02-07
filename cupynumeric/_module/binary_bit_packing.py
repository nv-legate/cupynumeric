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
from .creation_shape import empty

if TYPE_CHECKING:
    from .._array.array import ndarray
    from ..types import BitOrder


def _sanitize_arguments(
    a: ndarray, axis: int | None, bitorder: BitOrder
) -> tuple[ndarray, int]:
    if axis is None:
        if a.ndim > 1:
            a = a.ravel()
        sanitized_axis = 0
    elif axis < 0:
        sanitized_axis = axis + a.ndim
    else:
        sanitized_axis = axis

    if sanitized_axis < 0 or sanitized_axis >= a.ndim:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {a.ndim}"
        )

    if bitorder not in ("big", "little"):
        raise ValueError("'order' must be either 'little' or 'big'")

    return a, sanitized_axis


@add_boilerplate("a")
def packbits(
    a: ndarray, axis: int | None = None, bitorder: BitOrder = "big"
) -> ndarray:
    """

    Packs the elements of a binary-valued array into bits in a uint8 array.

    The result is padded to full bytes by inserting zero bits at the end.

    Parameters
    ----------
    a : array_like
        An array of integers or booleans whose elements should be packed to
        bits.
    axis : int, optional
        The dimension over which bit-packing is done.
        ``None`` implies packing the flattened array.
    bitorder : ``{"big", "little"}``, optional
        The order of the input bits. 'big' will mimic bin(val),
        ``[0, 0, 0, 0, 0, 0, 1, 1] => 3 = 0b00000011``, 'little' will
        reverse the order so ``[1, 1, 0, 0, 0, 0, 0, 0] => 3``.
        Defaults to "big".

    Returns
    -------
    packed : ndarray
        Array of type uint8 whose elements represent bits corresponding to the
        logical (0 or nonzero) value of the input elements. The shape of
        `packed` has the same number of dimensions as the input (unless `axis`
        is None, in which case the output is 1-D).

    See Also
    --------
    numpy.packbits

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if a.dtype.kind not in ("u", "i", "b"):
        raise TypeError(
            "Expected an input array of integer or boolean data type"
        )

    a, sanitized_axis = _sanitize_arguments(a, axis, bitorder)

    out_shape = tuple(
        (extent + 7) // 8 if dim == sanitized_axis else extent
        for dim, extent in enumerate(a.shape)
    )
    out = empty(out_shape, dtype="B")
    out._thunk.packbits(a._thunk, sanitized_axis, bitorder)

    return out


@add_boilerplate("a")
def unpackbits(
    a: ndarray,
    axis: int | None = None,
    count: int | None = None,
    bitorder: BitOrder = "big",
) -> ndarray:
    """
    Unpacks elements of a uint8 array into a binary-valued output array.

    Each element of `a` represents a bit-field that should be unpacked
    into a binary-valued output array. The shape of the output array is
    either 1-D (if `axis` is ``None``) or the same shape as the input
    array with unpacking done along the axis specified.

    Parameters
    ----------
    a : ndarray[uint8]
       Input array.
    axis : int, optional
        The dimension over which bit-unpacking is done.
        ``None`` implies unpacking the flattened array.
    count : int or None, optional
        The number of elements to unpack along `axis`, provided as a way
        of undoing the effect of packing a size that is not a multiple
        of eight. A non-negative number means to only unpack `count`
        bits. A negative number means to trim off that many bits from
        the end. ``None`` means to unpack the entire array (the
        default). Counts larger than the available number of bits will
        add zero padding to the output. Negative counts must not
        exceed the available number of bits.

    bitorder : ``{"big", "little"}``, optional
        The order of the returned bits. 'big' will mimic bin(val),
        ``3 = 0b00000011 => [0, 0, 0, 0, 0, 0, 1, 1]``, 'little' will reverse
        the order to ``[1, 1, 0, 0, 0, 0, 0, 0]``.
        Defaults to 'big'.

    Returns
    -------
    unpacked : ndarray[uint8]
       The elements are binary-valued (0 or 1).

    See Also
    --------
    numpy.unpackbits

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if a.dtype != "B":
        raise TypeError("Expected an input array of unsigned byte data type")

    if count is not None and not isinstance(count, int):
        raise TypeError("count must be an integer or None")

    a, sanitized_axis = _sanitize_arguments(a, axis, bitorder)

    axis_extent = a.shape[sanitized_axis] * 8
    out_shape = tuple(
        axis_extent if dim == sanitized_axis else extent
        for dim, extent in enumerate(a.shape)
    )
    out = empty(out_shape, dtype="B")
    out._thunk.unpackbits(a._thunk, sanitized_axis, bitorder)

    if count is not None and count != axis_extent:
        # TODO: We could handle `count` directly in the unpack task to avoid
        # generating the full unpack result. For now we simply slice out the
        # output for the count.
        slices = tuple(
            slice(None, count if dim == sanitized_axis else None)
            for dim in range(a.ndim)
        )
        out = out[slices]
    return out
