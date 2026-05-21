# Copyright 2026 NVIDIA Corporation
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

from typing import TYPE_CHECKING, Any

from ._array.array import ndarray
from ._array.util import add_boilerplate, check_writeable
from .config import NdimageConvolveModeCode

if TYPE_CHECKING:
    from .types import NdimageConvolveMode

_MODE_MAP: dict[str, NdimageConvolveModeCode] = {
    "reflect": NdimageConvolveModeCode.REFLECT,
    "constant": NdimageConvolveModeCode.CONSTANT,
    "nearest": NdimageConvolveModeCode.NEAREST,
    "mirror": NdimageConvolveModeCode.MIRROR,
    "wrap": NdimageConvolveModeCode.WRAP,
    "grid-mirror": NdimageConvolveModeCode.REFLECT,
    "grid-constant": NdimageConvolveModeCode.CONSTANT,
    "grid-wrap": NdimageConvolveModeCode.WRAP,
}


def _normalize_origin(
    origin: int | tuple[int, ...], ndim: int, weights_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Normalize and validate the convolution origin for each input dimension.

    A scalar ``origin`` applies the same offset to all dimensions. A tuple
    origin must provide exactly one offset per input dimension. Each normalized
    offset is checked against the corresponding filter extent so that the
    shifted kernel center remains inside the filter.

    Parameters
    ----------
    origin : int or tuple[int, ...]
        User-provided origin offset or offsets.
    ndim : int
        Number of dimensions in the input array.
    weights_shape : tuple[int, ...]
        Shape of the convolution weights.

    Returns
    -------
    tuple[int, ...]
        One validated origin value per input dimension.
    """
    if isinstance(origin, tuple):
        origin = tuple(int(o) for o in origin)
        if len(origin) != ndim:
            raise ValueError(
                "`origin` must have one entry per input dimension"
            )
    else:
        origin = (int(origin),) * ndim

    for axis, (current_origin, filter_size) in enumerate(
        zip(origin, weights_shape, strict=True)
    ):
        center = filter_size // 2
        if (
            center + current_origin < 0
            or center + current_origin >= filter_size
        ):
            raise ValueError(
                f"invalid origin {current_origin} for axis {axis} with filter size {filter_size}"
            )

    return origin


@add_boilerplate("input", "weights")
def convolve(
    input: ndarray,
    weights: ndarray,
    output: ndarray | None = None,
    mode: NdimageConvolveMode = "reflect",
    cval: Any = 0.0,
    origin: int | tuple[int, ...] = 0,
    *,
    axes: tuple[int, ...] | None = None,
) -> ndarray:
    """
    Multidimensional convolution.

    Performs a single convolution by sliding ``weights`` over ``input`` and
    writing one output value for each input element.

    Parameters
    ----------
    input : array_like
        Input array to convolve. Must have at least one dimension and the same
        number of dimensions as ``weights``.
    weights : array_like
        Convolution kernel. Must have at least one dimension. If its dtype does
        not match ``input``, it is cast to ``input.dtype`` before execution.
    output : ndarray, numpy.ndarray, dtype, or None, optional
        Destination for the result. If an array is provided, it must be
        writable and have the same shape and dtype as ``input``. If ``None``,
        a new array with the same shape and dtype as ``input`` is returned.
    mode : {"reflect", "constant", "nearest", "mirror", "wrap", \
            "grid-mirror", "grid-constant", "grid-wrap"}, optional
        Boundary handling mode used when the kernel overlaps points outside
        the input domain. The ``grid-*`` aliases follow SciPy's naming.
    cval : scalar, optional
        Fill value used for points outside ``input`` when ``mode`` is
        ``"constant"`` or ``"grid-constant"``.
    origin : int or tuple[int, ...], optional
        Placement of the kernel center relative to each input element. A scalar
        applies to every axis; a tuple must contain one value per input
        dimension.
    axes : tuple[int, ...] or None, optional
        Axis subset to convolve over. This argument is accepted for API
        compatibility but is not currently implemented.

    Returns
    -------
    ndarray
        The convolution result. This is ``output`` when an output array is
        provided; otherwise it is a newly allocated array.

    Availability
    ------------
    Single GPU, Multi GPU

    See Also
    --------
    scipy.ndimage.convolve
    """
    if axes is not None:
        raise NotImplementedError(
            "`axes` is not yet supported for cupynumeric.ndimage.convolve"
        )
    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be one of {set(_MODE_MAP.keys())}")
    if input.ndim == 0 or weights.ndim == 0:
        raise ValueError("input and weights must be at least 1-D")
    if input.ndim != weights.ndim:
        raise ValueError("input and weights must have the same dimensions")

    origin = _normalize_origin(origin, input.ndim, weights.shape)

    # check output array if given, otherwise, create a new array
    if output is not None:
        check_writeable(output)
        if output.shape != input.shape:
            raise ValueError(
                f"output array shape {output.shape} does not match input array shape {input.shape}"
            )
        if output.dtype != input.dtype:
            raise ValueError(
                f"output array dtype {output.dtype} does not match input array dtype {input.dtype}"
            )
    else:
        output = ndarray._from_inputs(shape=input.shape, dtype=input.dtype)

    if input.dtype != weights.dtype:
        weights = weights.astype(input.dtype)

    output._thunk.ndimage_convolve(
        input._thunk, weights._thunk, _MODE_MAP[mode], cval, origin
    )

    return output


@add_boilerplate("input", "weights")
def batched_convolve(
    input: ndarray,
    weights: ndarray,
    output: ndarray | None = None,
    mode: NdimageConvolveMode = "reflect",
    cval: Any = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> ndarray:
    """
    Batched multidimensional convolution.

    Treats axis 0 of ``input`` as an image batch and axis 0 of ``weights`` as
    an independent filter batch. The remaining axes are spatial convolution
    axes. If ``input`` has shape ``(N, *spatial_shape)`` and ``weights`` has
    shape ``(M, *filter_shape)``, then the result has shape
    ``(N, M, *spatial_shape)``. Each output slice ``result[n, m]`` is the
    convolution of ``input[n]`` with ``weights[m]`` using the same boundary and
    origin semantics as :func:`convolve`.

    Parameters
    ----------
    input : array_like
        Batched input array. Must have at least two dimensions: one image batch
        dimension followed by one or more spatial dimensions.
    weights : array_like
        Batched convolution filters. Must have the same number of dimensions as
        ``input``. Axis 0 indexes filters; the remaining axes define the filter
        extent for the corresponding input spatial axes. If its dtype does not
        match ``input``, it is cast to ``input.dtype`` before execution.
    output : ndarray, numpy.ndarray, dtype, or None, optional
        Destination for the result. If an array is provided, it must be
        writable and have shape ``(input.shape[0], weights.shape[0],
        *input.shape[1:])`` and dtype ``input.dtype``. If ``None``, a new array
        with that shape and dtype is returned.
    mode : {"reflect", "constant", "nearest", "mirror", "wrap", \
            "grid-mirror", "grid-constant", "grid-wrap"}, optional
        Boundary handling mode used when a filter overlaps points outside the
        input spatial domain.
    cval : scalar, optional
        Fill value used for points outside ``input`` when ``mode`` is
        ``"constant"`` or ``"grid-constant"``.
    origin : int or tuple[int, ...], optional
        Placement of each filter center relative to each input element. A scalar
        applies to every spatial axis; a tuple must contain one value per
        spatial axis and does not include the batch axis. All filters will have the
        same origin placement.

    Returns
    -------
    ndarray
        The batched convolution result with shape ``(N, M, *spatial_shape)``.
        This is ``output`` when an output array is provided; otherwise it is a
        newly allocated array.

    Availability
    ------------
    Single GPU, Multi GPU
    """
    if mode not in _MODE_MAP:
        raise ValueError(f"mode must be one of {set(_MODE_MAP.keys())}")
    if input.ndim < 2 or weights.ndim < 2:
        raise ValueError(
            "input and weights must include batch and spatial dimensions"
        )
    if input.ndim != weights.ndim:
        raise ValueError("input and weights must have the same dimensions")

    origins = _normalize_origin(origin, input.ndim - 1, weights.shape[1:])
    output_shape = (input.shape[0], weights.shape[0], *input.shape[1:])

    if output is not None:
        check_writeable(output)
        if output.shape != output_shape:
            raise ValueError(
                f"output array shape {output.shape} does not match expected shape {output_shape}"
            )
        if output.dtype != input.dtype:
            raise ValueError(
                f"output array dtype {output.dtype} does not match input array dtype {input.dtype}"
            )
    else:
        output = ndarray._from_inputs(shape=output_shape, dtype=input.dtype)

    if input.dtype != weights.dtype:
        weights = weights.astype(input.dtype)

    output._thunk.ndimage_batched_convolve(
        input._thunk, weights._thunk, _MODE_MAP[mode], cval, origins
    )

    return output
