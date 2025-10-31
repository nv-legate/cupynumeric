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

from typing import TYPE_CHECKING, Any

import numpy as np
from legate.core import Scalar

from ..config import (
    BinaryOpCode,
    ConvertCode,
    ScanCode,
    UnaryOpCode,
    UnaryRedCode,
)
from ..lib.array_utils import normalize_axis_index, normalize_axis_tuple
from ..types import NdShape
from .util import (
    broadcast_where,
    convert_to_cupynumeric_ndarray,
    find_common_type,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from .._thunk.thunk import NumPyThunk
    from .array import ndarray


def get_where_thunk(
    where: ndarray | None, out_shape: NdShape
) -> NumPyThunk | None:
    from .array import ndarray

    if where is None:
        return where
    if (
        not isinstance(where, ndarray)
        or where.dtype != bool
        or where.shape != out_shape
    ):
        raise RuntimeError("should have converted this earlier")
    return where._thunk


def perform_unary_op(
    op: UnaryOpCode,
    src: ndarray,
    out: ndarray | None = None,
    extra_args: Any = None,
) -> ndarray:
    from .array import ndarray

    if out is not None:
        # If the shapes don't match see if we can broadcast
        # This will raise an exception if they can't be broadcast together
        if np.broadcast_shapes(src.shape, out.shape) != out.shape:
            raise ValueError(
                f"non-broadcastable output operand with shape {out.shape} "
                f"doesn't match the broadcast shape {src.shape}"
            )
    else:
        # No output yet, so make one
        out_shape = src.shape

        if op == UnaryOpCode.ANGLE:
            dtype = np.dtype(np.float64)
        elif op == UnaryOpCode.ROUND or src.dtype.kind != "c":
            dtype = src.dtype
        else:
            if src.dtype == np.dtype(np.complex64):
                dtype = np.dtype(np.float32)  # type: ignore
            else:
                dtype = np.dtype(np.float64)

        out = ndarray._from_inputs(shape=out_shape, dtype=dtype, inputs=(src,))

    if out.dtype != src.dtype:
        if (
            op == UnaryOpCode.ABSOLUTE and src.dtype.kind == "c"
        ) or op == UnaryOpCode.ANGLE:
            out._thunk.unary_op(op, src._thunk, True, extra_args)
        else:
            temp = ndarray._from_inputs(
                out.shape, dtype=src.dtype, inputs=(src,)
            )
            temp._thunk.unary_op(op, src._thunk, True, extra_args)
            out._thunk.convert(temp._thunk)
    else:
        out._thunk.unary_op(op, src._thunk, True, extra_args)
    return out


def _need_upcast_for_reduction(op: UnaryRedCode, dtype: np.dtype[Any]) -> bool:
    return op in (UnaryRedCode.SUM, UnaryRedCode.PROD) and dtype.kind in (
        "b",
        "i",
        "u",
    )


_NON_DECOMPOSABLE_OPS: dict[UnaryRedCode, npt.DTypeLike] = {
    UnaryRedCode.ALL: bool,
    UnaryRedCode.ANY: bool,
    UnaryRedCode.ARGMAX: np.int64,
    UnaryRedCode.ARGMIN: np.int64,
    UnaryRedCode.CONTAINS: bool,
    UnaryRedCode.COUNT_NONZERO: np.uint64,
    UnaryRedCode.NANARGMAX: np.int64,
    UnaryRedCode.NANARGMIN: np.int64,
}


def perform_unary_reduction(
    op: UnaryRedCode,
    src: ndarray,
    axis: Any = None,
    dtype: np.dtype[Any] | None = None,
    out: ndarray | None = None,
    keepdims: bool = False,
    args: tuple[Scalar, ...] = (),
    initial: int | float | None = None,
    where: ndarray | None = None,
) -> ndarray:
    from .array import ndarray

    if axis is None:
        axes = tuple(range(src.ndim))
    else:
        axes = normalize_axis_tuple(axis, src.ndim)

    if 1 < len(axes) < src.ndim and op in _NON_DECOMPOSABLE_OPS:
        raise RuntimeError(
            "multi-axis only supported for decomposable reductions"
        )

    res_dtype: npt.DTypeLike | None

    if op in _NON_DECOMPOSABLE_OPS:
        if dtype is not None:
            raise TypeError(
                f"Cannot override dtype for reduction operator {op.name}"
            )
        dtype = src.dtype
        res_dtype = _NON_DECOMPOSABLE_OPS[op]
    else:
        if dtype is not None:
            # If 'dtype' exists, that determines both the accumulation dtype
            # and the output dtype
            pass
        elif out is not None:
            dtype = out.dtype
        elif _need_upcast_for_reduction(op, src.dtype):
            # upcast to conserve precision
            dtype = np.dtype(np.uint64 if src.dtype.kind == "u" else np.int64)
        else:
            dtype = src.dtype
        res_dtype = dtype

    # TODO: Need to require initial to be given when the array is empty
    #       or a where mask is given.
    if (
        op
        in (
            UnaryRedCode.ARGMAX,
            UnaryRedCode.ARGMIN,
            UnaryRedCode.MAX,
            UnaryRedCode.MIN,
        )
        and src.dtype.kind == "c"
    ):
        raise NotImplementedError(
            "(arg)max/min not supported for complex-type arrays"
        )

    out_shape: NdShape = ()
    for dim in range(src.ndim):
        if dim not in axes:
            out_shape += (src.shape[dim],)
        elif keepdims:
            out_shape += (1,)

    if out is None:
        out = ndarray._from_inputs(
            shape=out_shape, dtype=res_dtype, inputs=(src, where)
        )
    elif out.shape != out_shape:
        errmsg = f"the output shapes do not match: expected {out_shape} but got {out.shape}"
        raise ValueError(errmsg)

    if dtype != src.dtype:
        src = src.astype(dtype)

    if out.dtype == res_dtype:
        result = out
    else:
        result = ndarray._from_inputs(
            shape=out_shape, dtype=res_dtype, inputs=(src, where)
        )

    # src might have taken eager path above:
    if not isinstance(src, ndarray):
        from .array import ndarray  # type: ignore [unreachable]

        src = convert_to_cupynumeric_ndarray(src)
    where_array = broadcast_where(where, src.shape)
    result._thunk.unary_reduction(
        op,
        src._thunk,
        get_where_thunk(where_array, src.shape),
        axis,
        axes,
        keepdims,
        args,
        initial,
    )

    if result is not out:
        out._thunk.convert(result._thunk)

    return out


def perform_binary_reduction(
    op: BinaryOpCode,
    one: ndarray,
    two: ndarray,
    dtype: np.dtype[Any],
    extra_args: tuple[Scalar, ...] = (),
) -> ndarray:
    from .array import ndarray

    args = (one, two)

    # We only handle bool types here for now
    assert dtype is not None and dtype == np.dtype(bool)

    # Collapsing down to a single value in this case
    # Check to see if we need to broadcast between inputs
    broadcast: NdShape | None
    if one.shape != two.shape:
        broadcast = np.broadcast_shapes(one.shape, two.shape)
    else:
        broadcast = None

    common_type = find_common_type(one, two)
    one_thunk = one._maybe_convert(common_type, args)._thunk
    two_thunk = two._maybe_convert(common_type, args)._thunk

    dst = ndarray._from_inputs(shape=(), dtype=dtype, inputs=args)
    dst._thunk.binary_reduction(
        op, one_thunk, two_thunk, broadcast, extra_args
    )
    return dst


def perform_where(mask: ndarray, one: ndarray, two: ndarray) -> ndarray:
    from .array import ndarray

    args = (mask, one, two)

    mask = mask._maybe_convert(np.dtype(bool), args)

    common_type = find_common_type(one, two)
    one = one._maybe_convert(common_type, args)
    two = two._maybe_convert(common_type, args)

    # Compute the output shape
    out_shape = np.broadcast_shapes(mask.shape, one.shape, two.shape)
    out = ndarray._from_inputs(shape=out_shape, dtype=common_type, inputs=args)
    out._thunk.where(mask._thunk, one._thunk, two._thunk)
    return out


def perform_scan(
    op: ScanCode,
    src: ndarray,
    axis: Any = None,
    dtype: npt.DTypeLike | None = None,
    out: ndarray | None = None,
    nan_to_identity: bool = False,
) -> ndarray:
    from .array import ndarray

    if src.dtype.kind != "c" and src.dtype.kind != "f":
        nan_to_identity = False

    if dtype is None:
        if out is None:
            if src.dtype.kind == "i":
                # Set dtype to default platform integer
                dtype = np.int_
            else:
                dtype = src.dtype
        else:
            dtype = out.dtype

    # flatten input when axis is None
    if axis is None:
        axis = 0
        src_arr = src.ravel()
    else:
        axis = normalize_axis_index(axis, src.ndim)
        src_arr = src

    if out is not None:
        if dtype != out.dtype:
            # if out array is specified, its type overrules dtype
            dtype = out.dtype
        if out.shape != src_arr.shape:
            raise NotImplementedError(
                "Varried output shape not supported. Output must have "
                "same shape as input (same size if no axis is provided"
            )
    else:
        out = ndarray(shape=src_arr.shape, dtype=dtype)

    if dtype != src_arr.dtype:
        if nan_to_identity:
            if op is ScanCode.SUM:
                nan_op = ConvertCode.SUM
            else:
                nan_op = ConvertCode.PROD
            # If convert is called, it will handle NAN conversion
            nan_to_identity = False
        else:
            nan_op = ConvertCode.NOOP
        # convert input to temporary for type conversion
        temp = ndarray(shape=src_arr.shape, dtype=dtype)
        temp._thunk.convert(src_arr._thunk, nan_op=nan_op)
        src_arr = temp

    out._thunk.scan(
        op,
        src_arr._thunk,
        axis=axis,
        dtype=dtype,
        nan_to_identity=nan_to_identity,
    )
    return out
