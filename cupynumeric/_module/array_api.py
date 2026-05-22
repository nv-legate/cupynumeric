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

from typing import Any

import numpy as np
import numpy.typing as npt

from .._array.array import ndarray
from .._array.util import add_boilerplate
from .._array_api import _KIND_TO_DTYPES, _check_device
from .array_joining import concatenate as concat
from .array_transpose import transpose as permute_dims
from .._ufunc.bit_twiddling import (
    invert as bitwise_invert,
    left_shift as bitwise_left_shift,
    right_shift as bitwise_right_shift,
)
from .._ufunc.math import power as pow
from .._ufunc.trigonometric import (
    arccos as acos,
    arccosh as acosh,
    arcsin as asin,
    arcsinh as asinh,
    arctan as atan,
    arctan2 as atan2,
    arctanh as atanh,
)

__all__ = (
    "acos",
    "acosh",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "concat",
    "isdtype",
    "matrix_transpose",
    "permute_dims",
    "pow",
)


def _dtype_type(dtype: npt.DTypeLike) -> type[np.generic]:
    return np.dtype(dtype).type


def _matches_kind(dtype: type[np.generic], kind: npt.DTypeLike | str) -> bool:
    if isinstance(kind, str):
        return dtype in _KIND_TO_DTYPES[kind].values()
    return dtype is _dtype_type(kind)


def _is_complex_floating(dtype: type[np.generic]) -> bool:
    return dtype in _KIND_TO_DTYPES["complex floating"].values()


@add_boilerplate("x")
def astype(
    x: ndarray,
    dtype: npt.DTypeLike,
    /,
    *,
    copy: bool = True,
    device: Any | None = None,
) -> ndarray:
    """Cast an array to an Array API dtype.

    With ``copy=True`` (the default), a fresh array is always returned; with
    ``copy=False``, the input is returned unchanged when its dtype already
    matches.

    Supports only ``device=None``. Complex floating arrays cannot be cast to
    non-complex dtypes.
    """
    _check_device(device)
    target_dtype = np.dtype(dtype)
    if _is_complex_floating(x.dtype.type) and not _is_complex_floating(
        target_dtype.type
    ):
        raise TypeError(
            "Array API astype does not permit casting complex floating "
            "arrays to non-complex dtypes"
        )

    # ndarray.astype(copy=True) returns self for same-dtype casts despite its
    # docstring promising a copy. Implement Array API copy semantics here.
    if x.dtype == target_dtype:
        return x.copy() if copy else x
    return x.astype(target_dtype)


def isdtype(
    dtype: npt.DTypeLike,
    kind: npt.DTypeLike | str | tuple[npt.DTypeLike | str, ...],
    /,
) -> bool:
    """Return whether a dtype matches an Array API dtype kind.

    ``kind`` may be a dtype, a standard dtype-kind string, or a tuple of those
    values.
    """
    dtype_type = _dtype_type(dtype)
    kinds = kind if isinstance(kind, tuple) else (kind,)
    for item in kinds:
        if isinstance(item, str) and item not in _KIND_TO_DTYPES:
            raise ValueError(f"unrecognized Array API dtype kind {item!r}")
    return any(_matches_kind(dtype_type, item) for item in kinds)


@add_boilerplate("x")
def matrix_transpose(x: ndarray, /) -> ndarray:
    """Transpose the last two dimensions of an array.

    Requires an input with at least two dimensions.
    """
    return x.mT
