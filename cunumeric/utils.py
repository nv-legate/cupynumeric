# Copyright 2021-2022 NVIDIA Corporation
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

import traceback
from functools import reduce
from string import ascii_lowercase, ascii_uppercase
from types import FrameType
from typing import Any, Callable, List, Sequence, Tuple, TypeVar, Union

import legate.core.types as ty
import numpy as np
from legate.core.utils import OrderedSet

from .types import NdShape

SUPPORTED_DTYPES = {
    np.dtype(np.bool_): ty.bool_,
    np.dtype(np.int8): ty.int8,
    np.dtype(np.int16): ty.int16,
    np.dtype(np.int32): ty.int32,
    np.dtype(np.int64): ty.int64,
    np.dtype(np.uint8): ty.uint8,
    np.dtype(np.uint16): ty.uint16,
    np.dtype(np.uint32): ty.uint32,
    np.dtype(np.uint64): ty.uint64,
    np.dtype(np.float16): ty.float16,
    np.dtype(np.float32): ty.float32,
    np.dtype(np.float64): ty.float64,
    np.dtype(np.complex64): ty.complex64,
    np.dtype(np.complex128): ty.complex128,
}


def is_supported_type(dtype: Union[str, np.dtype[Any]]) -> bool:
    return np.dtype(dtype) in SUPPORTED_DTYPES


def to_core_dtype(dtype: Union[str, np.dtype[Any]]) -> ty.Dtype:
    core_dtype = SUPPORTED_DTYPES.get(np.dtype(dtype))
    if core_dtype is None:
        raise TypeError(f"cuNumeric does not support dtype={dtype}")
    return core_dtype


def is_advanced_indexing(key: Any) -> bool:
    if key is Ellipsis or key is None:  # np.newdim case
        return False
    if np.isscalar(key):
        return False
    if isinstance(key, slice):
        return False
    if isinstance(key, tuple):
        return any(is_advanced_indexing(k) for k in key)
    # Any other kind of thing leads to advanced indexing
    return True


_INTERNAL_MODULE_PREFIXES = ("cunumeric.", "legate.core.")


def is_internal_frame(frame: FrameType) -> bool:
    if "__name__" not in frame.f_globals:
        return False
    name = frame.f_globals["__name__"]
    return any(name.startswith(prefix) for prefix in _INTERNAL_MODULE_PREFIXES)


def find_last_user_stacklevel() -> int:
    stacklevel = 1
    for frame, _ in traceback.walk_stack(None):
        if not is_internal_frame(frame):
            break
        stacklevel += 1
    return stacklevel


def get_line_number_from_frame(frame: FrameType) -> str:
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def find_last_user_frames(top_only: bool = True) -> str:
    for last, _ in traceback.walk_stack(None):
        if not is_internal_frame(last):
            break

    if top_only:
        return get_line_number_from_frame(last)

    frames: list[FrameType] = []
    curr: Union[FrameType, None] = last
    while curr is not None:
        if "legion_top.py" in curr.f_code.co_filename:
            break
        frames.append(curr)
        curr = curr.f_back
    return "|".join(get_line_number_from_frame(f) for f in frames)


def calculate_volume(shape: NdShape) -> int:
    if len(shape) == 0:
        return 0
    return reduce(lambda x, y: x * y, shape)


T = TypeVar("T")


def tuple_pop(tup: Tuple[T, ...], index: int) -> Tuple[T, ...]:
    return tup[:index] + tup[index + 1 :]


Modes = Tuple[List[str], List[str], List[str]]


def dot_modes(a_ndim: int, b_ndim: int) -> Modes:
    a_modes = list(ascii_lowercase[:a_ndim])
    b_modes = list(ascii_uppercase[:b_ndim])
    if a_ndim == 0 or b_ndim == 0:
        out_modes = a_modes + b_modes
    elif b_ndim == 1:
        b_modes[-1] = a_modes[-1]
        out_modes = a_modes[:-1]
    else:
        b_modes[-2] = a_modes[-1]
        out_modes = a_modes[:-1] + b_modes[:-2] + [b_modes[-1]]
    return (a_modes, b_modes, out_modes)


def inner_modes(a_ndim: int, b_ndim: int) -> Modes:
    a_modes = list(ascii_lowercase[:a_ndim])
    b_modes = list(ascii_uppercase[:b_ndim])
    if a_ndim == 0 or b_ndim == 0:
        out_modes = a_modes + b_modes
    else:
        b_modes[-1] = a_modes[-1]
        out_modes = a_modes[:-1] + b_modes[:-1]
    return (a_modes, b_modes, out_modes)


def matmul_modes(a_ndim: int, b_ndim: int) -> Modes:
    if a_ndim == 0 or b_ndim == 0:
        raise ValueError("Scalars not allowed in matmul")
    a_modes = list(ascii_lowercase[-a_ndim:])
    b_modes = list(ascii_lowercase[-b_ndim:])
    if b_ndim >= 2:
        a_modes[-1] = "A"
        b_modes[-2] = "A"
    if b_ndim == 1:
        out_modes = a_modes[:-1]
    elif a_ndim == 1:
        out_modes = b_modes[:-2] + [b_modes[-1]]
    else:
        out_modes = (
            list(ascii_lowercase[-max(a_ndim, b_ndim) : -2])
            + [a_modes[-2]]
            + [b_modes[-1]]
        )
    return (a_modes, b_modes, out_modes)


Axes = Sequence[int]
AxesPair = Tuple[Axes, Axes]
AxesPairLikeTuple = Union[
    Tuple[int, int],
    Tuple[int, Axes],
    Tuple[Axes, int],
    Tuple[Axes, Axes],
]
AxesPairLike = Union[int, AxesPairLikeTuple]


def tensordot_modes(a_ndim: int, b_ndim: int, axes: AxesPairLike) -> Modes:
    def convert_int_axes(axes: int) -> AxesPair:
        return list(range(a_ndim - axes, a_ndim)), list(range(axes))

    def convert_seq_axes(axes: AxesPairLikeTuple) -> AxesPair:
        a_axes, b_axes = axes
        return (
            [a_axes] if isinstance(a_axes, int) else list(a_axes),
            [b_axes] if isinstance(b_axes, int) else list(b_axes),
        )

    def convert_axes(axes: AxesPairLike) -> AxesPair:
        if isinstance(axes, int):
            a_axes, b_axes = convert_int_axes(axes)
        else:
            a_axes, b_axes = convert_seq_axes(axes)

        return (
            [ax + a_ndim if ax < 0 else ax for ax in a_axes],
            [ax + b_ndim if ax < 0 else ax for ax in b_axes],
        )

    def check_axes(a_axes: Axes, b_axes: Axes) -> None:
        if (
            len(a_axes) != len(b_axes)
            or len(a_axes) > a_ndim
            or len(b_axes) > b_ndim
            or len(a_axes) != len(OrderedSet(a_axes))
            or len(b_axes) != len(OrderedSet(b_axes))
            or any(ax < 0 for ax in a_axes)
            or any(ax < 0 for ax in b_axes)
            or any(ax >= a_ndim for ax in a_axes)
            or any(ax >= b_ndim for ax in b_axes)
        ):
            raise ValueError("Invalid axes argument")

    a_axes, b_axes = convert_axes(axes)

    check_axes(a_axes, b_axes)

    a_modes = list(ascii_lowercase[:a_ndim])
    b_modes = list(ascii_uppercase[:b_ndim])
    for a_i, b_i in zip(a_axes, b_axes):
        b_modes[b_i] = a_modes[a_i]
    a_out = [
        a_modes[a_i]
        for a_i in sorted(OrderedSet(range(a_ndim)) - OrderedSet(a_axes))
    ]
    b_out = [
        b_modes[b_i]
        for b_i in sorted(OrderedSet(range(b_ndim)) - OrderedSet(b_axes))
    ]

    return (a_modes, b_modes, a_out + b_out)


def deep_apply(obj: Any, func: Callable[[Any], Any]) -> Any:
    """
    Apply the provided function to objects contained at any depth within a data
    structure.

    This function will recurse over arbitrary nestings of lists, tuples and
    dicts. This recursion logic is rather limited, but this function is
    primarily meant to be used for arguments of NumPy API calls, which
    shouldn't nest their arrays very deep.
    """
    if isinstance(obj, list):
        return [deep_apply(x, func) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_apply(x, func) for x in obj)
    elif isinstance(obj, dict):
        return {k: deep_apply(v, func) for k, v in obj.items()}
    else:
        return func(obj)
