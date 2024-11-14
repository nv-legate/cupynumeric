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

from string import ascii_lowercase, ascii_uppercase
from typing import Sequence

from legate.core.utils import OrderedSet

Modes = tuple[list[str], list[str], list[str]]


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
AxesPair = tuple[Axes, Axes]
AxesPairLikeTuple = tuple[int | Axes, int | Axes]
AxesPairLike = int | AxesPairLikeTuple


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
