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

from ..config import BinaryOpCode, UnaryOpCode
from .ufunc import create_binary_ufunc, create_unary_ufunc, integer_dtypes

bitwise_and = create_binary_ufunc(
    "Compute the bit-wise AND of two arrays element-wise.",
    "bitwise_and",
    BinaryOpCode.BITWISE_AND,
    ["?"] + integer_dtypes,
)

bitwise_or = create_binary_ufunc(
    "Compute the bit-wise OR of two arrays element-wise.",
    "bitwise_or",
    BinaryOpCode.BITWISE_OR,
    ["?"] + integer_dtypes,
)

bitwise_xor = create_binary_ufunc(
    "Compute the bit-wise XOR of two arrays element-wise.",
    "bitwise_xor",
    BinaryOpCode.BITWISE_XOR,
    ["?"] + integer_dtypes,
)

invert = create_unary_ufunc(
    "Compute bit-wise inversion, or bit-wise NOT, element-wise.",
    "invert",
    UnaryOpCode.INVERT,
    ["?"] + integer_dtypes,
    overrides={"?": UnaryOpCode.LOGICAL_NOT},
)

left_shift = create_binary_ufunc(
    "Shift the bits of an integer to the left.",
    "left_shift",
    BinaryOpCode.LEFT_SHIFT,
    integer_dtypes,
)

right_shift = create_binary_ufunc(
    "Shift the bits of an integer to the right.",
    "right_shift",
    BinaryOpCode.RIGHT_SHIFT,
    integer_dtypes,
)


bitwise_not = invert
