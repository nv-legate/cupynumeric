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
from .ufunc import (
    create_binary_ufunc,
    create_unary_ufunc,
    float_and_complex,
    float_dtypes,
)

sin = create_unary_ufunc(
    "Trigonometric sine, element-wise.",
    "sin",
    UnaryOpCode.SIN,
    float_and_complex,
)

cos = create_unary_ufunc(
    "Cosine element-wise.", "cos", UnaryOpCode.COS, float_and_complex
)

tan = create_unary_ufunc(
    "Compute tangent element-wise.", "tan", UnaryOpCode.TAN, float_and_complex
)

arcsin = create_unary_ufunc(
    "Inverse sine, element-wise.",
    "arcsin",
    UnaryOpCode.ARCSIN,
    float_and_complex,
)

arccos = create_unary_ufunc(
    "Trigonometric inverse cosine, element-wise.",
    "arccos",
    UnaryOpCode.ARCCOS,
    float_and_complex,
)

arctan = create_unary_ufunc(
    "Trigonometric inverse tangent, element-wise.",
    "arctan",
    UnaryOpCode.ARCTAN,
    float_and_complex,
)

arctan2 = create_binary_ufunc(
    "Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.",
    "arctan2",
    BinaryOpCode.ARCTAN2,
    float_dtypes,
)

hypot = create_binary_ufunc(
    "Given the “legs” of a right triangle, return its hypotenuse.",
    "hypot",
    BinaryOpCode.HYPOT,
    float_dtypes,
)

sinh = create_unary_ufunc(
    "Hyperbolic sine, element-wise.",
    "sinh",
    UnaryOpCode.SINH,
    float_and_complex,
)

cosh = create_unary_ufunc(
    "Hyperbolic cosine, element-wise.",
    "cosh",
    UnaryOpCode.COSH,
    float_and_complex,
)

tanh = create_unary_ufunc(
    "Compute hyperbolic tangent element-wise.",
    "tanh",
    UnaryOpCode.TANH,
    float_and_complex,
)

arcsinh = create_unary_ufunc(
    "Inverse hyperbolic sine element-wise.",
    "arcsinh",
    UnaryOpCode.ARCSINH,
    float_and_complex,
)

arccosh = create_unary_ufunc(
    "Inverse hyperbolic cosine, element-wise.",
    "arccosh",
    UnaryOpCode.ARCCOSH,
    float_and_complex,
)

arctanh = create_unary_ufunc(
    "Inverse hyperbolic tangent element-wise.",
    "arctanh",
    UnaryOpCode.ARCTANH,
    float_and_complex,
)

deg2rad = create_unary_ufunc(
    "Convert angles from degrees to radians.",
    "deg2rad",
    UnaryOpCode.DEG2RAD,
    float_dtypes,
)

rad2deg = create_unary_ufunc(
    "Convert angles from radians to degrees.",
    "rad2deg",
    UnaryOpCode.RAD2DEG,
    float_dtypes,
)

degrees = rad2deg

radians = deg2rad
