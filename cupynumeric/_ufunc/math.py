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

from ..config import BinaryOpCode, UnaryOpCode, UnaryRedCode
from .ufunc import (
    all_but_boolean,
    all_dtypes,
    complex_dtypes,
    create_binary_ufunc,
    create_unary_ufunc,
    float_and_complex,
    float_dtypes,
    integer_dtypes,
)

add = create_binary_ufunc(
    "Add arguments element-wise.",
    "add",
    BinaryOpCode.ADD,
    all_dtypes,
    red_code=UnaryRedCode.SUM,
)

subtract = create_binary_ufunc(
    "Subtract arguments, element-wise.",
    "subtract",
    BinaryOpCode.SUBTRACT,
    all_dtypes,
)

multiply = create_binary_ufunc(
    "Multiply arguments element-wise.",
    "multiply",
    BinaryOpCode.MULTIPLY,
    all_dtypes,
    red_code=UnaryRedCode.PROD,
)

true_divide = create_binary_ufunc(
    "Returns a true division of the inputs, element-wise.",
    "true_divide",
    BinaryOpCode.DIVIDE,
    [ty + ty + "d" for ty in integer_dtypes] + float_and_complex,
)

floor_divide = create_binary_ufunc(
    """Return the largest integer smaller or equal to the division of the
inputs. It is equivalent to the Python ``//`` operator and pairs with the
Python ``%`` (`remainder`), function so that ``a = a % b + b * (a // b)``
up to roundoff.""",
    "floor_divide",
    BinaryOpCode.FLOOR_DIVIDE,
    all_dtypes,
)

divide = true_divide

logaddexp = create_binary_ufunc(
    "Logarithm of the sum of exponentiations of the inputs.",
    "logaddexp",
    BinaryOpCode.LOGADDEXP,
    float_dtypes,
)

logaddexp2 = create_binary_ufunc(
    "Logarithm of the sum of exponentiations of the inputs in base-2.",
    "logaddexp2",
    BinaryOpCode.LOGADDEXP2,
    float_dtypes,
)

negative = create_unary_ufunc(
    "Numerical negative, element-wise.",
    "negative",
    UnaryOpCode.NEGATIVE,
    all_but_boolean,
)

positive = create_unary_ufunc(
    "Numerical positive, element-wise.",
    "positive",
    UnaryOpCode.POSITIVE,
    all_but_boolean,
)

power = create_binary_ufunc(
    "First array elements raised to powers from second array, element-wise.",
    "power",
    BinaryOpCode.POWER,
    all_dtypes,
)

float_power = create_binary_ufunc(
    """First array elements raised to powers from second array, element-wise.

This differs from the power function in that integers, float16, and float32 are
promoted to floats with a minimum precision of float64 so that the result is
always inexact. The intent is that the function will return a usable result for
negative powers and seldom overflow for positive powers.""",
    "float_power",
    BinaryOpCode.FLOAT_POWER,
    ["d", "FFD", "D"],
)

remainder = create_binary_ufunc(
    "Return element-wise remainder of division.",
    "remainder",
    BinaryOpCode.MOD,
    ["?"] + integer_dtypes + float_dtypes,
)

mod = remainder

absolute = create_unary_ufunc(
    "Calculate the absolute value element-wise.",
    "absolute",
    UnaryOpCode.ABSOLUTE,
    (
        ["?"]
        + integer_dtypes
        + float_dtypes
        + [ty + ty.lower() for ty in complex_dtypes]
    ),
)

abs = absolute

rint = create_unary_ufunc(
    "Round elements of the array to the nearest integer.",
    "rint",
    UnaryOpCode.RINT,
    float_and_complex,
)

sign = create_unary_ufunc(
    "Returns an element-wise indication of the sign of a number.",
    "sign",
    UnaryOpCode.SIGN,
    all_but_boolean,
)

conjugate = create_unary_ufunc(
    "Return the complex conjugate, element-wise.",
    "conjugate",
    UnaryOpCode.CONJ,
    all_but_boolean,
)

conj = conjugate

exp = create_unary_ufunc(
    "Calculate the exponential of all elements in the input array.",
    "exp",
    UnaryOpCode.EXP,
    float_and_complex,
)

exp2 = create_unary_ufunc(
    "Calculate `2**p` for all `p` in the input array.",
    "exp2",
    UnaryOpCode.EXP2,
    float_and_complex,
)

log = create_unary_ufunc(
    "Natural logarithm, element-wise.",
    "log",
    UnaryOpCode.LOG,
    float_and_complex,
)

log2 = create_unary_ufunc(
    "Base-2 logarithm of x.", "log2", UnaryOpCode.LOG2, float_and_complex
)

log10 = create_unary_ufunc(
    "Return the base 10 logarithm of the input array, element-wise.",
    "log10",
    UnaryOpCode.LOG10,
    float_and_complex,
)

expm1 = create_unary_ufunc(
    "Calculate ``exp(x) - 1`` for all elements in the array.",
    "expm1",
    UnaryOpCode.EXPM1,
    float_and_complex,
)

log1p = create_unary_ufunc(
    "Return the natural logarithm of one plus the input array, element-wise.",
    "log1p",
    UnaryOpCode.LOG1P,
    float_and_complex,
)

square = create_unary_ufunc(
    "Return the element-wise square of the input.",
    "square",
    UnaryOpCode.SQUARE,
    all_but_boolean,
)

sqrt = create_unary_ufunc(
    "Return the non-negative square-root of an array, element-wise.",
    "sqrt",
    UnaryOpCode.SQRT,
    float_and_complex,
)

cbrt = create_unary_ufunc(
    "Return the cube-root of an array, element-wise.",
    "cbrt",
    UnaryOpCode.CBRT,
    float_dtypes,
)

reciprocal = create_unary_ufunc(
    "Return the reciprocal of the argument, element-wise.",
    "reciprocal",
    UnaryOpCode.RECIPROCAL,
    all_but_boolean,
)

gcd = create_binary_ufunc(
    "Returns the greatest common divisor of ``|x1|`` and ``|x2|``",
    "gcd",
    BinaryOpCode.GCD,
    integer_dtypes,
)

lcm = create_binary_ufunc(
    "Returns the lowest common multiple of ``|x1|`` and ``|x2|``",
    "lcm",
    BinaryOpCode.LCM,
    integer_dtypes,
)
