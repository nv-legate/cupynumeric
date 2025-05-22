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

from .._array.util import convert_to_cupynumeric_ndarray
from ..config import BinaryOpCode, UnaryOpCode, UnaryRedCode
from .ufunc import (
    all_dtypes,
    create_binary_ufunc,
    create_unary_ufunc,
    float_dtypes,
    integer_dtypes,
    predicate_types_of,
    relation_types_of,
)

if TYPE_CHECKING:
    from .._array.array import ndarray


def _post_resolution_check(
    arr_x: ndarray,
    arr_y: ndarray,
    obj_x: Any,
    obj_y: Any,
    op_code: BinaryOpCode,
) -> tuple[ndarray, ndarray, BinaryOpCode]:
    """When we were passed Python integers, they may not fit into the operation
    dtype.  In that case, however, we can just define the result.
    Note that as of now, we don't try to do this if both operands were Python
    ints.
    """
    truthiness = None  # cannot guess the truthiness based on the scalar value

    if type(obj_x) is int and type(obj_y) is int:
        # No special behavior currently, check if values fit operation
        if arr_x.dtype.kind in "iu":
            # Check if original Python integer fits first operand.
            arr_x.dtype.type(obj_x)
        if arr_y.dtype.kind in "iu":
            # Check if original Python integer fits second operand.
            arr_y.dtype.type(obj_y)

    elif type(obj_x) is int and arr_x.dtype.kind in "iu":
        iinfo = np.iinfo(arr_x.dtype)
        if obj_x < iinfo.min:
            truthiness = op_code in {
                BinaryOpCode.NOT_EQUAL,
                BinaryOpCode.LESS,
                BinaryOpCode.LESS_EQUAL,
            }
        elif obj_x > iinfo.max:
            truthiness = op_code in {
                BinaryOpCode.NOT_EQUAL,
                BinaryOpCode.GREATER,
                BinaryOpCode.GREATER_EQUAL,
            }

        if truthiness is not None:
            # Replace with an always-true/always-false operation
            arr_x = convert_to_cupynumeric_ndarray(
                np.array(iinfo.min, dtype=arr_x.dtype)
            )
            op_code = (
                BinaryOpCode.GREATER_EQUAL if truthiness else BinaryOpCode.LESS
            )

    elif type(obj_y) is int and arr_y.dtype.kind in "iu":
        iinfo = np.iinfo(arr_y.dtype)
        if iinfo.min > obj_y:
            truthiness = op_code in {
                BinaryOpCode.NOT_EQUAL,
                BinaryOpCode.GREATER,
                BinaryOpCode.GREATER_EQUAL,
            }
        elif iinfo.max < obj_y:
            truthiness = op_code in {
                BinaryOpCode.NOT_EQUAL,
                BinaryOpCode.LESS,
                BinaryOpCode.LESS_EQUAL,
            }

        if truthiness is not None:
            # Replace with an always-true/always-false operation
            arr_y = convert_to_cupynumeric_ndarray(
                np.array(iinfo.min, dtype=arr_y.dtype)
            )
            op_code = (
                BinaryOpCode.GREATER_EQUAL if truthiness else BinaryOpCode.LESS
            )

    return arr_x, arr_y, op_code


greater = create_binary_ufunc(
    "Return the truth value of (x1 > x2) element-wise.",
    "greater",
    BinaryOpCode.GREATER,
    relation_types_of(all_dtypes),
    post_resolution_check=_post_resolution_check,
)

greater_equal = create_binary_ufunc(
    "Return the truth value of (x1 >= x2) element-wise.",
    "greater_equal",
    BinaryOpCode.GREATER_EQUAL,
    relation_types_of(all_dtypes),
    post_resolution_check=_post_resolution_check,
)

less = create_binary_ufunc(
    "Return the truth value of (x1 < x2) element-wise.",
    "less",
    BinaryOpCode.LESS,
    relation_types_of(all_dtypes),
    post_resolution_check=_post_resolution_check,
)

less_equal = create_binary_ufunc(
    "Return the truth value of (x1 =< x2) element-wise.",
    "less_equal",
    BinaryOpCode.LESS_EQUAL,
    relation_types_of(all_dtypes),
    post_resolution_check=_post_resolution_check,
)

not_equal = create_binary_ufunc(
    "Return (x1 != x2) element-wise.",
    "not_equal",
    BinaryOpCode.NOT_EQUAL,
    relation_types_of(all_dtypes),
    post_resolution_check=_post_resolution_check,
)

equal = create_binary_ufunc(
    "Return (x1 == x2) element-wise.",
    "equal",
    BinaryOpCode.EQUAL,
    relation_types_of(all_dtypes),
    post_resolution_check=_post_resolution_check,
)

logical_and = create_binary_ufunc(
    "Compute the truth value of x1 AND x2 element-wise.",
    "logical_and",
    BinaryOpCode.LOGICAL_AND,
    relation_types_of(all_dtypes),
    red_code=UnaryRedCode.ALL,
)

logical_or = create_binary_ufunc(
    "Compute the truth value of x1 OR x2 element-wise.",
    "logical_or",
    BinaryOpCode.LOGICAL_OR,
    relation_types_of(all_dtypes),
    red_code=UnaryRedCode.ANY,
)

logical_xor = create_binary_ufunc(
    "Compute the truth value of x1 XOR x2, element-wise.",
    "logical_xor",
    BinaryOpCode.LOGICAL_XOR,
    relation_types_of(all_dtypes),
)

logical_not = create_unary_ufunc(
    "Compute bit-wise inversion, or bit-wise NOT, element-wise.",
    "logical_not",
    UnaryOpCode.LOGICAL_NOT,
    (
        ["??"]
        + predicate_types_of(integer_dtypes)
        + predicate_types_of(float_dtypes)
    ),
    overrides={"?": UnaryOpCode.LOGICAL_NOT},
)

maximum = create_binary_ufunc(
    "Element-wise maximum of array elements.",
    "maximum",
    BinaryOpCode.MAXIMUM,
    all_dtypes,
    red_code=UnaryRedCode.MAX,
)

fmax = maximum

minimum = create_binary_ufunc(
    "Element-wise minimum of array elements.",
    "minimum",
    BinaryOpCode.MINIMUM,
    all_dtypes,
    red_code=UnaryRedCode.MIN,
)

fmin = minimum
