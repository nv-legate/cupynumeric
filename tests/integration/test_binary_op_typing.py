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

from itertools import product

import numpy as np
import pytest

import cupynumeric as num


def value_type(obj):
    if np.isscalar(obj):
        return "scalar"
    elif obj.ndim == 0:
        return f"{obj.dtype} 0d array"
    else:
        return f"{obj.dtype} array"


TYPES = [
    "b",
    "B",
    "h",
    "H",
    "i",
    "I",
    "l",
    "L",
    "q",
    "Q",
    "e",
    "f",
    "d",
    "F",
    "D",
]

ARRAY_VALUES = [[1]]

SCALAR_VALUES = [1, -1, 1.0, 1e-50, 1j]


def generate_array_array_cases():
    for idx, lhs_type in enumerate(TYPES):
        for rhs_type in TYPES[idx:]:
            for lhs_value, rhs_value in product(ARRAY_VALUES, ARRAY_VALUES):
                lhs_np = np.array(lhs_value, dtype=lhs_type)
                rhs_np = np.array(rhs_value, dtype=rhs_type)
                lhs_num = num.array(lhs_np)
                rhs_num = num.array(rhs_np)
                yield (lhs_np, rhs_np, lhs_num, rhs_num)

            for lhs_value, rhs_value in product(SCALAR_VALUES, SCALAR_VALUES):
                try:
                    lhs_np = np.array(lhs_value).astype(lhs_type)
                    rhs_np = np.array(rhs_value).astype(rhs_type)
                    lhs_num = num.array(lhs_np)
                    rhs_num = num.array(rhs_np)
                    yield (lhs_np, rhs_np, lhs_num, rhs_num)
                except TypeError:
                    pass

    for ty in TYPES:
        for array, scalar in product(ARRAY_VALUES, SCALAR_VALUES):
            array_np = np.array(array, dtype=ty)
            array_num = num.array(array_np)

            yield (array_np, scalar, array_num, scalar)


# TODO: NumPy's type coercion rules are confusing at best and impossible
# for any human being to understand in my opinion. My attempt to make
# sense of it for the past two days failed miserably. I managed to make
# the code somewhat compatible with NumPy for cases where Python scalars
# are passed.
#
# If anyone can do a better job than me and finally make cuPyNumeric
# implement the same typing rules, please put these tests back.
def generate_array_scalar_cases():
    for idx, lhs_type in enumerate(TYPES):
        for rhs_type in TYPES[idx:]:
            for array, scalar in product(ARRAY_VALUES, SCALAR_VALUES):
                try:
                    lhs_np = np.array(array).astype(lhs_type)
                    rhs_np = np.array(scalar).astype(rhs_type)
                    lhs_num = num.array(lhs_np)
                    rhs_num = num.array(rhs_np)
                    yield (lhs_np, rhs_np, lhs_num, rhs_num)
                except TypeError:
                    pass

                try:
                    lhs_np = np.array(scalar).astype(lhs_type)
                    rhs_np = np.array(array).astype(rhs_type)
                    lhs_num = num.array(lhs_np)
                    rhs_num = num.array(rhs_np)
                    yield (lhs_np, rhs_np, lhs_num, rhs_num)
                except TypeError:
                    pass


@pytest.mark.parametrize(
    "lhs_np, rhs_np, lhs_num, rhs_num", generate_array_array_cases(), ids=str
)
def test_array_array(lhs_np, rhs_np, lhs_num, rhs_num):
    print(f"{value_type(lhs_np)} x {value_type(rhs_np)}")

    try:
        out_np = np.add(lhs_np, rhs_np)
    except Exception as e:
        with pytest.raises(type(e)):
            num.add(lhs_num, rhs_num)
        return

    out_num = num.add(lhs_num, rhs_num)

    print(f"LHS {lhs_np}")
    print(f"RHS {rhs_np}")
    print(f"NumPy type: {out_np.dtype}, cuPyNumeric type: {out_num.dtype}")

    assert out_np.dtype == out_num.dtype


@pytest.mark.parametrize(
    "lhs_np, rhs_np, lhs_num, rhs_num", generate_array_scalar_cases(), ids=str
)
def test_array_scalar(lhs_np, rhs_np, lhs_num, rhs_num):
    print(f"{value_type(lhs_np)} x {value_type(rhs_np)}")

    out_np = np.add(lhs_np, rhs_np)
    out_num = num.add(lhs_num, rhs_num)

    print(f"LHS {lhs_np}")
    print(f"RHS {rhs_np}")
    print(f"NumPy type: {out_np.dtype}, cuPyNumeric type: {out_num.dtype}")

    assert out_np.dtype == out_num.dtype


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
