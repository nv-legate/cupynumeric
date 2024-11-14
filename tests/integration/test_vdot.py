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

import numpy as np
import pytest
from utils.comparisons import allclose
from utils.generators import mk_0to1_array

import cupynumeric as num

DTYPES = [np.float32, np.complex64]


def _vdot(a_dtype, b_dtype, shapeA, shapeB, lib):
    return lib.vdot(
        mk_0to1_array(lib, shapeA, dtype=a_dtype),
        mk_0to1_array(lib, shapeB, dtype=b_dtype),
    )


SIZES = (
    ((0,), (0,)),
    ((6,), (6,)),
    ((6,), (2, 3)),
    ((1, 2, 3), (6,)),
    ((6, 6), (2, 9, 2)),
)


@pytest.mark.parametrize("a_dtype", DTYPES)
@pytest.mark.parametrize("b_dtype", DTYPES)
@pytest.mark.parametrize(
    "shapeAB",
    SIZES,
    ids=lambda shapeAB: f"(shapeAB={shapeAB})",
)
def test_vdot_arrays(a_dtype, b_dtype, shapeAB):
    shapeA, shapeB = shapeAB
    assert allclose(
        _vdot(a_dtype, b_dtype, shapeA, shapeB, np),
        _vdot(a_dtype, b_dtype, shapeA, shapeB, num),
    )


@pytest.mark.parametrize(
    "shapeB",
    ((1,), (1, 1)),
    ids=lambda shapeB: f"(shapeB={shapeB})",
)
def test_vdot_scalar_and_arrays(shapeB):
    A = 5
    B_np = mk_0to1_array(np, shapeB)
    res1_np = np.vdot(A, B_np)
    res2_np = np.vdot(B_np, A)

    B_num = mk_0to1_array(num, shapeB)
    res1_num = num.vdot(A, B_num)
    res2_num = num.vdot(B_num, A)

    assert allclose(res1_np, res1_num)
    assert allclose(res2_np, res2_num)


def test_vdot_scalar():
    A = 5.2
    B = 1 + 2j
    res1_np = np.vdot(A, B)
    res2_np = np.vdot(B, A)

    res1_num = num.vdot(A, B)
    res2_num = num.vdot(B, A)

    assert allclose(res1_np, res1_num)
    assert allclose(res2_np, res2_num)


SIZES_ERRORS = (
    pytest.param(((0,), (1,)), marks=pytest.mark.xfail),
    pytest.param(((1,), (0,)), marks=pytest.mark.xfail),
    ((6,), (5,)),
    ((6,), (2, 4)),
    ((2, 3), (2, 4)),
)


class TestVdotErrors:
    @pytest.mark.parametrize(
        "shapeAB",
        SIZES_ERRORS,
        ids=lambda shapeAB: f"(shapeAB={shapeAB})",
    )
    def test_a_b_invalid_shape(self, shapeAB):
        # for ((0,), (1,)) and ((1,), (0,))
        # In Numpy, it raises ValueError
        # In cuPyNumeric, it pass
        expected_exc = ValueError
        shapeA, shapeB = shapeAB
        A_np = mk_0to1_array(np, shapeA)
        B_np = mk_0to1_array(np, shapeB)
        with pytest.raises(expected_exc):
            np.vdot(A_np, B_np)

        A_num = mk_0to1_array(num, shapeA)
        B_num = mk_0to1_array(num, shapeB)
        with pytest.raises(expected_exc):
            num.vdot(A_num, B_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "shapeB",
        ((0,), (2,), (1, 2)),
        ids=lambda shapeB: f"(shapeB={shapeB})",
    )
    def test_a_b_scalar_and_arrays(self, shapeB):
        # For shape of (0,), (2,), (1, 2),
        # In Numpy, it raises ValueError
        # In cuPyNumeric, it pass
        expected_exc = ValueError
        A = 5
        B_np = mk_0to1_array(np, shapeB)
        with pytest.raises(expected_exc):
            np.vdot(A, B_np)

        B_num = mk_0to1_array(num, shapeB)
        with pytest.raises(expected_exc):
            num.vdot(A, B_num)

    @pytest.mark.parametrize(
        "out_shape",
        ((0,), (1,)),
        ids=lambda out_shape: f"(out_shape={out_shape})",
    )
    def test_out_invalid_shape(self, out_shape):
        # np.vdot has no argument as out
        expected_exc = ValueError
        shapeA = (5,)
        shapeB = (5,)

        A_num = mk_0to1_array(num, shapeA)
        B_num = mk_0to1_array(num, shapeB)
        out_num = num.zeros(out_shape)
        with pytest.raises(expected_exc):
            num.vdot(A_num, B_num, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
