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
from utils.generators import mk_0to1_array

import cupynumeric as num

SHAPES = ((), (0,), (1,), (10,), (4, 5), (1, 4, 5))


@pytest.mark.parametrize(
    "shape_b", SHAPES, ids=lambda shape_b: f"(shape_b={shape_b})"
)
@pytest.mark.parametrize(
    "shape_a", SHAPES, ids=lambda shape_a: f"(shape_a={shape_a})"
)
def test_basic(shape_a, shape_b):
    a_np = mk_0to1_array(np, shape_a)
    b_np = mk_0to1_array(np, shape_b)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    res_np = np.outer(a_np, b_np)
    res_num = num.outer(a_num, b_num)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize(
    "shape_b", SHAPES, ids=lambda shape_b: f"(shape_b={shape_b})"
)
@pytest.mark.parametrize(
    "shape_a", SHAPES, ids=lambda shape_a: f"(shape_a={shape_a})"
)
def test_out(shape_a, shape_b):
    a_np = mk_0to1_array(np, shape_a)
    b_np = mk_0to1_array(np, shape_b)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    # if shape_a is (), prod is 1.0. Convert it into int.
    size_a = np.prod(shape_a).astype(int)
    size_b = np.prod(shape_b).astype(int)
    shape_out = (size_a, size_b)
    res_np = np.empty(shape_out)
    res_num = num.empty(shape_out)

    np.outer(a_np, b_np, out=res_np)
    num.outer(a_num, b_num, out=res_num)

    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize(
    "out_dt",
    (np.float32, np.complex128),
    ids=lambda out_dt: f"(out_dt={out_dt})",
)
def test_out_dtype(out_dt):
    shape_a = (4,)
    shape_b = (5,)
    a_np = mk_0to1_array(np, shape_a)
    b_np = mk_0to1_array(np, shape_b)
    a_num = num.array(a_np)
    b_num = num.array(b_np)

    size_a = np.prod(shape_a)
    size_b = np.prod(shape_b)
    shape_out = (size_a, size_b)
    res_np = np.empty(shape_out, dtype=out_dt)
    res_num = num.empty(shape_out, dtype=out_dt)

    np.outer(a_np, b_np, out=res_np)
    num.outer(a_num, b_num, out=res_num)

    assert np.array_equal(res_np, res_num)


class TestOuterErrors:
    def setup_method(self):
        shape_a = (4,)
        shape_b = (5,)
        self.a_np = mk_0to1_array(np, shape_a)
        self.b_np = mk_0to1_array(np, shape_b)
        self.a_num = num.array(self.a_np)
        self.b_num = num.array(self.b_np)

    @pytest.mark.parametrize(
        "out_shape",
        ((1, 20), (1,)),
        ids=lambda out_shape: f"(out_shape={out_shape})",
    )
    def test_out_invalid_shape(self, out_shape):
        expected_exc = ValueError
        out_np = np.empty(out_shape)
        out_num = num.empty(out_shape)
        with pytest.raises(expected_exc):
            np.outer(self.a_np, self.b_np, out=out_np)
        with pytest.raises(expected_exc):
            num.outer(self.a_num, self.b_num, out=out_num)

    @pytest.mark.parametrize(
        ("src_dt", "out_dt"),
        ((np.float64, np.int32), (np.complex128, np.float64)),
    )
    def test_out_invalid_dtype(self, src_dt, out_dt):
        expected_exc = TypeError
        shape_a = (4,)
        shape_b = (5,)
        a_np = mk_0to1_array(np, shape_a, dtype=src_dt)
        b_np = mk_0to1_array(np, shape_b, dtype=src_dt)
        a_num = num.array(a_np)
        b_num = num.array(b_np)

        out_shape = (a_np.size, b_np.size)
        out_np = np.empty(out_shape, dtype=out_dt)
        out_num = num.empty(out_shape, dtype=out_dt)

        with pytest.raises(expected_exc):
            np.outer(a_np, b_np, out=out_np)
        with pytest.raises(expected_exc):
            num.outer(a_num, b_num, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
