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

import cupynumeric as num

SHAPES = [
    (10, 20),
]


DTYPES = ["h", "i", "l", "H", "I", "L", "e", "f", "d"]


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_modf(shape):
    x_np = np.random.random(shape)
    x_num = num.array(x_np)

    outs_np = np.modf(x_np)
    outs_num = num.modf(x_num)

    for out_np, out_num in zip(outs_np, outs_num):
        assert allclose(out_np, out_num)

    # Test integer input
    outs_np = np.modf(x_np.astype("i"))
    outs_num = num.modf(x_num.astype("i"))

    for out_np, out_num in zip(outs_np, outs_num):
        assert allclose(out_np, out_num)

    # Test positional outputs
    out1_np = np.empty(shape, dtype="f")
    out2_np = np.empty(shape, dtype="e")

    out1_num = num.empty(shape, dtype="f")
    out2_num = num.empty(shape, dtype="e")

    np.modf(x_np, out1_np, out2_np)
    num.modf(x_num, out1_num, out2_num)

    assert allclose(out1_np, out1_num)
    assert allclose(out2_np, out2_num)

    (tmp1_np, out2_np) = np.modf(x_np, out1_np)
    (tmp1_num, out2_num) = num.modf(x_num, out1_num)

    assert allclose(out1_np, out1_num)
    assert allclose(out2_np, out2_num)
    assert tmp1_num is out1_num

    # Test keyword outputs
    out1_np = np.empty(shape, dtype="f")
    out2_np = np.empty(shape, dtype="e")

    out1_num = num.empty(shape, dtype="f")
    out2_num = num.empty(shape, dtype="e")

    np.modf(x_np, out=(out1_np, out2_np))
    num.modf(x_num, out=(out1_num, out2_num))

    assert allclose(out1_np, out1_num)
    assert allclose(out2_np, out2_num)

    (tmp2_np, out2_np) = np.modf(x_np, out=(out1_np, None))
    (tmp2_num, out2_num) = num.modf(x_num, out=(out1_num, None))

    assert allclose(out1_np, out1_num)
    assert allclose(out2_np, out2_num)
    assert out1_num is tmp2_num

    (out1_np, tmp2_np) = np.modf(x_np, out=(None, out2_np))
    (out1_num, tmp2_num) = num.modf(x_num, out=(None, out2_num))

    assert allclose(out1_np, out1_num)
    assert allclose(out2_np, out2_num)
    assert out2_num is tmp2_num


@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_floating(shape):
    x_np = np.random.random(shape)
    x_num = num.array(x_np)

    frexp_np = np.frexp(x_np)
    frexp_num = num.frexp(x_num)

    for out_np, out_num in zip(frexp_np, frexp_num):
        assert allclose(out_np, out_num)

    out1_np = np.empty(shape, dtype="f")
    out2_np = np.empty(shape, dtype="l")

    out1_num = num.empty(shape, dtype="f")
    out2_num = num.empty(shape, dtype="l")

    np.frexp(x_np, out=(out1_np, out2_np))
    num.frexp(x_num, out=(out1_num, out2_num))

    assert allclose(out1_np, out1_num)
    assert allclose(out2_np, out2_num)

    ldexp_np = np.ldexp(*frexp_np)
    ldexp_num = num.ldexp(*frexp_num)

    assert allclose(ldexp_np, ldexp_num)

    ldexp_np = np.ldexp(out1_np, out2_np)
    ldexp_num = num.ldexp(out1_num, out2_num)

    assert allclose(ldexp_np, ldexp_num)

    ldexp_np = np.ldexp(1.0, np.array([1, 2], dtype=np.int8))
    ldexp_num = num.ldexp(1.0, num.array([1, 2], dtype=np.int8))

    assert ldexp_np.dtype == ldexp_num.dtype == np.float64


@pytest.mark.parametrize("fun", ("modf", "frexp"))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_typing_unary(fun, dtype, shape):
    fn_np = getattr(np, fun)
    fn_num = getattr(num, fun)
    assert np.array_equal(
        fn_np(np.ones(shape, dtype=dtype)),
        fn_num(np.ones(shape, dtype=dtype)),
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
