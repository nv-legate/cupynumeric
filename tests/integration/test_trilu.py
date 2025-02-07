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

import cupynumeric as num

KS = (0, -1, 1, -2, 2)
FUNCTIONS = ("tril", "triu")


def _test(func, anp, a, k):
    num_f = getattr(num, func)
    np_f = getattr(np, func)

    b = num_f(a, k=k)
    bnp = np_f(anp, k=k)

    assert num.array_equal(b, bnp)


ARRAY_SHAPE = (
    (0,),
    (1,),
    (10,),
    (1, 10),
    (10, 10),
    (1, 1, 10),
    (1, 10, 10),
    (10, 10, 10),
)


@pytest.mark.parametrize("k", KS + (-10, 10), ids=lambda k: f"(k={k})")
@pytest.mark.parametrize("dtype", (int, float), ids=str)
@pytest.mark.parametrize(
    "shape", ARRAY_SHAPE, ids=lambda shape: f"(shape={shape})"
)
@pytest.mark.parametrize("func", FUNCTIONS)
def test_trilu(func, shape, dtype, k):
    anp = np.ones(shape, dtype=dtype)
    a = num.ones(shape, dtype=dtype)

    _test(func, anp, a, k)


@pytest.mark.xfail
@pytest.mark.parametrize("k", (-2.5, 0.0, 2.5), ids=lambda k: f"(k={k})")
@pytest.mark.parametrize("func", FUNCTIONS)
def test_trilu_float_k(func, k):
    # cuPyNumeric: struct.error: required argument is not an integer
    # Numpy: pass
    shape = (10, 10)
    anp = np.ones(shape)
    a = num.ones(shape)

    _test(func, anp, a, k)


class TestTriluErrors:
    def test_arr_none(self):
        msg = "'NoneType' object has no attribute 'ndim'"
        with pytest.raises(AttributeError, match=msg):
            num.tril(None)

    @pytest.mark.xfail
    def test_k_none(self):
        # In cuPyNumeric, it raises struct.error,
        # msg is required argument is not an integer
        # In Numpy, it raises TypeError,
        # msg is bad operand type for unary -: 'NoneType'
        a = num.ones((3, 3))
        with pytest.raises(TypeError):
            num.tril(a, k=None)

    def test_m_scalar(self):
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.tril(0)
        with pytest.raises(expected_exc):
            num.tril(0)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
