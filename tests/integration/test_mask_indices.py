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
N = 100


def _test(mask_func, n, k):
    num_f = getattr(num, mask_func)
    np_f = getattr(np, mask_func)

    a = num.mask_indices(n, num_f, k=k)
    an = np.mask_indices(n, np_f, k=k)
    assert num.array_equal(a, an)


def _test_default_k(mask_func, n):
    num_f = getattr(num, mask_func)
    np_f = getattr(np, mask_func)

    a = num.mask_indices(n, num_f)
    an = np.mask_indices(n, np_f)
    assert num.array_equal(a, an)


@pytest.mark.parametrize("n", [0, 1, 100], ids=lambda n: f"(n={n})")
@pytest.mark.parametrize("mask_func", FUNCTIONS)
def test_mask_indices_default_k(n, mask_func):
    _test_default_k(mask_func, n)


@pytest.mark.parametrize(
    "k", KS + (-N, N, -10 * N, 10 * N), ids=lambda k: f"(k={k})"
)
@pytest.mark.parametrize("mask_func", FUNCTIONS)
def test_mask_indices(k, mask_func):
    _test(mask_func, N, k)


@pytest.mark.xfail
@pytest.mark.parametrize(
    "k", [-10.5, -0.5, 0.5, 10.5], ids=lambda k: f"(k={k})"
)
@pytest.mark.parametrize("mask_func", FUNCTIONS)
def test_mask_indices_float_k(k, mask_func):
    # cuPyNumeric: struct.error: required argument is not an integer
    # Numpy: pass
    _test(mask_func, N, k)


class TestMaskIndicesErrors:
    def test_negative_int_n(self):
        with pytest.raises(ValueError):
            num.mask_indices(-1, num.tril)

    @pytest.mark.parametrize("n", [-10.0, 0.0, 10.5])
    def test_float_n(self, n):
        msg = "expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.mask_indices(n, num.tril)

    @pytest.mark.xfail
    def test_k_complex(self):
        # In cuPyNumeric, it raises struct.error,
        # msg is required argument is not an integer
        # In Numpy, it raises TypeError,
        # msg is '<=' not supported between instances of 'complex' and 'int'
        with pytest.raises(TypeError):
            num.mask_indices(10, num.tril, 1 + 2j)

    @pytest.mark.xfail
    def test_k_none(self):
        # In cuPyNumeric, it raises struct.error,
        # msg is required argument is not an integer
        # In Numpy, it raises TypeError,
        # msg is unsupported operand type(s) for -: 'NoneType' and 'int'
        with pytest.raises(TypeError):
            num.mask_indices(10, num.tril, None)

    def test_mask_func_bad_argument(self):
        msg = "takes 1 positional argument but 2 were given"
        with pytest.raises(TypeError, match=msg):
            num.mask_indices(10, num.block)

    def test_mask_func_str(self):
        msg = "'str' object is not callable"
        with pytest.raises(TypeError, match=msg):
            num.mask_indices(10, "abc")

    def test_mask_func_none(self):
        msg = "'NoneType' object is not callable"
        with pytest.raises(TypeError, match=msg):
            num.mask_indices(10, None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
