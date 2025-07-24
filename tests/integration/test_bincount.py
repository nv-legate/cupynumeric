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

N = 8000
MAX_VAL = 9
LARGE_NUM_BINS = 20000

DTYPES = [np.int64, np.int32, np.int16]
MINLENGTHS = [0, 5, 15]


class TestBincountNegative:
    def test_dtype_negative(self):
        expected_exc = TypeError
        arr_np = np.arange(5, dtype=float)
        arr_num = num.arange(5, dtype=float)
        with pytest.raises(expected_exc):
            np.bincount(arr_np)
        with pytest.raises(expected_exc):
            num.bincount(arr_num)

    def test_array_negative(self):
        expected_exc = ValueError
        arr_np = np.array((-1, 2, 5))
        arr_num = num.array((-1, 2, 5))
        with pytest.raises(expected_exc):
            np.bincount(arr_np)
        with pytest.raises(expected_exc):
            num.bincount(arr_num)

    def test_array_ndim(self):
        expected_exc = ValueError
        size = (2,) * 3
        arr_np = np.random.randint(0, high=9, size=size)
        arr_num = num.random.randint(0, high=9, size=size)
        with pytest.raises(expected_exc):
            np.bincount(arr_np)
        with pytest.raises(expected_exc):
            num.bincount(arr_num)

    def test_minlength_negative(self):
        expected_exc = ValueError
        minlength = -5
        arr_np = np.arange(5)
        arr_num = num.arange(5)
        with pytest.raises(expected_exc):
            np.bincount(arr_np, minlength=minlength)
        with pytest.raises(expected_exc):
            num.bincount(arr_num, minlength=minlength)

    def test_weight_mismatch(self):
        expected_exc = ValueError
        v_np = np.random.randint(0, 9, size=N)
        w_np = np.random.randn(N + 1)
        v_num = num.random.randint(0, 9, size=N)
        w_num = num.random.randn(N + 1)
        with pytest.raises(expected_exc):
            np.bincount(v_np, weights=w_np)
        with pytest.raises(expected_exc):
            num.bincount(v_num, weights=w_num)

    @pytest.mark.parametrize(
        "weight",
        [
            ("1", "2"),
            ("2", "x"),
            (b"x", b"y"),
            (np.datetime64(1, "Y"), np.datetime64(123, "Y")),
            np.array((5, 3), dtype="F"),
        ],
    )
    @pytest.mark.xfail(
        reason="different behavior when casting weight to float64"
    )
    def test_weight_dtype(self, weight):
        expected_exc = TypeError
        arr_np = np.arange(2)
        arr_num = num.arange(2)
        w_np = np.array(weight)
        w_num = num.array(weight)
        with pytest.raises(expected_exc):
            # TypeError: Cannot cast array data from dtype('<U1') to
            # dtype('float64') according to the rule 'safe'
            np.bincount(arr_np, weights=w_np)
        with pytest.raises(expected_exc):
            # - does not raise exception when the values can be casted float64
            # - raises ValueError when weights are complex types
            # - other dtype will hit ValueError that bubbles from eager.py:
            #   could not convert string to float: 'x'
            num.bincount(arr_num, weights=w_num)


def test_out_size():
    arr = num.array([0, 1, 1, 3, 2, 1, 7, 23])
    assert num.bincount(arr).size == num.amax(arr) + 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("minlength", MINLENGTHS)
def test_bincount_basic(dtype, minlength):
    v_num = num.random.randint(0, MAX_VAL, size=N, dtype=dtype)
    out_num = num.bincount(v_num, minlength=minlength)

    v_np = v_num.__array__()
    out_np = np.bincount(v_np, minlength=minlength)
    assert num.array_equal(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_weights(dtype):
    v_num = num.random.randint(0, MAX_VAL, size=N, dtype=dtype)
    w_num = num.random.randn(N)
    out_num = num.bincount(v_num, weights=w_num)

    v_np = v_num.__array__()
    w_np = w_num.__array__()
    out_np = np.bincount(v_np, weights=w_np)

    assert allclose(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_high_bins(dtype):
    v_num = num.array([0, LARGE_NUM_BINS], dtype=dtype)
    out_num = num.bincount(v_num)

    v_np = v_num.__array__()
    out_np = np.bincount(v_np)

    assert num.array_equal(out_np, out_num)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bincount_weights_high_bins(dtype):
    v_num = num.array([0, LARGE_NUM_BINS], dtype=dtype)
    w_num = num.random.randn(2)
    out_num = num.bincount(v_num, weights=w_num)

    v_np = v_num.__array__()
    w_np = w_num.__array__()
    out_np = np.bincount(v_np, weights=w_np)

    assert allclose(out_np, out_num)


@pytest.mark.parametrize(
    "weights", [None, np.array((0.5,)), np.array((22,))], ids=str
)
def test_bincount_size_one(weights):
    arr_np = np.random.randint(0, 255, size=1)
    arr_num = num.array(arr_np)
    bins_np = np.bincount(arr_np, weights=weights)
    bins_num = num.bincount(arr_num, weights=weights)
    assert np.array_equal(bins_np, bins_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
