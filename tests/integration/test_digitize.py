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

import math

import numpy as np
import pytest

import cupynumeric as num

DTYPES = (np.uint32, np.uint64, np.float32, np.float64)

SHAPES = ((10,), (2, 5), (3, 7, 10))


class TestDigitizeErrors(object):
    def test_complex_array(self):
        a = np.array([2, 3, 10, 9], dtype=np.complex64)
        bins = [0, 3, 5]
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            num.digitize(a, bins)
        with pytest.raises(expected_exc):
            np.digitize(a, bins)

    @pytest.mark.xfail
    def test_bad_array(self):
        bins = [0, 5, 3]
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            # cupynumeric raises TypeError
            num.digitize(None, bins)
        with pytest.raises(expected_exc):
            np.digitize(None, bins)

    @pytest.mark.xfail
    def test_bad_bins(self):
        a = [2, 3, 10, 9]
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            # cupynumeric raises TypeError
            num.digitize(a, None)
        with pytest.raises(expected_exc):
            np.digitize(a, None)

    def test_bins_non_monotonic(self):
        a = [2, 3, 10, 9]
        bins = [0, 5, 3]
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.digitize(a, bins)
        with pytest.raises(expected_exc):
            np.digitize(a, bins)

    def test_bins_ndim(self):
        a = [2, 3, 10, 9]
        bins = np.array([[0], [5], [3]])
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            num.digitize(a, bins)
        with pytest.raises(expected_exc):
            np.digitize(a, bins)


def generate_random(shape, dtype):
    a_np = None
    size = math.prod(shape)
    if np.issubdtype(dtype, np.integer):
        a_np = np.array(
            np.random.randint(
                np.iinfo(dtype).min,
                np.iinfo(dtype).max,
                size=size,
                dtype=dtype,
            ),
            dtype=dtype,
        )
    elif np.issubdtype(dtype, np.floating):
        a_np = np.array(np.random.random(size=size), dtype=dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        a_np = np.array(
            np.random.random(size=size) + np.random.random(size=size) * 1j,
            dtype=dtype,
        )
    else:
        assert False
    return a_np.reshape(shape)


@pytest.mark.parametrize("right", (True, False))
def test_empty(right):
    bins = [0, 3, 5]
    assert len(num.digitize([], bins, right=right)) == 0


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("right", (True, False))
def test_increasing_bins(shape, dtype, right):
    a = generate_random(shape, dtype)
    bins = [0, 3, 5]

    a_num = num.array(a)
    bins_num = num.array(bins)

    res_np = np.digitize(a, bins, right=right)
    res_num = num.digitize(a, bins, right=right)
    assert num.array_equal(res_np, res_num)

    res_np = np.digitize(a, bins, right=right)
    res_num = num.digitize(a_num, bins, right=right)
    assert num.array_equal(res_np, res_num)

    res_np = np.digitize(a, bins, right=right)
    res_num = num.digitize(a_num, bins_num, right=right)
    assert num.array_equal(res_np, res_num)


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("right", (True, False))
def test_decreasing_bins(shape, dtype, right):
    a = generate_random(shape, dtype)
    bins = [5, 3, 0]

    a_num = num.array(a)
    bins_num = num.array(bins)

    res_np = np.digitize(a, bins, right=right)
    res_num = num.digitize(a, bins, right=right)
    assert num.array_equal(res_np, res_num)

    res_np = np.digitize(a, bins, right=right)
    res_num = num.digitize(a_num, bins, right=right)
    assert num.array_equal(res_np, res_num)

    res_np = np.digitize(a, bins, right=right)
    res_num = num.digitize(a_num, bins_num, right=right)
    assert num.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
