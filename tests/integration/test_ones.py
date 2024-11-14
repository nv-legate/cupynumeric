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

# cupynumeric.ones(shape: NdShapeLike,
# dtype: npt.DTypeLike = <class 'numpy.float64'>) → ndarray

DIM = 5
SIZES = [
    (0,),
    (1),
    (DIM),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]
ALL_BUT_COMPLEX = ["?", "b", "h", "i", "l", "B", "H", "I", "L", "e", "f", "d"]
ALL_TYPES = ALL_BUT_COMPLEX + ["F", "D"]


def to_dtype(s):
    return str(np.dtype(s))


class TestOnes(object):
    @pytest.mark.xfail
    def test_size_none(self):
        res_np = np.ones(None)  # output is 1.0
        res_num = num.ones(None)
        # cupynumeric raises AssertionError
        # 'assert shape is not None'
        # in cupynumeric/array.py:ndarray:__init__
        assert np.equal(res_np, res_num)

    @pytest.mark.parametrize("size", (200 + 20j, "hello"))
    def test_size_invalid(self, size):
        with pytest.raises(TypeError):
            num.ones(size)

    def test_size_negative(self):
        size = -100
        with pytest.raises(ValueError):
            num.ones(size)

    @pytest.mark.parametrize("size", SIZES)
    def test_basic(self, size):
        res_np = np.ones(size)
        res_num = num.ones(size)
        assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", ALL_TYPES, ids=to_dtype)
    def test_basic_dtype(self, size, dtype):
        res_np = np.ones(size, dtype=dtype)
        res_num = num.ones(size, dtype=dtype)
        assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
