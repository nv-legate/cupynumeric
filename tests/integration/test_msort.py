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
from cupynumeric._utils import is_np2

# cupynumeric.msort(a: ndarray) → ndarray

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


@pytest.mark.skipif(is_np2, reason="numpy 2.0")
class TestmSort(object):
    @pytest.mark.xfail
    def test_arr_none(self):
        res_np = np.msort(
            None
        )  # AxisError: axis 0 is out of bounds for array of dimension 0
        res_num = num.msort(
            None
        )  # AttributeError: 'NoneType' object has no attribute 'shape'
        assert np.equal(res_np, res_num)

    @pytest.mark.parametrize("arr", ([], [[]], [[], []]))
    def test_arr_empty(self, arr):
        res_np = np.msort(arr)
        res_num = num.msort(arr)
        assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    def test_basic(self, size):
        arr_np = np.random.randint(-100, 100, size)
        arr_num = num.array(arr_np)

        res_np = np.msort(arr_np)
        res_num = num.msort(arr_num)
        assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize("size", SIZES)
    def test_basic_complex(self, size):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        )
        arr_num = num.array(arr_np)
        res_np = np.msort(arr_np)
        res_num = num.msort(arr_num)
        assert np.array_equal(res_num, res_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
