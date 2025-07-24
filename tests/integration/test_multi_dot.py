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

SHAPES = [
    # 2 arrays
    [(6, 5), (5, 4)],
    [(6, 5), (5,)],
    [(5,), (5, 4)],
    [(5,), (5,)],
    # 3 arrays
    [(6, 5), (5, 4), (4, 3)],
    [(6, 5), (5, 4), (4,)],
    [(5,), (5, 4), (4, 3)],
    [(5,), (5, 4), (4,)],
    # 4 arrays
    [(6, 5), (5, 4), (4, 3), (3, 2)],
    [(6, 5), (5, 4), (4, 3), (3,)],
    [(5,), (5, 4), (4, 3), (3, 2)],
    [(5,), (5, 4), (4, 3), (3,)],
]


@pytest.mark.parametrize("shapes", SHAPES)
def test_multi_dot(shapes):
    np_arrays = [mk_0to1_array(np, shape) for shape in shapes]
    num_arrays = [mk_0to1_array(num, shape) for shape in shapes]
    res_np = np.linalg.multi_dot(np_arrays)
    res_num = num.linalg.multi_dot(num_arrays)
    assert allclose(res_np, res_num)

    if len(shapes[0]) == 1:
        if len(shapes[-1]) == 1:
            out = num.zeros(())
        else:
            out = num.zeros((shapes[-1][1],))
    else:
        if len(shapes[-1]) == 1:
            out = num.zeros((shapes[0][0],))
        else:
            out = num.zeros((shapes[0][0], shapes[-1][1]))
    res_num = num.linalg.multi_dot(num_arrays, out=out)
    assert allclose(res_np, out)


class TestMultiDotErrors:
    def setup_method(self):
        A = mk_0to1_array(num, (2, 2))
        B = mk_0to1_array(num, (2, 2))
        C = mk_0to1_array(num, (2, 2))
        self.arrays = [A, B, C]

    def test_zero_array(self):
        arrays = []
        msg = "at least two arrays"
        with pytest.raises(ValueError, match=msg):
            num.linalg.multi_dot(arrays)

    def test_one_array(self):
        arrays = [num.array([[1, 2], [3, 4]])]
        msg = "at least two arrays"
        with pytest.raises(ValueError, match=msg):
            num.linalg.multi_dot(arrays)

    def test_invalid_array_dim_zero(self):
        A = num.array(3)
        B = mk_0to1_array(num, (2, 2))
        C = mk_0to1_array(num, (2, 2))
        arrays = [A, B, C]
        with pytest.raises(ValueError):
            num.linalg.multi_dot(arrays)

    def test_invalid_array_dim_one(self):
        A = mk_0to1_array(num, (2, 2))
        B = mk_0to1_array(num, (2,))
        C = mk_0to1_array(num, (2, 2))
        arrays = [A, B, C]
        with pytest.raises(ValueError):
            num.linalg.multi_dot(arrays)

    def test_invalid_array_dim_three(self):
        A = mk_0to1_array(num, (2, 2, 2))
        B = mk_0to1_array(num, (2, 2, 2))
        C = mk_0to1_array(num, (2, 2, 2))
        arrays = [A, B, C]
        with pytest.raises(ValueError):
            num.linalg.multi_dot(arrays)

    def test_invalid_array_shape(self):
        A = mk_0to1_array(num, (2, 2))
        B = mk_0to1_array(num, (3, 2))
        C = mk_0to1_array(num, (2, 2))
        arrays = [A, B, C]
        with pytest.raises(ValueError):
            num.linalg.multi_dot(arrays)

    def test_out_invalid_dim(self):
        out = num.zeros((2,))
        with pytest.raises(ValueError):
            num.linalg.multi_dot(self.arrays, out=out)

    @pytest.mark.xfail
    def test_out_invalid_shape(self):
        # In cuPyNumeric, it raises AssertionError
        out = num.zeros((2, 1))
        with pytest.raises(ValueError):
            num.linalg.multi_dot(self.arrays, out=out)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "dtype", (np.float32, np.int64), ids=lambda dtype: f"(dtype={dtype})"
    )
    def test_out_invalid_dtype(self, dtype):
        # In Numpy, for np.float32 and np.int64, it raises ValueError
        # In cuPyNumeric,
        # for np.float32, it pass
        # for np.int64, it raises TypeError: Unsupported type: int64
        out = num.zeros((2, 2), dtype=dtype)
        with pytest.raises(ValueError):
            num.linalg.multi_dot(self.arrays, out=out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
