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
from utils.generators import mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


class TestAngleErrors:
    def test_none_array(self):
        expected_exc = AttributeError
        msg = "'int' object has no attribute 'arctan2'"
        with pytest.raises(expected_exc, match=msg):
            np.angle(None)
        expected_exc = TypeError
        msg = "can't compute 'angle' for None"
        with pytest.raises(expected_exc, match=msg):
            num.angle(None)


class TestAngle:
    def test_empty_array(self):
        res_np = np.angle([])
        res_num = num.angle([])
        assert np.array_equal(res_np, res_num)

    def test_zero_input(self):
        res_np = np.angle(0)
        res_num = num.angle(0)
        assert np.array_equal(res_np, res_num)

    def test_pure_real_and_imaginary(self):
        # Testing pure real and pure imaginary numbers
        assert np.array_equal(num.angle(5), np.angle(5))
        assert np.array_equal(num.angle(-5), np.angle(-5))
        assert np.array_equal(num.angle(5j), np.angle(5j))
        assert np.array_equal(num.angle(-5j), np.angle(-5j))

    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    @pytest.mark.parametrize("in_type", (int, float, complex))
    @pytest.mark.parametrize("deg", (False, True))
    def test_basic(self, ndim, in_type, deg):
        shape = (5,) * ndim
        np_arr = mk_seq_array(np, shape).astype(in_type)
        num_arr = mk_seq_array(num, shape).astype(in_type)

        res_np = np.angle(np_arr, deg)
        res_num = num.angle(num_arr, deg)
        assert np.array_equal(res_num, res_np)

    @pytest.mark.parametrize(
        "array",
        (
            [1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j],
            [[1 + 1j, -1 - 1j], [1 - 1j, -1 + 1j]],
            [
                [[1 + 1j, -1 - 1j], [1 - 1j, -1 + 1j]],
                [[1j, -1j], [1 + 1j, -1 - 1j]],
            ],
        ),
    )
    @pytest.mark.parametrize("deg", (False, True))
    def test_complex_arrays(self, array, deg):
        res_np = np.angle(array, deg)
        res_num = num.angle(array, deg)
        assert np.array_equal(res_num, res_np)

    def test_edge_cases(self):
        # Testing angles with large and small numbers
        assert np.array_equal(np.angle(1e10 + 1e10j), num.angle(1e10 + 1e10j))
        assert np.array_equal(
            np.angle(1e-10 + 1e-10j), num.angle(1e-10 + 1e-10j)
        )

    def test_nan(self):
        # Testing behavior with NaN and Inf values
        assert np.array_equal(
            np.angle(np.nan + 1j), num.angle(np.nan + 1j), equal_nan=True
        )

    def test_inf(self):
        assert np.array_equal(np.angle(np.inf + 1j), num.angle(np.inf + 1j))
        assert np.array_equal(np.angle(1j + np.inf), num.angle(1j + np.inf))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
