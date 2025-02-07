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

import cupynumeric as num

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


class TestSwapAxesModule:
    def test_small(self):
        a_num = num.array(a)
        b = np.swapaxes(a, 0, 1)
        b_num = num.swapaxes(a_num, 0, 1)
        assert np.array_equal(b, b_num)

    def test_tall(self):
        a_tall = np.concatenate((a,) * 100)
        a_tall_num = num.array(a_tall)

        b_tall = np.swapaxes(a_tall, 0, 1)
        b_tall_num = num.swapaxes(a_tall_num, 0, 1)
        assert np.array_equal(b_tall, b_tall_num)

    def test_wide(self):
        a_wide = np.concatenate((a,) * 100, axis=1)
        a_wide_num = num.array(a_wide)

        b_wide = np.swapaxes(a_wide, 0, 1)
        b_wide_num = num.swapaxes(a_wide_num, 0, 1)
        assert np.array_equal(b_wide, b_wide_num)

    def test_big(self):
        a_tall = np.concatenate((a,) * 100)
        a_big = np.concatenate((a_tall,) * 100, axis=1)
        a_big_num = num.array(a_big)

        b_big = np.swapaxes(a_big, 0, 1)
        b_big_num = num.swapaxes(a_big_num, 0, 1)
        assert np.array_equal(b_big, b_big_num)

    @pytest.mark.parametrize(
        "axes",
        ((0, 0), (-3, 1), (0, 2), (-3, -2)),
        ids=lambda axes: f"(axes={axes})",
    )
    def test_axes(self, axes):
        shape = (3, 4, 5)
        np_arr = mk_seq_array(np, shape)
        num_arr = num.array(np_arr)
        axis1, axis2 = axes

        res_np = np.swapaxes(np_arr, axis1, axis2)
        res_num = num.swapaxes(num_arr, axis1, axis2)
        assert np.array_equal(res_num, res_np)

    def test_emtpy_array(self):
        np_arr = np.array([])
        num_arr = num.array([])
        axis1 = 0
        axis2 = 0

        res_np = np.swapaxes(np_arr, axis1, axis2)
        res_num = num.swapaxes(num_arr, axis1, axis2)
        assert np.array_equal(res_num, res_np)


class TestSwapAxesModuleErrors:
    def setup_method(self):
        self.a = mk_seq_array(num, (3, 3))

    def test_a_none(self):
        msg = "has no attribute 'swapaxes'"
        with pytest.raises(AttributeError, match=msg):
            num.swapaxes(None, 0, 0)

    @pytest.mark.parametrize(
        "axes", ((3, 0), (0, 3)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_out_of_bound1(self, axes):
        axis1, axis2 = axes
        msg = "too large for swapaxes"
        with pytest.raises(ValueError, match=msg):
            num.swapaxes(self.a, axis1, axis2)

    @pytest.mark.parametrize(
        "axes", ((-4, 0), (0, -4)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_out_of_bound2(self, axes):
        axis1, axis2 = axes
        with pytest.raises(IndexError):
            num.swapaxes(self.a, axis1, axis2)

    @pytest.mark.parametrize(
        "axes", ((None, 0), (0, None)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_none(self, axes):
        axis1, axis2 = axes
        msg = "not supported between instances of 'NoneType' and 'int'"
        with pytest.raises(TypeError, match=msg):
            num.swapaxes(self.a, axis1, axis2)


class TestSwapAxesArrayMethod:
    def test_small(self):
        a_num = num.array(a)
        b = a.swapaxes(0, 1)
        b_num = a_num.swapaxes(0, 1)
        assert np.array_equal(b, b_num)

    def test_tall(self):
        a_tall = np.concatenate((a,) * 100)
        a_tall_num = num.array(a_tall)

        b_tall = a_tall.swapaxes(0, 1)
        b_tall_num = a_tall_num.swapaxes(0, 1)
        assert np.array_equal(b_tall, b_tall_num)

    def test_wide(self):
        a_wide = np.concatenate((a,) * 100, axis=1)
        a_wide_num = num.array(a_wide)

        b_wide = a_wide.swapaxes(0, 1)
        b_wide_num = a_wide_num.swapaxes(0, 1)
        assert np.array_equal(b_wide, b_wide_num)

    def test_big(self):
        a_tall = np.concatenate((a,) * 100)
        a_big = np.concatenate((a_tall,) * 100, axis=1)
        a_big_num = num.array(a_big)

        b_big = a_big.swapaxes(0, 1)
        b_big_num = a_big_num.swapaxes(0, 1)
        assert np.array_equal(b_big, b_big_num)

    @pytest.mark.parametrize(
        "axes",
        ((0, 0), (-3, 1), (0, 2), (-3, -2)),
        ids=lambda axes: f"(axes={axes})",
    )
    def test_axes(self, axes):
        shape = (3, 4, 5)
        np_arr = mk_seq_array(np, shape)
        num_arr = num.array(np_arr)
        axis1, axis2 = axes

        res_np_arr = np_arr.swapaxes(axis1, axis2)
        res_num_arr = num_arr.swapaxes(axis1, axis2)
        assert np.array_equal(res_num_arr, res_np_arr)

    def test_emtpy_array(self):
        np_arr = np.array([])
        num_arr = num.array([])
        axis1 = 0
        axis2 = 0

        res_np_arr = np_arr.swapaxes(axis1, axis2)
        res_num_arr = num_arr.swapaxes(axis1, axis2)
        assert np.array_equal(res_num_arr, res_np_arr)


class TestSwapAxesArrayMethodErrors:
    def setup_method(self):
        self.a = mk_seq_array(num, (3, 3))

    @pytest.mark.parametrize(
        "axes", ((3, 0), (0, 3)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_out_of_bound1(self, axes):
        axis1, axis2 = axes
        msg = "too large for swapaxes"
        with pytest.raises(ValueError, match=msg):
            self.a.swapaxes(axis1, axis2)

    @pytest.mark.parametrize(
        "axes", ((-4, 0), (0, -4)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_out_of_bound2(self, axes):
        axis1, axis2 = axes
        with pytest.raises(IndexError):
            self.a.swapaxes(axis1, axis2)

    @pytest.mark.parametrize(
        "axes", ((None, 0), (0, None)), ids=lambda axes: f"(axes={axes})"
    )
    def test_axes_none(self, axes):
        axis1, axis2 = axes
        msg = "not supported between instances of 'NoneType' and 'int'"
        with pytest.raises(TypeError, match=msg):
            self.a.swapaxes(axis1, axis2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
