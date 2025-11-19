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

# roll tests adapted directly from numpy/_core/tests/test_numeric.py


class TestRoll:
    def test_roll1d(self):
        x = np.arange(10)
        xr_np = np.roll(x, 2)
        xr_num = num.roll(x, 2)
        assert np.array_equal(xr_num, xr_np)

    def test_roll2d_single_axis(self):
        x2 = np.reshape(np.arange(10), (2, 5))
        x2r_np = np.roll(x2, 1)
        x2r_num = num.roll(x2, 1)
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, 1, axis=0)
        x2r_num = num.roll(x2, 1, axis=0)
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, 1, axis=1)
        x2r_num = num.roll(x2, 1, axis=1)
        assert np.array_equal(x2r_num, x2r_np)

    def test_roll2d_multi_axis(self):
        x2 = np.reshape(np.arange(10), (2, 5))
        x2r_np = np.roll(x2, 1, axis=(0, 1))
        x2r_num = num.roll(x2, 1, axis=(0, 1))
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, (1, 0), axis=(0, 1))
        x2r_num = num.roll(x2, (1, 0), axis=(0, 1))
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, (-1, 0), axis=(0, 1))
        x2r_num = num.roll(x2, (-1, 0), axis=(0, 1))
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, (0, 1), axis=(0, 1))
        x2r_num = num.roll(x2, (0, 1), axis=(0, 1))
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, (0, -1), axis=(0, 1))
        x2r_num = num.roll(x2, (0, -1), axis=(0, 1))
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, (1, 1), axis=(0, 1))
        x2r_num = num.roll(x2, (1, 1), axis=(0, 1))
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, (-1, -1), axis=(0, 1))
        x2r_num = num.roll(x2, (-1, -1), axis=(0, 1))
        assert np.array_equal(x2r_num, x2r_np)

    def test_roll2d_same_axis_multiple_times(self):
        x2 = np.reshape(np.arange(10), (2, 5))
        x2r_np = np.roll(x2, 1, axis=(0, 0))
        x2r_num = num.roll(x2, 1, axis=(0, 0))
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, 1, axis=(1, 1))
        x2r_num = num.roll(x2, 1, axis=(1, 1))
        assert np.array_equal(x2r_num, x2r_np)

    def test_roll2d_multiple_turns(self):
        x2 = np.reshape(np.arange(10), (2, 5))
        x2r_np = np.roll(x2, 6, axis=1)
        x2r_num = num.roll(x2, 6, axis=1)
        assert np.array_equal(x2r_num, x2r_np)

        x2r_np = np.roll(x2, -4, axis=1)
        x2r_num = num.roll(x2, -4, axis=1)
        assert np.array_equal(x2r_num, x2r_np)

    def test_roll_empty(self):
        x = num.array([])
        assert np.array_equal(num.roll(x, 1), np.array([]))

    @pytest.mark.parametrize(
        "shift,axis",
        [
            ([[1, 2], [3, 4]], (0, 1)),
            ([[1, 2]], 0),
            (np.array([[1, 2], [3, 4]]), 0),
            ([[1], [2], [3]], 1),
        ],
    )
    def test_roll_invalid_shift_axis_broadcast(self, shift, axis) -> None:
        x = np.arange(24).reshape(2, 3, 4)
        msg = r"'shift' and 'axis' should be scalars or 1D sequences"
        with pytest.raises(ValueError, match=msg):
            num.roll(x, shift, axis=axis)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
