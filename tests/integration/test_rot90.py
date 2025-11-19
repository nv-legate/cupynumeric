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


class TestRot90:
    def test_errors(self):
        with pytest.raises(ValueError):
            num.rot90(np.ones(4))
        with pytest.raises(ValueError):
            num.rot90(np.ones((2, 2, 2)), axes=(0, 1, 2))
        with pytest.raises(ValueError):
            num.rot90(np.ones((2, 2)), axes=(0, 2))
        with pytest.raises(ValueError):
            num.rot90(np.ones((2, 2)), axes=(1, 1))
        with pytest.raises(ValueError):
            num.rot90(np.ones((2, 2, 2)), axes=(-2, 1))

    @pytest.mark.parametrize(
        "shape,axes",
        [
            ((2, 2, 2), (-4, 0)),
            ((2, 2, 2), (0, -4)),
            ((2, 2, 2), (5, 10)),
            ((2, 2, 2, 2), (4, 1)),
            ((3, 3), (-3, 0)),
        ],
    )
    def test_axes_out_of_range(self, shape, axes) -> None:
        msg = r"out of range for array"
        with pytest.raises(ValueError, match=msg):
            num.rot90(np.ones(shape), axes=axes)

    @pytest.mark.parametrize("k", range(-3, 13))
    def test_basic(self, k: int) -> None:
        a = [[0, 1, 2], [3, 4, 5]]
        a_np = np.rot90(a, k=k)
        a_num = num.rot90(a, k=k)

        assert np.array_equal(a_np, a_num)

    def test_axes(self):
        a = np.ones((50, 40, 3))
        assert np.array_equal(num.rot90(a).shape, (40, 50, 3))
        assert np.array_equal(
            num.rot90(a, axes=(0, 2)), num.rot90(a, axes=(0, -1))
        )
        assert np.array_equal(
            num.rot90(a, axes=(1, 2)), num.rot90(a, axes=(-2, -1))
        )

    def test_inverse_axes(self):
        a = [[0, 1, 2], [3, 4, 5]]
        assert np.array_equal(
            num.rot90(num.rot90(a, axes=(0, 1)), axes=(1, 0)), a
        )

    def test_inverse_k(self):
        a = [[0, 1, 2], [3, 4, 5]]
        assert np.array_equal(
            num.rot90(a, k=1, axes=(1, 0)), num.rot90(a, k=-1, axes=(0, 1))
        )

    @pytest.mark.parametrize("axes", ((0, 1), (1, 0), (1, 2)))
    def test_rotation_axes(self, axes):
        a = np.arange(8).reshape((2, 2, 2))
        a_np = np.rot90(a, axes=(0, 1))
        a_num = num.rot90(a, axes=(0, 1))

        assert np.array_equal(a_np, a_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
