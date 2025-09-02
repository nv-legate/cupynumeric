# Copyright 2025 NVIDIA Corporation
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

SHAPES = ((), (0,), (1,), (10,), (4, 5), (1, 4, 5))


class TestFromDLPack:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_default(self, shape: tuple[int, ...]) -> None:
        np_array = np.random.random(shape)
        num_array = num.from_dlpack(np_array)
        assert np.array_equal(np_array, num_array)

    def test_copy_true(self) -> None:
        np_array = np.ones((3, 4, 5))
        num_array = num.from_dlpack(np_array, copy=True)
        assert np.array_equal(np_array, num_array)

        np_array[1, 2, 3] = 10
        assert not np.array_equal(np_array, num_array)

    def test_copy_false(self) -> None:
        np_array = np.ones((3, 4, 5))
        num_array = num.from_dlpack(np_array, copy=False)
        assert np.array_equal(np_array, num_array)

        np_array[1, 2, 3] = 10
        assert np.array_equal(np_array, num_array)


class TestToDLPack:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_default(self, shape: tuple[int, ...]) -> None:
        np_array = np.random.random(shape)
        num_array = num.array(np_array)

        result = np.from_dlpack(num_array)
        assert np.array_equal(np_array, result)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
