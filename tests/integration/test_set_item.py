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


def test_basic():
    x = num.array([0, 0, 0])
    x[0] = 1
    x[1] = 2
    x[2] = 3

    assert x[0] == 1
    assert x[1] == 2
    assert x[2] == 3


def test_newaxis():
    arr = num.ones((4,))
    arr[None] = 1
    assert np.array_equal(arr, [1, 1, 1, 1])
    arr[None, :] = 2
    assert np.array_equal(arr, [2, 2, 2, 2])


ARRAYS_4_3_2_1_0 = [4 - num.arange(5), 4 - np.arange(5), [4, 3, 2, 1, 0]]


@pytest.mark.parametrize("arr", ARRAYS_4_3_2_1_0)
def test_scalar_ndarray_as_index(arr):
    offsets = num.arange(5)  # [0, 1, 2, 3, 4]
    offset = offsets[3]  # 3
    arr[offset] = -1
    arr[offset - 2 : offset] = [-1, -1]
    assert np.array_equal(arr, [4, -1, -1, -1, 0])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
