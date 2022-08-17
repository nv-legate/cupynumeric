# Copyright 2021-2022 NVIDIA Corporation
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

import cunumeric as num

np.random.seed(42)


def test_extract():
    cnp = np.array(
        [1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0, 1]
    ).reshape(
        (6, 3)
    )  # noqa E501
    c = num.array(cnp)
    bnp = np.random.randn(6, 3)
    b = num.array(bnp)
    assert num.array_equal(num.extract(c, b), np.extract(cnp, bnp))


ARR = [
    [1, 54, 4, 4, 0, 45, 5, 58, 0, 9, 0, 4, 0, 0, 0, 5, 0, 1],
    [[1, 54, 4], [4, 0, 45], [5, 58, 0], [9, 0, 4], [0, 0, 0], [5, 0, 1]],
    [
        [[1, 54, 4], [4, 0, 45]],
        [[5, 58, 0], [9, 0, 4]],
        [[0, 0, 0], [5, 0, 1]],
    ],
    [[[1 + 2j, 54, 4], [4, 0 + 1j, 45]], [[5, 58, 0], [9, 0, 4]]],
    [[True, False], [True, True], [True, False]],
]


def check_extract(condition_np, arr_np):
    arr_num = num.array(arr_np)
    condition_num = num.array(condition_np)
    result_np = np.extract(condition_np, arr_np)
    result_num = num.extract(condition_num, arr_num)
    assert np.array_equal(result_np, result_num)


@pytest.mark.parametrize("arr", ARR, ids=str)
def test_extract_bool(arr):
    arr_np = np.array(arr)
    condition_np = arr_np != 0
    check_extract(condition_np, arr_np)
    check_extract(condition_np.flatten(), arr_np)
    check_extract(condition_np, arr_np.flatten())
    check_extract(condition_np.swapaxes(0, condition_np.ndim - 1), arr_np)
    check_extract(condition_np, arr_np.swapaxes(0, condition_np.ndim - 1))


@pytest.mark.parametrize("arr", ARR, ids=str)
def test_extract_nonzero(arr):
    arr_np = np.array(arr)
    condition_np = arr_np.copy()
    check_extract(condition_np, arr_np)
    check_extract(condition_np.flatten(), arr_np)
    check_extract(condition_np, arr_np.flatten())
    check_extract(condition_np.swapaxes(0, condition_np.ndim - 1), arr_np)
    check_extract(condition_np, arr_np.swapaxes(0, condition_np.ndim - 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
