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
from numpy.lib import NumpyVersion

import cupynumeric as num

arr_np = np.eye(4)
vec_np = np.arange(4).astype(np.float64)

arr_num = num.array(arr_np)
vec_num = num.array(vec_np)

indices = [0, 3, 1, 2]


class ArrayLike:
    # Class to test that we defer to unknown ArrayLikes
    def __array_function__(self, *args, **kwargs):
        return "deferred"

    def __array_ufunc__(self, *args, **kwargs):
        return "deferred"


def test_array_function_implemented():
    res_np = np.dot(arr_np, vec_np)
    res_num = np.dot(arr_num, vec_num)

    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, num.ndarray)  # implemented


def test_array_function_unimplemented():
    res_np = np.linalg.tensorsolve(arr_np, vec_np)
    res_num = np.linalg.tensorsolve(arr_num, vec_num)

    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, np.ndarray)  # unimplemented


def test_array_function_defer():
    assert np.concatenate([arr_num, ArrayLike()]) == "deferred"


def test_array_ufunc_through_array_op():
    assert np.array_equal(vec_num + vec_num, vec_np + vec_np)
    assert isinstance(vec_num + vec_np, num.ndarray)
    assert isinstance(vec_np + vec_num, num.ndarray)


def test_array_ufunc_call():
    res_np = np.add(vec_np, vec_np)
    res_num = np.add(vec_num, vec_num)

    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, num.ndarray)  # implemented


def test_array_ufunc_reduce():
    res_np = np.add.reduce(vec_np)
    res_num = np.add.reduce(vec_num)

    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, num.ndarray)  # implemented


def test_array_ufunc_accumulate():
    res_np = np.add.accumulate(vec_np)
    res_num = np.add.accumulate(vec_num)

    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, np.ndarray)  # unimplemented


def test_array_ufunc_reduceat():
    res_np = np.add.reduceat(vec_np, indices)
    res_num = np.add.reduceat(vec_num, indices)

    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, np.ndarray)  # unimplemented


def test_array_ufunc_outer():
    res_np = np.add.outer(vec_np, vec_np)
    res_num = np.add.outer(vec_num, vec_num)

    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, np.ndarray)  # unimplemented


def test_array_ufunc_at():
    res_np = np.full((4,), 42)
    res_num = num.full((4,), 42)

    np.add.at(res_np, indices, vec_np)
    np.add.at(res_num, indices, vec_num)

    assert np.array_equal(res_np, res_num)
    assert isinstance(res_num, num.ndarray)


def test_array_ufunc_defer():
    assert np.add(arr_num, ArrayLike()) == "deferred"
    assert np.add(arr_num, arr_num, out=ArrayLike()) == "deferred"
    if NumpyVersion(np.__version__) >= "1.25.0":
        # NumPy should also dispatch where
        assert np.add(arr_num, arr_num, where=ArrayLike()) == "deferred"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
