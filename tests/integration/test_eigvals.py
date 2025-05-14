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
from numpy.linalg.linalg import LinAlgError  # noqa: F401

import cupynumeric as num

SIZES = [
    (5, 5),
    (
        3,
        3,
    ),
    (2, 5, 5),
    (12, 3, 3),
    (1, 5, 5),
    (3, 1, 1),
    (
        10,
        2,
        2,
    ),
    (1, 4, 4),
    (1, 0, 0),
    (1, 1, 1),
]


SIZES_4D = [
    (3, 2, 5, 5),
    (1, 2, 5, 5),
    (4, 1, 5, 5),
    (2, 1, 0, 0),
]


def assert_all(ew_np, ew_num, sort=True):
    # convert to numpy
    ew_num = np.asarray(ew_num)

    # sort eigenvalues
    if sort:
        ew_np = np.sort(ew_np, axis=-1)
        ew_num = np.sort(ew_num, axis=-1)

    assert num.allclose(ew_num, ew_np, rtol=1e-5, atol=1e-4)


def assert_eigvals(a, ew):
    ew_np = np.linalg.eigvals(a)
    assert_all(ew_np, ew)


def assert_eigvalsh(a, uplo, ew):
    ew_np = np.linalg.eigvalsh(a, uplo)
    assert_all(ew_np, ew, False)


class TestEigvals(object):
    @pytest.mark.xfail
    def test_arr_none(self):
        res_np = np.linalg.eigvals(
            None
        )  # AxisError: axis -1 is out of bounds for array of dimension 0
        res_num = num.linalg.eigvals(
            None
        )  # AttributeError: 'NoneType' object has no attribute 'shape'
        assert np.equal(res_np, res_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("arr", ([], [[]], [[], []]))
    def test_arr_empty(self, arr):
        res_np = np.linalg.eigvals(arr)
        res_num = num.linalg.eigvals(arr)
        assert np.equal(res_np, res_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "arr", ([1], [[2]], [[2], [1]], [[[2], [1]], [[3], [4]]])
    )
    def atest_arr_dim_1(self, arr):
        res_np = np.linalg.eigvals(arr)
        res_num = num.linalg.eigvals(arr)
        assert np.equal(res_np, res_num)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_arr_basic_real(self, size, dtype):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvals(arr_num)
        assert_eigvals(arr_np, ew)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    def test_arr_basic_complex(self, size, dtype):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvals(arr_num)
        assert_eigvals(arr_np, ew)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.int32, np.int64))
    def test_arr_basic_int(self, size, dtype):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvals(arr_num)
        assert_eigvals(arr_np, ew)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_arr_4d_real(self, size, dtype):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvals(arr_num)
        assert_eigvals(arr_np, ew)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    def test_arr_4d_complex(self, size, dtype):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvals(arr_num)
        assert_eigvals(arr_np, ew)


class TestEigvalsh(object):
    @pytest.mark.xfail
    def test_arr_none(self):
        res_np = np.linalg.eigvalsh(
            None
        )  # AxisError: axis -1 is out of bounds for array of dimension 0
        res_num = num.linalg.eigvalsh(
            None
        )  # AttributeError: 'NoneType' object has no attribute 'shape'
        assert np.equal(res_np, res_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("arr", ([], [[]], [[], []]))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_empty(self, arr, uplo):
        res_np = np.linalg.eigvalsh(arr, uplo)
        res_num = num.linalg.eigvalsh(arr, uplo)
        assert np.equal(res_np, res_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "arr", ([1], [[2]], [[2], [1]], [[[2], [1]], [[3], [4]]])
    )
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def atest_arr_dim_1(self, arr, uplo):
        res_np = np.linalg.eigvalsh(arr, uplo)
        res_num = num.linalg.eigvalsh(arr, uplo)
        assert np.equal(res_np, res_num)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_basic_real(self, size, dtype, uplo):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvalsh(arr_num, uplo)
        assert_eigvalsh(arr_np, uplo, ew)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_basic_complex(self, size, dtype, uplo):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvalsh(arr_num, uplo)
        assert_eigvalsh(arr_np, uplo, ew)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.int32, np.int64))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_basic_int(self, size, dtype, uplo):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvalsh(arr_num, uplo)
        assert_eigvalsh(arr_np, uplo, ew)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_4d_real(self, size, dtype, uplo):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvalsh(arr_num, uplo)
        assert_eigvalsh(arr_np, uplo, ew)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_4d_complex(self, size, dtype, uplo):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew = num.linalg.eigvalsh(arr_num, uplo)
        assert_eigvalsh(arr_np, uplo, ew)

    def test_eigvals_1d_array(self) -> None:
        arr = np.array([1, 2, 3])
        msg = (
            r"1-dimensional array given. "
            "Array must be at least two-dimensional"
        )
        with pytest.raises(LinAlgError, match=msg):
            num.linalg.eigvals(arr)
        with pytest.raises(LinAlgError, match=msg):
            np.linalg.eig(arr)

    def test_eigvals_non_square(self) -> None:
        arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        msg = r"Last 2 dimensions of the array must be square"
        with pytest.raises(LinAlgError, match=msg):
            num.linalg.eigvals(arr)
        with pytest.raises(LinAlgError, match=msg):
            np.linalg.eig(arr)

    def test_eigvals_float16(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.float16)
        msg = r"array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.eigvals(arr)
        with pytest.raises(TypeError, match=msg):
            np.linalg.eig(arr)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
