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

import re

import numpy as np
import pytest
from numpy.linalg.linalg import LinAlgError  # noqa: F401

import cupynumeric as num

SIZES = [
    (5, 5),
    (3, 3),
    (2, 5, 5),
    (12, 3, 3),
    (1, 5, 5),
    (3, 1, 1),
    (10, 2, 2),
    (1, 4, 4),
    (1, 0, 0),
    (1, 1, 1),
]


SIZES_4D = [(3, 2, 5, 5), (1, 2, 5, 5), (4, 1, 5, 5), (2, 1, 0, 0)]


def assert_all(a_np, ew_np, ev_np, ew_num, ev_num, sort=True):
    # convert to numpy
    ew_num = np.asarray(ew_num)
    ev_num = np.asarray(ev_num)

    # sort by eigenvalues
    if sort:
        ew_np_ind = np.argsort(ew_np, axis=-1)
        ew_num_ind = np.argsort(ew_num, axis=-1)

        ew_np = np.take_along_axis(ew_np, ew_np_ind, axis=-1)
        ew_num = np.take_along_axis(ew_num, ew_num_ind, axis=-1)

        ev_np = np.take_along_axis(ev_np, ew_np_ind[..., None, :], axis=-1)
        ev_num = np.take_along_axis(ev_num, ew_num_ind[..., None, :], axis=-1)

    assert num.allclose(ew_num, ew_np, rtol=1e-5, atol=1e-4)

    # check EVs
    def assert_evs(a, ew, ev):
        if len(ew) > 0:
            assert np.linalg.norm(ev, ord=np.inf) != 0
            ew_diag = np.diagflat(ew)
            a_ev = np.matmul(a, ev)
            ev_ew = np.matmul(ev, ew_diag)
            assert num.allclose(a_ev, ev_ew, rtol=1e-5, atol=1e-4)

    for ind in np.ndindex(a_np.shape[:-2]):
        assert_evs(a_np[ind], ew_num[ind], ev_num[ind])


def assert_eig(a, ew, ev):
    ew_np, ev_np = np.linalg.eig(a)
    assert_all(a, ew_np, ev_np, ew, ev)


def assert_eigh(a, uplo, ew, ev):
    ew_np, ev_np = np.linalg.eigh(a, uplo)

    # symmetrize A to eventually check A*ev = ev*ew
    def symmetrize_a(a, is_complex, uplo_l):
        n = a.shape[1]
        for i in range(n):
            if is_complex:
                a[i, i] = a[i, i].real
            for j in range(i + 1, n):
                if uplo_l:
                    a[i, j] = a[j, i].conj() if is_complex else a[j, i]
                else:
                    a[j, i] = a[i, j].conj() if is_complex else a[i, j]

    is_complex = np.issubdtype(a.dtype, np.complexfloating)
    for ind in np.ndindex(a.shape[:-2]):
        symmetrize_a(a[ind], is_complex, uplo == "L")

    assert_all(a, ew_np, ev_np, ew, ev, False)


class TestEig(object):
    @pytest.mark.xfail
    def test_arr_none(self):
        res_np = np.linalg.eig(
            None
        )  # AxisError: axis -1 is out of bounds for array of dimension 0
        res_num = num.linalg.eig(
            None
        )  # AttributeError: 'NoneType' object has no attribute 'shape'
        assert np.equal(res_np, res_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("arr", ([], [[]], [[], []]))
    def test_arr_empty(self, arr):
        res_np = np.linalg.eig(arr)
        res_num = num.linalg.eig(arr)
        assert np.equal(res_np, res_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "arr", ([1], [[2]], [[2], [1]], [[[2], [1]], [[3], [4]]])
    )
    def atest_arr_dim_1(self, arr):
        res_np = np.linalg.eig(arr)
        res_num = num.linalg.eig(arr)
        assert np.equal(res_np, res_num)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_arr_basic_real(self, size, dtype):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_eig(arr_np, ew, ev)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    def test_arr_basic_complex(self, size, dtype):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_eig(arr_np, ew, ev)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.int32, np.int64))
    def test_arr_basic_int(self, size, dtype):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_eig(arr_np, ew, ev)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_arr_4d_real(self, size, dtype):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_eig(arr_np, ew, ev)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    def test_arr_4d_complex(self, size, dtype):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_eig(arr_np, ew, ev)

    def test_eig_linalgerror(self) -> None:
        arr_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        arr_num = num.array(arr_np)
        msg = r"Last 2 dimensions of the array must be square"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.eig(arr_num)
        with pytest.raises(np.linalg.LinAlgError, match=msg):
            np.linalg.eig(arr_np)


class TestEigh(object):
    @pytest.mark.xfail
    def test_arr_none(self):
        res_np = np.linalg.eigh(
            None
        )  # AxisError: axis -1 is out of bounds for array of dimension 0
        res_num = num.linalg.eigh(
            None
        )  # AttributeError: 'NoneType' object has no attribute 'shape'
        assert np.equal(res_np, res_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("arr", ([], [[]], [[], []]))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_empty(self, arr, uplo):
        res_np = np.linalg.eigh(arr, uplo)
        res_num = num.linalg.eigh(arr, uplo)
        assert np.equal(res_np, res_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "arr", ([1], [[2]], [[2], [1]], [[[2], [1]], [[3], [4]]])
    )
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def atest_arr_dim_1(self, arr, uplo):
        res_np = np.linalg.eigh(arr, uplo)
        res_num = num.linalg.eigh(arr, uplo)
        assert np.equal(res_np, res_num)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_basic_real(self, size, dtype, uplo):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eigh(arr_num, uplo)
        assert_eigh(arr_np, uplo, ew, ev)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_basic_complex(self, size, dtype, uplo):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eigh(arr_num, uplo)
        assert_eigh(arr_np, uplo, ew, ev)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.int32, np.int64))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_basic_int(self, size, dtype, uplo):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eigh(arr_num, uplo)
        assert_eigh(arr_np, uplo, ew, ev)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_4d_real(self, size, dtype, uplo):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eigh(arr_num, uplo)
        assert_eigh(arr_np, uplo, ew, ev)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    @pytest.mark.parametrize("uplo", ("L", "U"))
    def test_arr_4d_complex(self, size, dtype, uplo):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eigh(arr_num, uplo)
        assert_eigh(arr_np, uplo, ew, ev)

    def test_eig_1d_array(self) -> None:
        arr = np.array([1, 2, 3])
        msg = (
            r"1-dimensional array given. "
            "Array must be at least two-dimensional"
        )
        with pytest.raises(LinAlgError, match=re.escape(msg)):
            num.linalg.eig(arr)
        with pytest.raises(LinAlgError, match=re.escape(msg)):
            np.linalg.eig(arr)

    def test_eig_non_square(self) -> None:
        arr = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 matrix
        msg = r"Last 2 dimensions of the array must be square"
        with pytest.raises(LinAlgError, match=msg):
            num.linalg.eig(arr)
        with pytest.raises(LinAlgError, match=msg):
            np.linalg.eig(arr)

    def test_eig_float16(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype=np.float16)
        msg = r"array type float16 is unsupported in linalg"
        with pytest.raises(TypeError, match=msg):
            num.linalg.eig(arr)
        with pytest.raises(TypeError, match=msg):
            np.linalg.eig(arr)

    def test_eigh_linalgerror(self) -> None:
        arr_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        arr_num = num.array(arr_np)
        msg = r"Last 2 dimensions of the array must be square"
        with pytest.raises(num.linalg.LinAlgError, match=msg):
            num.linalg.eigh(arr_num)
        with pytest.raises(np.linalg.LinAlgError, match=msg):
            np.linalg.eigh(arr_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
