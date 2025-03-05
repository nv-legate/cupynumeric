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


def assert_individual(a, ew, ev):
    assert num.linalg.norm(ev, ord=np.inf) > 0

    ew_diag = num.array(np.diagflat(ew))
    a_ev = num.matmul(a, ev)
    ev_ew = num.matmul(ev, ew_diag)

    if ev_ew.dtype is np.dtype(np.complex64):
        rtol = 1e-02
        atol = 1e-04
    else:
        rtol = 1e-05
        atol = 1e-08

    assert num.allclose(a_ev, ev_ew, rtol=rtol, atol=atol)


def assert_result(a, ew, ev):
    m = a.shape[-1]
    if m == 0:
        return
    num_matrices = int(np.prod(a.shape) // (m * m))
    batch_view_a = a.reshape(num_matrices, m, m)
    batch_view_ew = ew.reshape(num_matrices, m)
    batch_view_ev = ev.reshape(num_matrices, m, m)

    for idx in range(num_matrices):
        assert_individual(
            batch_view_a[idx, :, :],
            batch_view_ew[idx, :],
            batch_view_ev[idx, :, :],
        )


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
        assert_result(arr_num, ew, ev)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    def test_arr_basic_complex(self, size, dtype):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_result(arr_num, ew, ev)

    @pytest.mark.parametrize("size", SIZES)
    @pytest.mark.parametrize("dtype", (np.int32, np.int64))
    def test_arr_basic_int(self, size, dtype):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_result(arr_num, ew, ev)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_arr_4d_real(self, size, dtype):
        arr_np = np.random.randint(-100, 100, size).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_result(arr_num, ew, ev)

    @pytest.mark.parametrize("size", SIZES_4D)
    @pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
    def test_arr_4d_complex(self, size, dtype):
        arr_np = (
            np.random.randint(-100, 100, size)
            + np.random.randint(-100, 100, size) * 1.0j
        ).astype(dtype)
        arr_num = num.array(arr_np)
        ew, ev = num.linalg.eig(arr_num)
        assert_result(arr_num, ew, ev)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
