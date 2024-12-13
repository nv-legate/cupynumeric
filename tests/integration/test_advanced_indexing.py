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
from legate.core import LEGATE_MAX_DIM
from utils.generators import mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE, TWO_MAX_DIM_RANGE

import cupynumeric as num


@pytest.fixture
def arr_region():
    return num.full((5,), 42)[2:3]


@pytest.fixture
def arr_future():
    return num.full((1,), 42)


@pytest.fixture
def arr_empty1d():
    return num.full((0), 0)


idx_region_1d = num.zeros((3,), dtype=np.int64)[2:3]
idx_future_1d = num.zeros((1,), dtype=np.int64)
idx_region_0d = num.zeros((3,), dtype=np.int64)[2:3].reshape(())
idx_future_0d = num.zeros((3,), dtype=np.int64).max()
idx_empty_1d = num.array([], dtype=int)

val_region_1d = num.full((3,), -1)[2:3]
val_future_1d = num.full((1,), -1)
val_region_0d = num.full((3,), -1)[2:3].reshape(())
val_future_0d = num.full((3,), -1).max()


# We need fixtures for `arr` because the `set_item` tests modify their inputs.
# However, pytest_lazy_fixture is no longer supported, so we pass the fixture
# names and rely on request.getfixturevalue to retrieve the computed values
ARRS_FIXTURES = ("arr_region", "arr_future")
ARRS_EMPTY_1D_FIXTURES = ("arr_empty1d", "arr_region", "arr_future")
IDXS_0D = (idx_future_0d,)  # TODO: idx_region_0d fails
VALS_0D = (val_future_0d,)  # TODO: val_region_0d fails
IDXS_1D = (idx_region_1d, idx_future_1d)
VALS_1D = (val_region_1d, val_future_1d)
IDXS_EMPTY_1D = (idx_empty_1d,)
VALS_EMPTY_1D = (num.array([]),)


@pytest.mark.parametrize("idx", IDXS_0D)  # idx = 0
@pytest.mark.parametrize("arr", ARRS_FIXTURES)  # arr = [42]
def test_getitem_scalar_0d(arr, idx, request):
    arr = request.getfixturevalue(arr)
    assert np.array_equal(arr[idx], 42)


@pytest.mark.parametrize("val", VALS_0D)  # val = -1
@pytest.mark.parametrize("idx", IDXS_0D)  # idx = 0
@pytest.mark.parametrize("arr", ARRS_FIXTURES)  # arr = [42]
def test_setitem_scalar_0d(arr, idx, val, request):
    arr = request.getfixturevalue(arr)
    arr[idx] = val
    assert np.array_equal(arr, [-1])


@pytest.mark.parametrize("idx", IDXS_1D)  # idx = [0]
@pytest.mark.parametrize("arr", ARRS_FIXTURES)  # arr = [42]
def test_getitem_scalar_1d(arr, idx, request):
    arr = request.getfixturevalue(arr)
    assert np.array_equal(arr[idx], [42])


@pytest.mark.parametrize("val", VALS_1D)  # val = [-1]
@pytest.mark.parametrize("idx", IDXS_1D)  # idx = [0]
@pytest.mark.parametrize("arr", ARRS_FIXTURES)  # arr = [42]
def test_setitem_scalar_1d(arr, idx, val, request):
    arr = request.getfixturevalue(arr)
    arr[idx] = val
    assert np.array_equal(arr, [-1])


@pytest.mark.parametrize("idx", IDXS_EMPTY_1D)  # idx = []
@pytest.mark.parametrize("arr", ARRS_EMPTY_1D_FIXTURES)  # arr = [42], [5], []
def test_getitem_empty_1d(arr, idx, request):
    arr = request.getfixturevalue(arr)
    assert np.array_equal(arr[idx], [])


@pytest.mark.parametrize("idx", IDXS_EMPTY_1D)  # idx = []
@pytest.mark.parametrize("arr", ARRS_EMPTY_1D_FIXTURES)  # arr = []
@pytest.mark.parametrize("val", VALS_EMPTY_1D)  # val = []
def test_setitem_empty_1d(arr, idx, val, request):
    arr = request.getfixturevalue(arr)
    arr[idx] = val
    assert np.array_equal(arr[idx], [])


def mk_deferred_array(lib, shape):
    if np.prod(shape) != 0:
        return lib.ones(shape)
    # for shape (2,0,3,4): good_shape = (2,1,3,4)
    good_shape = tuple(max(1, dim) for dim in shape)
    # for shape (2,0,3,4): key = [:,[False],:,:]
    key = tuple([False] if dim == 0 else slice(None) for dim in shape)
    return lib.ones(good_shape)[key]


def gen_args():
    for arr_ndim in ONE_MAX_DIM_RANGE[:-1]:
        for idx_ndim in range(1, arr_ndim + 1):
            for zero_dim in range(arr_ndim):
                yield arr_ndim, idx_ndim, zero_dim


@pytest.mark.parametrize("arr_ndim,idx_ndim,zero_dim", gen_args())
def test_zero_size(arr_ndim, idx_ndim, zero_dim):
    arr_shape = tuple(0 if dim == zero_dim else 3 for dim in range(arr_ndim))
    np_arr = mk_deferred_array(np, arr_shape)
    num_arr = mk_deferred_array(num, arr_shape)
    idx_shape = arr_shape[:idx_ndim]
    val_shape = (
        arr_shape
        if idx_ndim == 1
        else (np.prod(idx_shape),) + arr_shape[idx_ndim:]
    )
    np_idx = np.ones(idx_shape, dtype=np.bool_)
    num_idx = num.ones(idx_shape, dtype=np.bool_)
    assert np.array_equal(np_arr[np_idx], num_arr[num_idx])

    np_val = np.random.random(val_shape)
    num_val = num.array(np_val)
    np_arr[np_idx] = np_val
    num_arr[num_idx] = num_val
    assert np.array_equal(np_arr, num_arr)


def test_empty_bool():
    # empty arrays and indices
    arr_np = np.array([[]])
    arr_num = num.array([[]])
    idx_np = np.array([[]], dtype=bool)
    idx_num = num.array([[]], dtype=bool)
    res_np = arr_np[idx_np]
    res_num = arr_num[idx_num]
    assert np.array_equal(res_np, res_num)

    res_np = res_np.reshape((0,))
    res_num = res_num.reshape((0,))

    # set_item
    val_np = np.array([])
    val_num = num.array([])
    arr_np[idx_np] = val_np
    arr_num[idx_num] = val_num
    assert np.array_equal(arr_np, arr_num)

    # empty output
    arr_np = np.array([[-1]])
    arr_num = num.array([[-1]])
    idx_np = np.array([[False]], dtype=bool)
    idx_num = num.array([[False]], dtype=bool)
    res_np = arr_np[idx_np]
    res_num = arr_num[idx_num]
    assert np.array_equal(res_np, res_num)

    arr_np[idx_np] = val_np
    arr_num[idx_num] = val_num
    assert np.array_equal(arr_np, arr_num)

    arr_np = np.array([[1, 2, 3], [2, 3, 4]])
    arr_num = num.array(arr_np)
    idx_np = np.array([False, False], dtype=bool)
    idx_num = num.array([False, False], dtype=bool)
    assert np.array_equal(arr_np[idx_np], arr_num[idx_num])

    assert np.array_equal(arr_np[idx_np, 0:0], arr_num[idx_num, 0:0])
    assert np.array_equal(arr_np[idx_np, 2:1], arr_num[idx_num, 2:1])
    arr_np[idx_np, 0:0] = 5
    arr_num[idx_num, 0:0] = 5
    assert np.array_equal(arr_np, arr_num)


def test_future_stores():
    # array is a future:
    arr_np = np.array([4])
    index_np = np.zeros(8, dtype=int)
    arr_num = num.array(arr_np)
    index_num = num.array(index_np)
    res_np = arr_np[index_np]
    res_num = arr_num[index_num]
    assert np.array_equal(res_np, res_num)

    # index and array and lhs are futures:
    res_np = arr_np[index_np[1]]
    res_num = arr_num[index_num[1]]
    assert np.array_equal(res_np, res_num)

    # all futures
    b_np = np.array([10, 11, 12])
    b_num = num.array(b_np)
    arr_np[index_np[1]] = b_np[0]
    arr_num[index_num[1]] = b_num[0]
    assert np.array_equal(arr_np, arr_num)

    # index and lhs are futures:
    arr_np = np.array([4, 3, 2, 1])
    arr_num = num.array(arr_np)
    res_np = arr_np[index_np[3]]
    res_num = arr_num[index_num[3]]
    assert np.array_equal(res_np, res_num)

    # rhs is a future
    arr_np[index_np[3]] = b_np[2]
    arr_num[index_num[3]] = b_num[2]
    assert np.array_equal(arr_np, arr_num)


def test():
    # tests on 1D input array:
    print("advanced indexing test 1")

    # a: simple 1D test
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    indx = np.array([1, 3, 5])
    res = x[indx]
    x_num = num.array(x)
    indx_num = num.array(indx)
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    # b: after base array transformation:
    xt = x[1:]
    xt_num = x_num[1:]
    res = xt[indx]
    res_num = xt_num[indx_num]
    assert np.array_equal(res, res_num)

    # c: after index array transformation:
    indxt = indx[1:]
    indxt_num = indx_num[1:]
    res = x[indxt]
    res_num = x_num[indxt_num]
    assert np.array_equal(res, res_num)

    # d: test in-place assignment with scalar:
    x[indx] = 13
    x_num[indx_num] = 13
    assert np.array_equal(x, x_num)

    # e: test in-place assignment with array:
    xt[indx] = np.array([3, 5, 7])
    xt_num[indx_num] = num.array([3, 5, 7])
    assert np.array_equal(xt, xt_num)
    assert np.array_equal(x, x_num)

    # f: test in-place assignment with transformed rhs array:
    b = np.array([3, 5, 7, 8])
    b_num = num.array([3, 5, 7, 8])
    bt = b[1:]
    bt_num = b_num[1:]
    x[indx] = bt
    x_num[indx_num] = bt_num
    assert np.array_equal(x, x_num)

    # g: test in-place assignment with transformed
    #    rhs and lhs arrays:
    b = np.array([3, 5, 7, 8])
    b_num = num.array([3, 5, 7, 8])
    b1 = b[1:]
    b1_num = b_num[1:]
    xt[indx] = b1
    xt_num[indx_num] = b1_num
    assert np.array_equal(xt, xt_num)
    assert np.array_equal(x, x_num)

    # h: in-place assignment with transformed index array:
    b = np.array([5, 7])
    b_num = num.array([5, 7])
    x[indxt] = b
    x_num[indxt_num] = b_num
    assert np.array_equal(x, x_num)

    # i: the case when index.ndim > input.ndim:
    index = np.array([[1, 0, 1, 3, 0, 0], [2, 4, 0, 4, 4, 4]])
    index_num = num.array(index)
    assert np.array_equal(x[index], x_num[index_num])

    # j: test for bool array of the same dimension
    index = np.array([True, False, False, True, True, False, True])
    index_num = num.array(index)
    assert np.array_equal(x[index], x_num[index_num])

    index = np.array([False] * 7)
    index_num = num.array(index)
    assert np.array_equal(x[index], x_num[index_num])

    # k: test in-place assignment fir the case when idx arr
    #    is 1d bool array:
    x[index] = 3
    x_num[index_num] = 3
    assert np.array_equal(x, x_num)

    # l: test when type of a base array is different from int:
    x_float = x.astype(float)
    x_num_float = x_num.astype(float)
    index = np.array([[1, 0, 1, 3, 0, 0], [2, 4, 0, 4, 4, 4]])
    index_num = num.array(index)
    assert np.array_equal(x_float[index], x_num_float[index_num])

    # m: test when type of the index array is not int64
    index = np.array([1, 3, 5], dtype=np.int16)
    index_num = num.array(index)
    assert np.array_equal(x[index], x_num[index_num])

    # n: the case when rhs is a different type
    x[index] = 3.5
    x_num[index_num] = 3.5
    assert np.array_equal(x, x_num)

    # o: the case when rhs is an array of different type
    b = np.array([2.1, 3.3, 7.2])
    b_num = num.array(b)
    x[index] = b
    x_num[index_num] = b_num
    assert np.array_equal(x, x_num)

    # p: in-place assignment where some indices point to the
    # same location:
    index = np.array([2, 4, 0, 4, 4, 4])
    index_num = num.array(index)
    x[index] = 0
    x_num[index_num] = 0
    assert np.array_equal(x, x_num)

    # q: in-place assignment in the case when broadcast is needed:
    index = np.array([[1, 4, 3], [2, 0, 5]])
    index_num = num.array(index)
    x[index] = np.array([[1, 2, 3]])
    x_num[index_num] = num.array([[1, 2, 3]])
    assert np.array_equal(x, x_num)

    # r negative indices
    indx = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    indx_num = num.array(indx)
    res = x[indx]
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    # s index array as a future
    index = np.array([3])
    index_num = num.array(index)
    res = x[index]
    res_num = x_num[index]
    assert np.array_equal(res, res_num)

    # Nd cases
    print("advanced indexing test 2")

    x = mk_seq_array(np, (2, 3, 4, 5))
    x_num = mk_seq_array(num, (2, 3, 4, 5))
    xt = x.transpose(
        (
            1,
            0,
            2,
            3,
        )
    )
    xt_num = x_num.transpose(
        (
            1,
            0,
            2,
            3,
        )
    )

    # a: 1d index  array passed to a different indices:
    indx = np.array([1, 1])
    indx_num = num.array(indx)
    res = x[indx]
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    res = xt[indx]
    res_num = xt_num[indx_num]
    assert np.array_equal(res, res_num)

    res = x[:, :, indx]
    res_num = x_num[:, :, indx_num]
    assert np.array_equal(res, res_num)

    res = xt[:, :, indx]
    res_num = xt_num[:, :, indx_num]
    assert np.array_equal(res, res_num)

    res = x[:, :, :, indx]
    res_num = x_num[:, :, :, indx_num]
    assert np.array_equal(res, res_num)

    res = xt[:, :, :, indx]
    res_num = xt_num[:, :, :, indx_num]
    assert np.array_equal(res, res_num)

    res = x[:, indx, :]
    res_num = x_num[:, indx_num, :]
    assert np.array_equal(res, res_num)

    res = xt[:, indx, :]
    res_num = xt_num[:, indx_num, :]
    assert np.array_equal(res, res_num)

    # test with negative indices:
    indx = np.array([-1, 1])
    indx_num = num.array(indx)
    res = x[indx]
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    # b : 2 1d index arrays passed
    indx0 = np.array([1, 1])
    indx1 = np.array([1, 0])
    indx0_num = num.array(indx0)
    indx1_num = num.array(indx1)
    res = x[indx0, indx1]
    res_num = x_num[indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    res = xt[indx0, indx1]
    res_num = xt_num[indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    res = x[:, indx0, indx1]
    res_num = x_num[:, indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    res = xt[:, indx0, indx1]
    res_num = xt_num[:, indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    # test with negative indices:
    indx0 = np.array([1, -1])
    indx1 = np.array([-1, 0])
    indx0_num = num.array(indx0)
    indx1_num = num.array(indx1)
    res = x[indx0, indx1]
    res_num = x_num[indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    # c:  2 index arrays passed in a sparse way:
    res = x[:, [0, 1], :, [0, 1]]
    res_num = x_num[:, [0, 1], :, [0, 1]]
    assert np.array_equal(res, res_num)

    res = xt[:, [0, 1], :, [0, 1]]
    res_num = xt_num[:, [0, 1], :, [0, 1]]
    assert np.array_equal(res, res_num)

    res = x[[0, 1], :, [0, 1], 1:]
    res_num = x_num[[0, 1], :, [0, 1], 1:]
    assert np.array_equal(res, res_num)

    res = xt[[0, 1], :, [0, 1], 1:]
    res_num = xt_num[[0, 1], :, [0, 1], 1:]
    assert np.array_equal(res, res_num)

    res = x[:, [0, 1], :, 1:]
    res_num = x_num[:, [0, 1], :, 1:]
    assert np.array_equal(res, res_num)

    res = xt[:, [0, 1], :, 1:]
    res_num = xt_num[:, [0, 1], :, 1:]
    assert np.array_equal(res, res_num)

    x[[0, 1], [0, 1]] = 11
    x_num[[0, 1], [0, 1]] = 11
    assert np.array_equal(x, x_num)

    x[[0, 1], :, [0, 1]] = 12
    x_num[[0, 1], :, [0, 1]] = 12
    assert np.array_equal(x, x_num)

    x[[0, 1], 1:3, [0, 1]] = 3.5
    x_num[[0, 1], 1:3, [0, 1]] = 3.5
    assert np.array_equal(x, x_num)

    x[1:2, :, [0, 1]] = 7
    x_num[1:2, :, [0, 1]] = 7
    assert np.array_equal(x, x_num)

    # d: newaxis is passed along with array:

    res = x[..., [1, 0]]
    res_num = x_num[..., [1, 0]]
    assert np.array_equal(res, res_num)

    x[..., [1, 0]] = 8
    x_num[..., [1, 0]] = 8
    assert np.array_equal(res, res_num)

    xt = x.transpose(
        (
            1,
            3,
            0,
            2,
        )
    )
    xt_num = x_num.transpose(
        (
            1,
            3,
            0,
            2,
        )
    )
    res = xt[..., [0, 1], 1:]
    res_num = xt_num[..., [0, 1], 1:]
    assert np.array_equal(res, res_num)

    res = x[..., [0, 1], [1, 1]]
    res_num = x_num[..., [0, 1], [1, 1]]
    assert np.array_equal(res, res_num)

    # e: index arrays that have different shape:
    indx0 = np.array([1, 1])
    indx1 = np.array([[1, 0], [1, 0]])
    indx0_num = num.array(indx0)
    indx1_num = num.array(indx1)
    res = x[indx0, indx1]
    res_num = x_num[indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    res = xt[indx0, indx1]
    res_num = xt_num[indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    res = x[indx0, indx1, indx0, indx1]
    res_num = x_num[indx0_num, indx1_num, indx0_num, indx1_num]
    assert np.array_equal(res, res_num)

    res = x[indx0, :, indx1]
    res_num = x_num[indx0_num, :, indx1_num]
    assert np.array_equal(res, res_num)

    res = xt[:, indx0, indx1, 1:]
    res_num = xt_num[:, indx0_num, indx1_num, 1:]
    assert np.array_equal(res, res_num)

    # f: single boolean array passed:
    indx_bool = np.array([True, False])
    indx_bool_num = num.array(indx_bool)
    res = x[indx_bool]
    res_num = x_num[indx_bool_num]
    assert np.array_equal(res, res_num)

    indx_bool = np.array([True, False, True])
    indx_bool_num = num.array(indx_bool)
    res = x[:, indx_bool]
    res_num = x_num[:, indx_bool_num]
    assert np.array_equal(res, res_num)

    # on the transposed base
    indx_bool = np.array([True, False, True])
    indx_bool_num = num.array(indx_bool)
    res = xt[indx_bool]
    res_num = xt_num[indx_bool_num]
    assert np.array_equal(res, res_num)

    indx_bool = np.array([True, False, True, False, False])
    indx_bool_num = num.array(indx_bool)
    res = x[..., indx_bool]
    res_num = x_num[..., indx_bool_num]
    assert np.array_equal(res, res_num)

    indx1_bool = np.array([True, False])
    indx1_bool_num = num.array(indx1_bool)
    indx2_bool = np.array([True, False, True, True])
    indx2_bool_num = num.array(indx2_bool)
    res = x[indx1_bool, :, indx2_bool]
    res_num = x_num[indx1_bool_num, :, indx2_bool_num]
    assert np.array_equal(res, res_num)

    res = x[indx1_bool, 1, indx2_bool]
    res_num = x_num[indx1_bool_num, 1, indx2_bool_num]
    assert np.array_equal(res, res_num)

    # g: boolean array with the same shape is passed to x:
    indx = x % 2
    indx = indx.astype(bool)
    indx_num = num.array(indx)
    res = x[indx]
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    # h: inplace assignment with bool arays
    z = x
    z_num = x_num
    z[indx] = 1
    z_num[indx_num] = 1
    assert np.array_equal(z, z_num)

    indx_bool = np.array([True, False, True])
    indx_bool_num = num.array(indx_bool)
    z[:, indx_bool] = 5
    z_num[:, indx_bool_num] = 5
    assert np.array_equal(z, z_num)

    # i: two bool array of the same shape are passed:
    x = mk_seq_array(
        np,
        (
            3,
            4,
            3,
            4,
        ),
    )
    x_num = mk_seq_array(
        num,
        (
            3,
            4,
            3,
            4,
        ),
    )
    indx = np.array(
        [
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, True],
        ]
    )
    indx_num = num.array(indx)
    res = x[indx, indx]
    res_num = x_num[indx_num, indx_num]
    assert np.array_equal(res, res_num)

    # call to advanced indexing task:
    # indx.ndim< arrya.ndim
    indx_num = num.array(indx)
    res = x[indx]
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    # call to advanced indexing task:
    # indx.ndim< arrya.ndim
    indx_num = num.array(indx)
    res = x[:, :, indx]
    res_num = x_num[:, :, indx_num]
    assert np.array_equal(res, res_num)

    if LEGATE_MAX_DIM > 4:
        x = mk_seq_array(
            np,
            (
                3,
                4,
                5,
                3,
                4,
            ),
        )
        x_num = mk_seq_array(
            num,
            (
                3,
                4,
                5,
                3,
                4,
            ),
        )
        # 2 bool arrays separated by scalar
        res = x[indx, 1, indx]
        res_num = x_num[indx_num, 1, indx_num]
        assert np.array_equal(res, res_num)

        # 2 bool arrays separated by :
        res = x[indx, :, indx]
        res_num = x_num[indx_num, :, indx_num]
        assert np.array_equal(res, res_num)

    # j: 2 bool arrays should be broadcasted:
    x = mk_seq_array(
        np,
        (
            3,
            4,
            3,
            4,
        ),
    )
    x_num = mk_seq_array(
        num,
        (
            3,
            4,
            3,
            4,
        ),
    )
    res = x[indx, [True, False, False]]
    res_num = x_num[indx_num, [True, False, False]]
    assert np.array_equal(res, res_num)

    # 2d bool array not at the first index:
    indx = np.full((4, 3), True)
    indx_num = num.array(indx)
    res = x[:, indx]
    res_num = x_num[:, indx]
    assert np.array_equal(res, res_num)

    # 3: testing mixed type of the arguments passed:

    # a: bool and index arrays
    x = mk_seq_array(
        np,
        (
            2,
            3,
            4,
            5,
        ),
    )
    x_num = mk_seq_array(
        num,
        (
            2,
            3,
            4,
            5,
        ),
    )
    res = x[[1, 1], [False, True, False]]
    res_num = x_num[[1, 1], [False, True, False]]
    assert np.array_equal(res, res_num)

    res = x[[1, 1], :, [False, True, False, True]]
    res_num = x_num[[1, 1], :, [False, True, False, True]]
    assert np.array_equal(res, res_num)

    # set item with mixed indices
    x[1, :, [False, True, False, True]] = 129
    x_num[1, :, [False, True, False, True]] = 129
    assert np.array_equal(x, x_num)

    # set item with mixed indices
    x[:, [False, True, False], 1] = 111
    x_num[:, [False, True, False], 1] = 111
    assert np.array_equal(x, x_num)

    x[..., [False, True, False, True, False]] = 200
    x_num[..., [False, True, False, True, False]] = 200
    assert np.array_equal(x, x_num)

    # b: combining basic and advanced indexing schemes
    ind0 = np.array([1, 1])
    ind0_num = num.array(ind0)
    res = x[ind0, :, -1]
    res_num = x_num[ind0_num, :, -1]
    assert np.array_equal(res, res_num)

    res = x[ind0, :, 1:3]
    res_num = x_num[ind0_num, :, 1:3]
    assert np.array_equal(res, res_num)

    res = x[1, :, ind0]
    res_num = x_num[1, :, ind0_num]
    assert np.array_equal(res, res_num)

    x = mk_seq_array(np, (3, 4, 5, 6))
    x_num = mk_seq_array(num, (3, 4, 5, 6))
    res = x[[0, 1], [0, 1], :, 2]
    res_num = x_num[[0, 1], [0, 1], :, 2]
    assert np.array_equal(res, res_num)

    res = x[..., [0, 1], 2]
    res_num = x_num[..., [0, 1], 2]
    assert np.array_equal(res, res_num)

    res = x[:, [0, 1], :, -1]
    res_num = x_num[:, [0, 1], :, -1]
    assert np.array_equal(res, res_num)

    res = x[:, [0, 1], :, 1:]
    res_num = x_num[:, [0, 1], :, 1:]
    assert np.array_equal(res, res_num)

    # c: transformed base or index or rhs:
    z = x[:, 1:]
    z_num = x_num[:, 1:]
    indx = np.array([1, 1])
    indx_num = num.array(indx)
    res = z[indx]
    res_num = z_num[indx_num]
    assert np.array_equal(res, res_num)

    indx = np.array([1, 1, 0])
    indx_num = num.array(indx)
    indx = indx[1:]
    indx_num = indx_num[1:]
    res = z[1, indx]
    res_num = z_num[1, indx_num]
    assert np.array_equal(res, res_num)

    b = np.ones((2, 3, 6, 5))
    b_num = num.array(b)
    b = b.transpose((0, 1, 3, 2))
    b_num = b_num.transpose((0, 1, 3, 2))
    z[indx] = b
    z_num[indx_num] = b_num
    assert np.array_equal(z, z_num)

    # d: shape mismatch case:
    x = np.array(
        [
            [0.38, -0.16, 0.38, -0.41, -0.04],
            [-0.47, -0.01, -0.18, -0.5, -0.49],
            [0.02, 0.4, 0.33, 0.33, -0.13],
        ]
    )
    x_num = num.array(x)

    indx = np.ones((2, 2, 2), dtype=int)
    indx_num = num.array(indx)
    res = x[indx]
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    x = np.ones(
        (
            3,
            4,
        ),
        dtype=int,
    )
    x_num = num.array(x)
    ind = np.full((4,), True)
    ind_num = num.array(ind)
    res = x[:, ind]
    res_num = x_num[:, ind_num]
    assert np.array_equal(res, res_num)

    if LEGATE_MAX_DIM > 7:
        x = np.ones((2, 3, 4, 5, 3, 4))
        ind1 = np.full((3, 4), True)
        ind2 = np.full((3, 4), True)
        x_num = num.array(x)
        ind1_num = num.array(ind1)
        ind2_num = num.array(ind2)
        res = x[:, ind1, :, ind2]
        res_num = x[:, ind1_num, :, ind2_num]
        assert np.array_equal(res, res_num)

    # e: type mismatch case:
    x = np.ones((3, 4))
    x_num = num.array(x)
    ind = np.full((3,), 1, dtype=np.int32)
    ind_num = num.array(ind)
    res = x[ind, ind]
    res_num = x_num[ind_num, ind_num]
    assert np.array_equal(res, res_num)

    x = np.ones((3, 4), dtype=float)
    x_num = num.array(x)
    ind = np.arange(3)
    ind_num = num.array(ind)
    res = x[ind, ind]
    res_num = x_num[ind_num, ind_num]
    assert np.array_equal(res, res_num)

    x[ind, ind] = 5
    x_num[ind_num, ind_num] = 5
    assert np.array_equal(x, x_num)

    # some additional tests for bool index arrays:
    # 2d:
    x = mk_seq_array(
        np,
        (
            3,
            4,
        ),
    )
    x_num = mk_seq_array(
        num,
        (
            3,
            4,
        ),
    )
    indx = np.array(
        [
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, True],
        ]
    )
    indx_num = num.array(indx)
    res = x[indx]
    res_num = x_num[indx_num]
    assert np.array_equal(res, res_num)

    # we do less than LEGATE_MAX_DIM becasue the dimension will be increased by
    # 1 when passig 2d index array
    for ndim in TWO_MAX_DIM_RANGE[:-1]:
        a_shape = tuple(np.random.randint(2, 5) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)
        # check when N of index arrays == N of dims
        num_tuple_of_indices = tuple()
        np_tuple_of_indices = tuple()
        for i in range(ndim):
            i_shape = (2, 4)
            idx_arr_np = mk_seq_array(np, i_shape) % np_array.shape[i]
            idx_arr_num = num.array(idx_arr_np)
            np_tuple_of_indices += (idx_arr_np,)
            num_tuple_of_indices += (idx_arr_num,)
        assert np.array_equal(
            np_array[np_tuple_of_indices], num_array[num_tuple_of_indices]
        )
        # check when N of index arrays == N of dims
        i_shape = (2, 2)
        idx_arr_np = mk_seq_array(np, i_shape) % np_array.shape[0]
        idx_arr_num = num.array(idx_arr_np)
        assert np.array_equal(np_array[idx_arr_np], num_array[idx_arr_num])
        # test in-place assignment
        np_array[idx_arr_np] = 2
        num_array[idx_arr_num] = 2
        assert np.array_equal(num_array, np_array)
        idx_arr_np = np.array([[1, 0, 1], [1, 1, 0]])
        idx_arr_num = num.array(idx_arr_np)
        assert np.array_equal(
            np_array[:, idx_arr_np], num_array[:, idx_arr_num]
        )
        # test in-place assignment
        np_array[:, idx_arr_np] = 3
        num_array[:, idx_arr_num] = 3
        assert np.array_equal(num_array, np_array)
        if ndim > 2:
            assert np.array_equal(
                np_array[1, :, idx_arr_np], num_array[1, :, idx_arr_num]
            )
            assert np.array_equal(
                np_array[:, idx_arr_np, idx_arr_np],
                num_array[:, idx_arr_num, idx_arr_num],
            )
        if ndim > 3:
            assert np.array_equal(
                np_array[:, idx_arr_np, :, idx_arr_np],
                num_array[:, idx_arr_num, :, idx_arr_num],
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
