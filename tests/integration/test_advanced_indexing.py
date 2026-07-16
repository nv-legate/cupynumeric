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
    xt = x.transpose((1, 0, 2, 3))
    xt_num = x_num.transpose((1, 0, 2, 3))

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

    xt = x.transpose((1, 3, 0, 2))
    xt_num = x_num.transpose((1, 3, 0, 2))
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
    x = mk_seq_array(np, (3, 4, 3, 4))
    x_num = mk_seq_array(num, (3, 4, 3, 4))
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
        x = mk_seq_array(np, (3, 4, 5, 3, 4))
        x_num = mk_seq_array(num, (3, 4, 5, 3, 4))
        # 2 bool arrays separated by scalar
        res = x[indx, 1, indx]
        res_num = x_num[indx_num, 1, indx_num]
        assert np.array_equal(res, res_num)

        # 2 bool arrays separated by :
        res = x[indx, :, indx]
        res_num = x_num[indx_num, :, indx_num]
        assert np.array_equal(res, res_num)

    # j: 2 bool arrays should be broadcasted:
    x = mk_seq_array(np, (3, 4, 3, 4))
    x_num = mk_seq_array(num, (3, 4, 3, 4))
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
    x = mk_seq_array(np, (2, 3, 4, 5))
    x_num = mk_seq_array(num, (2, 3, 4, 5))
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

    x = np.ones((3, 4), dtype=int)
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
    x = mk_seq_array(np, (3, 4))
    x_num = mk_seq_array(num, (3, 4))
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


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE[:-1])
@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.int32, np.int64]
)
def test_einsum_path_different_dimensions(ndim, dtype):
    """Test einsum path for different array dimensions and float dtypes."""
    # Create test array with unique values for verification
    shape = tuple(4 + i for i in range(ndim))  # e.g., (4,5,6,7,8,9) for 6D
    np_array = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    num_array = num.array(np_array)

    # Test each possible mask_axis position
    for mask_axis in range(ndim):
        axis_size = shape[mask_axis]

        # Test with boolean mask
        bool_mask_np = np.array(
            [True, False, True] + [False] * (axis_size - 3)
        )[:axis_size]
        bool_mask_num = num.array(bool_mask_np)

        # Create indexing key with boolean mask
        key_np = tuple(
            bool_mask_np if i == mask_axis else slice(None)
            for i in range(ndim)
        )
        key_num = tuple(
            bool_mask_num if i == mask_axis else slice(None)
            for i in range(ndim)
        )

        # Compare results
        expected = np_array[key_np]
        actual = num_array[key_num]
        assert np.array_equal(actual, expected), (
            f"Boolean mask failed for ndim={ndim}, mask_axis={mask_axis}"
        )

        # Test with integer indices
        int_indices_np = np.array([0, 2, min(3, axis_size - 1)])[
            : min(3, axis_size)
        ]
        int_indices_num = num.array(int_indices_np)

        # Create indexing key with integer indices
        key_np = tuple(
            int_indices_np if i == mask_axis else slice(None)
            for i in range(ndim)
        )
        key_num = tuple(
            int_indices_num if i == mask_axis else slice(None)
            for i in range(ndim)
        )

        # Compare results
        expected = np_array[key_np]
        actual = num_array[key_num]
        assert np.array_equal(actual, expected), (
            f"Integer indices failed for ndim={ndim}, mask_axis={mask_axis}"
        )


def test_einsum_path_edge_cases():
    """Test edge cases for einsum path."""
    # Test 1D array (minimal case)
    np_array_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    num_array_1d = num.array(np_array_1d)

    bool_mask = np.array([True, False, True, False, True])
    bool_mask_num = num.array(bool_mask)

    expected = np_array_1d[bool_mask]
    actual = num_array_1d[bool_mask_num]
    assert np.array_equal(actual, expected)

    # Test with all True mask
    all_true_mask = np.array([True, True, True, True, True])
    all_true_mask_num = num.array(all_true_mask)

    expected = np_array_1d[all_true_mask]
    actual = num_array_1d[all_true_mask_num]
    assert np.array_equal(actual, expected)

    # Test with all False mask (empty result)
    all_false_mask = np.array([False, False, False, False, False])
    all_false_mask_num = num.array(all_false_mask)

    expected = np_array_1d[all_false_mask]
    actual = num_array_1d[all_false_mask_num]
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE[:-1])
def test_einsum_path_different_mask_positions(ndim):
    """Test einsum path with mask at different positions."""
    shape = tuple(3 + i for i in range(ndim))
    np_array = np.random.rand(*shape).astype(np.float32)
    num_array = num.array(np_array)

    for mask_axis in range(ndim):
        axis_size = shape[mask_axis]

        # Create a mask that selects roughly half the elements
        mask_indices = np.arange(0, axis_size, 2)  # [0, 2, 4, ...]
        int_mask_np = mask_indices
        int_mask_num = num.array(int_mask_np)

        # Build the indexing tuple
        key_np = tuple(
            int_mask_np if i == mask_axis else slice(None) for i in range(ndim)
        )
        key_num = tuple(
            int_mask_num if i == mask_axis else slice(None)
            for i in range(ndim)
        )

        # Compare results
        expected = np_array[key_np]
        actual = num_array[key_num]
        assert np.array_equal(actual, expected), (
            f"Failed for ndim={ndim}, mask_axis={mask_axis}"
        )


def test_einsum_path_shape_mismatch():
    """Test that einsum path handles shape mismatches correctly."""
    np_array = np.random.rand(4, 5, 6).astype(np.float32)
    num_array = num.array(np_array)

    # Create mask with wrong shape (should not trigger einsum path)
    wrong_shape_mask_np = np.array(
        [True, False, True]
    )  # length 3, but axis 0 has size 4
    wrong_shape_mask_num = num.array(wrong_shape_mask_np)

    # This should work with regular indexing (will raise appropriate error or handle correctly)
    with pytest.raises((IndexError, ValueError)):
        _ = np_array[wrong_shape_mask_np, :, :]

    with pytest.raises((IndexError, ValueError)):
        _ = num_array[wrong_shape_mask_num, :, :]


def test_advanced_indexing_int_mask_different_shapes():
    """Test advanced indexing with integer mask arrays having different shapes from original_array[axis]."""

    # Test Case 1: Use 1D array (ndim=1) to bypass einsum path (einsum requires ndim > 1)
    np_array_1d = np.arange(6, dtype=np.int32)
    num_array_1d = num.array(np_array_1d)

    # Integer mask with fewer elements than array size
    int_mask_short_np = np.array([0, 2, 4], dtype=np.int32)
    int_mask_short_num = num.array(int_mask_short_np)

    expected = np_array_1d[int_mask_short_np]
    actual = num_array_1d[int_mask_short_num]
    assert np.array_equal(actual, expected), (
        "1D array with int mask should work"
    )

    # Test Case 2: Use mixed indexing (array + non-slice(None)) to bypass einsum path
    np_array_2d = np.arange(20, dtype=np.int32).reshape(5, 4)
    num_array_2d = num.array(np_array_2d)

    int_mask_np = np.array([0, 2], dtype=np.int32)
    int_mask_num = num.array(int_mask_np)

    # Mix array with specific slice (not slice(None)) - should bypass einsum
    key_np = (int_mask_np, slice(1, 3))
    key_num = (int_mask_num, slice(1, 3))

    expected = np_array_2d[key_np]
    actual = num_array_2d[key_num]
    assert np.array_equal(actual, expected), "Mixed indexing should work"

    # Test Case 3: Multiple array indices - should bypass einsum path
    row_indices_np = np.array([0, 2, 4], dtype=np.int32)
    col_indices_np = np.array([1, 3, 0], dtype=np.int32)
    row_indices_num = num.array(row_indices_np)
    col_indices_num = num.array(col_indices_np)

    key_np = (row_indices_np, col_indices_np)
    key_num = (row_indices_num, col_indices_num)

    expected = np_array_2d[key_np]
    actual = num_array_2d[key_num]
    assert np.array_equal(actual, expected), (
        "Multiple array indices should work"
    )

    # Test Case 4: Empty integer mask
    int_mask_empty_np = np.array([], dtype=np.int32)
    int_mask_empty_num = num.array(int_mask_empty_np)

    expected = np_array_1d[int_mask_empty_np]
    actual = num_array_1d[int_mask_empty_num]
    assert np.array_equal(actual, expected), "Empty int mask should work"

    # Test Case 5: Integer mask with valid out-of-sequence indices
    int_mask_seq_np = np.array([4, 1, 3], dtype=np.int32)
    int_mask_seq_num = num.array(int_mask_seq_np)

    expected = np_array_1d[int_mask_seq_np]
    actual = num_array_1d[int_mask_seq_num]
    assert np.array_equal(actual, expected), (
        "Out-of-sequence int mask should work"
    )

    # Test Case 6: Integer mask with repeated indices
    int_mask_repeat_np = np.array([0, 2, 0, 4, 2], dtype=np.int32)
    int_mask_repeat_num = num.array(int_mask_repeat_np)

    expected = np_array_1d[int_mask_repeat_np]
    actual = num_array_1d[int_mask_repeat_num]
    assert np.array_equal(actual, expected), (
        "Repeated int mask indices should work"
    )

    # Test Case 7: Integer mask with negative indices
    int_mask_negative_np = np.array(
        [-1, -3, -2], dtype=np.int32
    )  # Should convert to [5, 3, 4]
    int_mask_negative_num = num.array(int_mask_negative_np)

    expected = np_array_1d[int_mask_negative_np]
    actual = num_array_1d[int_mask_negative_num]
    assert np.array_equal(actual, expected), (
        "Negative int mask indices should work"
    )

    # Test Case 8: Mixed positive and negative indices
    int_mask_mixed_np = np.array(
        [0, -1, 2, -2], dtype=np.int32
    )  # [0, 5, 2, 4] for array of size 6
    int_mask_mixed_num = num.array(int_mask_mixed_np)

    expected = np_array_1d[int_mask_mixed_np]
    actual = num_array_1d[int_mask_mixed_num]
    assert np.array_equal(actual, expected), (
        "Mixed positive/negative int mask indices should work"
    )

    # Test Case 9: Invalid negative indices (too negative) should raise IndexError
    int_mask_invalid_neg_np = np.array(
        [0, -7], dtype=np.int32
    )  # -7 is out of bounds for array of size 6
    int_mask_invalid_neg_num = num.array(int_mask_invalid_neg_np)

    with pytest.raises(IndexError):
        _ = np_array_1d[int_mask_invalid_neg_np]

    with pytest.raises(IndexError):
        _ = num_array_1d[int_mask_invalid_neg_num]


def test_einsum_path_with_negative_indices():
    """Test that einsum path correctly handles negative indices."""
    # Create a 3D test array that should trigger einsum path
    np_array = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
    num_array = num.array(np_array)

    # Test Case 1: Negative indices on axis 0 (size 4)
    int_mask_neg_np = np.array(
        [-1, -3], dtype=np.int32
    )  # Should convert to [3, 1]
    int_mask_neg_num = num.array(int_mask_neg_np)

    key_np = (int_mask_neg_np, slice(None), slice(None))
    key_num = (int_mask_neg_num, slice(None), slice(None))

    expected = np_array[key_np]
    actual = num_array[key_num]
    assert np.array_equal(actual, expected), (
        "Einsum path should handle negative indices on axis 0"
    )

    # Test Case 2: Mixed positive and negative indices
    int_mask_mixed_np = np.array(
        [0, -1, 1], dtype=np.int32
    )  # [0, 3, 1] for axis of size 4
    int_mask_mixed_num = num.array(int_mask_mixed_np)

    key_np = (int_mask_mixed_np, slice(None), slice(None))
    key_num = (int_mask_mixed_num, slice(None), slice(None))

    expected = np_array[key_np]
    actual = num_array[key_num]
    assert np.array_equal(actual, expected), (
        "Einsum path should handle mixed positive/negative indices"
    )


def test_einsum_path_axis_validation():
    """Test that einsum path correctly validates mask_axis bounds."""
    # This test specifically targets the axis validation logic in _advanced_indexing_using_einsum

    # Create test array
    np_array = np.arange(12, dtype=np.float32).reshape(3, 4)  # 2D array
    num_array = num.array(np_array)

    # Create a simple integer mask that would trigger einsum path
    int_mask_np = np.array([0, 2], dtype=np.int32)
    int_mask_num = num.array(int_mask_np)

    # Test Case 1: Negative axis (should work - converts -1 to 1 for 2D array)
    key_np = (slice(None), int_mask_np)  # Last axis
    key_num = (slice(None), int_mask_num)

    expected = np_array[key_np]
    actual = num_array[key_num]
    assert np.array_equal(actual, expected), (
        "Negative axis should be converted correctly"
    )

    # Test Case 2: Valid positive axis
    key_np = (int_mask_np, slice(None))  # First axis
    key_num = (int_mask_num, slice(None))

    expected = np_array[key_np]
    actual = num_array[key_num]
    assert np.array_equal(actual, expected), "Valid positive axis should work"


def test_advanced_indexing_int_mask_no_einsum():
    """Test advanced indexing with integer masks that bypass einsum path due to shape incompatibility."""
    # Create test arrays where einsum path should NOT be triggered due to shape mismatch
    np_array = np.arange(20, dtype=np.float32).reshape(5, 4)
    num_array = num.array(np_array)

    # Test with 1D array that has mixed slices and arrays (should bypass einsum)
    mixed_indices_np = np.array([0, 3], dtype=np.int32)
    mixed_indices_num = num.array(mixed_indices_np)

    # Mix of array index and specific slice (not slice(None)) - should bypass einsum
    key_np = (mixed_indices_np, slice(1, 3))
    key_num = (mixed_indices_num, slice(1, 3))

    expected = np_array[key_np]
    actual = num_array[key_num]
    assert np.array_equal(actual, expected), (
        "Mixed indexing with array and non-full slice should work"
    )

    # Test with multiple array indices (should bypass einsum)
    row_indices_np = np.array([0, 2, 4], dtype=np.int32)
    col_indices_np = np.array([1, 3, 0], dtype=np.int32)
    row_indices_num = num.array(row_indices_np)
    col_indices_num = num.array(col_indices_np)

    key_np = (row_indices_np, col_indices_np)
    key_num = (row_indices_num, col_indices_num)

    expected = np_array[key_np]
    actual = num_array[key_num]
    assert np.array_equal(actual, expected), (
        "Multiple array indices should work"
    )


@pytest.mark.parametrize("n", [5, 500])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
class TestGeneralPathGather:
    """Tests that exercise the general (ZIP + gather) path with various sizes."""

    def test_two_1d_indices(self, n: int, dtype: np.dtype) -> None:
        rng = np.random.default_rng(42)
        data_np = np.arange(n * n, dtype=dtype).reshape(n, n)
        rows_np = rng.integers(0, n, size=n)
        cols_np = rng.integers(0, n, size=n)

        data_num = num.array(data_np)
        rows_num = num.array(rows_np)
        cols_num = num.array(cols_np)

        expected = data_np[rows_np, cols_np]
        actual = data_num[rows_num, cols_num]
        assert np.array_equal(expected, actual)

    def test_broadcast_indices(self, n: int, dtype: np.dtype) -> None:
        rng = np.random.default_rng(43)
        m = max(n // 5, 2)
        data_np = np.arange(n * n, dtype=dtype).reshape(n, n)
        rows_np = rng.integers(0, n, size=(m, 1))
        cols_np = rng.integers(0, n, size=(1, m))

        data_num = num.array(data_np)
        rows_num = num.array(rows_np)
        cols_num = num.array(cols_np)

        expected = data_np[rows_np, cols_np]
        actual = data_num[rows_num, cols_num]
        assert np.array_equal(expected, actual)

    def test_mixed_slice_and_arrays(self, n: int, dtype: np.dtype) -> None:
        rng = np.random.default_rng(44)
        k = max(n // 10, 2)
        data_np = np.arange(n * n * k, dtype=dtype).reshape(n, n, k)
        idx0_np = rng.integers(0, n, size=k)
        idx1_np = rng.integers(0, k, size=k)

        data_num = num.array(data_np)
        idx0_num = num.array(idx0_np)
        idx1_num = num.array(idx1_np)

        expected = data_np[idx0_np, :, idx1_np]
        actual = data_num[idx0_num, :, idx1_num]
        assert np.array_equal(expected, actual)

    def test_noncontiguous_sparse_indices(
        self, n: int, dtype: np.dtype
    ) -> None:
        rng = np.random.default_rng(45)
        k = max(n // 10, 2)
        data_np = np.arange(k * n * k, dtype=dtype).reshape(k, n, k)
        idx0_np = rng.integers(0, k, size=k)
        idx1_np = rng.integers(0, k, size=k)

        data_num = num.array(data_np)
        idx0_num = num.array(idx0_np)
        idx1_num = num.array(idx1_np)

        expected = data_np[idx0_np, :, idx1_np]
        actual = data_num[idx0_num, :, idx1_num]
        assert np.array_equal(expected, actual)

    def test_negative_indices(self, n: int, dtype: np.dtype) -> None:
        rng = np.random.default_rng(46)
        data_np = np.arange(n * n, dtype=dtype).reshape(n, n)
        rows_np = rng.integers(-n, n, size=n)
        cols_np = rng.integers(-n, n, size=n)

        data_num = num.array(data_np)
        rows_num = num.array(rows_np)
        cols_num = num.array(cols_np)

        expected = data_np[rows_np, cols_np]
        actual = data_num[rows_num, cols_num]
        assert np.array_equal(expected, actual)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
class TestBoolMaskSetArrayRhs:
    """``a[bool_mask] = b`` with non-scalar ``b``.

    Single-process bool-mask SET with ``b.size > 1`` is routed through
    ``General`` (ZIP + scatter); multi-process uses the bool fast path
    (ADVANCED_INDEXING + scatter). These tests lock behavioral equivalence
    with NumPy for both paths.

    Scalar-rhs bool SET is already covered elsewhere (PUTMASK fast path).
    """

    def test_1d_full_shape(self, dtype: np.dtype) -> None:
        rng = np.random.default_rng(101)
        n = 64
        a_np = rng.integers(0, 100, size=n).astype(dtype)
        mask_np = rng.random(n) < 0.4
        b_np = rng.integers(-50, 50, size=int(mask_np.sum())).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[mask_np] = b_np
        a_num[num.array(mask_np)] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_2d_full_shape(self, dtype: np.dtype) -> None:
        # mask.shape == a.shape; result selects scalars (1-D rhs).
        rng = np.random.default_rng(102)
        a_np = rng.integers(0, 100, size=(8, 12)).astype(dtype)
        mask_np = rng.random((8, 12)) < 0.3
        b_np = rng.integers(-50, 50, size=int(mask_np.sum())).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[mask_np] = b_np
        a_num[num.array(mask_np)] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_2d_leading_axis(self, dtype: np.dtype) -> None:
        # mask.shape == a.shape[:1]; each True selects a row of length
        # a.shape[1]. Exercises the leading-axes slab pattern.
        rng = np.random.default_rng(103)
        a_np = rng.integers(0, 100, size=(10, 5)).astype(dtype)
        mask_np = rng.random(10) < 0.5
        n_true = int(mask_np.sum())
        b_np = rng.integers(-50, 50, size=(n_true, 5)).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[mask_np] = b_np
        a_num[num.array(mask_np)] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_3d_leading_axes(self, dtype: np.dtype) -> None:
        # 2-D mask on 3-D array; rhs is (n_true, trailing_dim).
        rng = np.random.default_rng(104)
        a_np = rng.integers(0, 100, size=(4, 5, 6)).astype(dtype)
        mask_np = rng.random((4, 5)) < 0.4
        n_true = int(mask_np.sum())
        b_np = rng.integers(-50, 50, size=(n_true, 6)).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[mask_np] = b_np
        a_num[num.array(mask_np)] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_bool_on_leading_axis_tuple_key(self, dtype: np.dtype) -> None:
        # a[mask, :] = b — BoolMask(transpose_index=0), tuple key form.
        rng = np.random.default_rng(108)
        a_np = rng.integers(0, 100, size=(10, 6)).astype(dtype)
        mask_np = rng.random(10) < 0.5
        n_true = int(mask_np.sum())
        b_np = rng.integers(-50, 50, size=(n_true, 6)).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[mask_np, :] = b_np
        a_num[num.array(mask_np), :] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_bool_on_trailing_axis(self, dtype: np.dtype) -> None:
        # a[:, mask] = b — BoolMask(transpose_index=1), 2-D column SET.
        # k>0 fix: nonzero + ZIP scatter in original space (no transposed-view
        # materialization).
        rng = np.random.default_rng(109)
        a_np = rng.integers(0, 100, size=(10, 8)).astype(dtype)
        mask_np = rng.random(8) < 0.5
        n_true = int(mask_np.sum())
        b_np = rng.integers(-50, 50, size=(10, n_true)).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[:, mask_np] = b_np
        a_num[:, num.array(mask_np)] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_bool_on_middle_axis(self, dtype: np.dtype) -> None:
        # a[:, mask, :] = b — bool with slice(None) co-keys.
        # Routes through BoolMask (transpose axis 1 to front, scatter back).
        rng = np.random.default_rng(105)
        a_np = rng.integers(0, 100, size=(4, 8, 3)).astype(dtype)
        mask_np = rng.random(8) < 0.5
        n_true = int(mask_np.sum())
        b_np = rng.integers(-50, 50, size=(4, n_true, 3)).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[:, mask_np, :] = b_np
        a_num[:, num.array(mask_np), :] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_all_true_mask(self, dtype: np.dtype) -> None:
        rng = np.random.default_rng(106)
        n = 30
        a_np = rng.integers(0, 100, size=n).astype(dtype)
        mask_np = np.ones(n, dtype=bool)
        b_np = rng.integers(-50, 50, size=n).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[mask_np] = b_np
        a_num[num.array(mask_np)] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_empty_mask_no_op(self, dtype: np.dtype) -> None:
        # All-False mask: nothing to scatter; array unchanged.
        a_np = np.arange(20, dtype=dtype)
        mask_np = np.zeros(20, dtype=bool)
        b_np = np.array([], dtype=dtype)

        a_num = num.array(a_np.copy())
        a_np[mask_np] = b_np
        a_num[num.array(mask_np)] = num.array(b_np)
        assert np.array_equal(a_np, a_num)

    def test_larger_2d(self, dtype: np.dtype) -> None:
        # Larger problem to give multi-GPU partitioning something to chew on.
        rng = np.random.default_rng(107)
        a_np = rng.integers(0, 1000, size=(64, 64)).astype(dtype)
        mask_np = rng.random((64, 64)) < 0.25
        b_np = rng.integers(-500, 500, size=int(mask_np.sum())).astype(dtype)

        a_num = num.array(a_np.copy())
        a_np[mask_np] = b_np
        a_num[num.array(mask_np)] = num.array(b_np)
        assert np.array_equal(a_np, a_num)


def test_bool_set_dtype_conversion() -> None:
    """``a[mask] = b`` with mismatched dtypes — ``ndarray.__setitem__``
    converts ``b`` to ``a.dtype`` before the scatter, so a float64 rhs
    assigned into an int64 array is silently truncated, matching NumPy."""
    rng = np.random.default_rng(201)
    a_np = rng.integers(0, 100, size=20).astype(np.int64)
    mask_np = np.zeros(20, dtype=bool)
    mask_np[::3] = True
    # Non-scalar float rhs assigned into int array (silently truncates).
    b_np = np.linspace(0.5, 5.5, int(mask_np.sum()), dtype=np.float64)

    a_num = num.array(a_np.copy())
    a_np[mask_np] = b_np
    a_num[num.array(mask_np)] = num.array(b_np)
    assert np.array_equal(a_np, a_num)


def test_bool_set_dtype_conversion_scalar_rhs() -> None:
    """Scalar-rhs variant of the dtype-conversion contract — PUTMASK
    fast path with mismatched scalar dtype."""
    a_np = np.zeros(10, dtype=np.int32)
    mask_np = np.array([True, False] * 5)
    rhs = np.float64(7.9)

    a_num = num.array(a_np.copy())
    a_np[mask_np] = rhs
    a_num[num.array(mask_np)] = rhs
    assert np.array_equal(a_np, a_num)


def test_bool_col_set_scalar_rhs() -> None:
    """a[:, mask] = scalar — BoolMask k=1 SET with scalar rhs.

    k>0 fix: nonzero + ZIP scatter in original array space avoids the
    O(n²) transposed-view materialization that ``_perform_scatter``
    previously triggered for transformed destinations.
    """
    rng = np.random.default_rng(202)
    a_np = rng.integers(0, 100, size=(12, 10)).astype(np.float64)
    mask_np = np.array(
        [True, False, True, True, False, False, True, False, True, False]
    )
    scalar = 0.0

    a_num = num.array(a_np.copy())
    a_np[:, mask_np] = scalar
    a_num[:, num.array(mask_np)] = scalar
    assert np.array_equal(a_np, a_num)


def test_int_index_scatter_0d_rhs() -> None:
    """``a[indices] = scalar_0d`` via General scatter with a 0-D rhs.

    On single-GPU, _lower_to_strategy promotes the 0-D rhs to 1-D for the
    _can_skip_transformed_index_copy decision, then calls execute_set with
    the original 0-D value.  Verifies the scatter produces the correct result
    and does not crash or silently write wrong values.
    """
    rng = np.random.default_rng(301)
    a_np = rng.integers(0, 100, size=(8, 6)).astype(np.float32)
    rows_np = rng.integers(0, 8, size=5)
    cols_np = rng.integers(0, 6, size=5)
    scalar = np.float32(42.0)

    a_num = num.array(a_np.copy())
    a_np[rows_np, cols_np] = scalar
    a_num[num.array(rows_np), num.array(cols_np)] = scalar
    assert np.array_equal(a_np, a_num)


def test_int_index_scatter_0d_rhs_1d() -> None:
    """1-D variant: ``a[indices] = scalar_0d`` on a 1-D array."""
    rng = np.random.default_rng(302)
    a_np = rng.integers(0, 100, size=20).astype(np.int64)
    idx_np = rng.integers(0, 20, size=7)
    scalar = np.int64(-1)

    a_num = num.array(a_np.copy())
    a_np[idx_np] = scalar
    a_num[num.array(idx_np)] = scalar
    assert np.array_equal(a_np, a_num)


def test_integer_indexing_out_of_bounds() -> None:
    arr = num.arange(12, dtype=np.float32).reshape(3, 4)
    idx = num.array([0, 999], dtype=np.int64)
    with pytest.raises(IndexError):
        _ = arr[idx, :]


def test_too_many_index_arrays() -> None:
    arr = num.arange(12).reshape(3, 4)
    idx1 = num.array([0])
    idx2 = num.array([1])
    idx3 = num.array([2])
    idx4 = num.array([0])  # 4 arrays for 2D array - too many!
    expect_msg = r"wrong number of index arrays passed"
    with pytest.raises(ValueError, match=expect_msg):
        # This should try to use 4 index arrays on a 2D array
        _ = arr._thunk._zip_indices(
            0,
            (idx1._thunk, idx2._thunk, idx3._thunk, idx4._thunk),
            check_bounds=True,
        )


def test_boolean_array_dimension_mismatch() -> None:
    arr = num.arange(12).reshape(3, 4)  # 2D array, ndim=2
    bool_idx = num.ones((4, 3), dtype=bool)  # 2D bool array

    # All arrays are DeferredArray since eager mode removed
    with pytest.raises(ValueError, match="Boolean array has .* dimensions"):
        _ = arr[:, bool_idx]


def test_bool_leading_mask_too_many_dims_raises() -> None:
    # (mask2d, :, :) on a 3-D array indexes 2 + 2 = 4 dims > 3 -> IndexError,
    # like NumPy. Pins the hoisted too-many-dims guard in
    # _prepare_boolean_array_indexing (the bool-on-leading-axis path).
    arr = num.arange(24).reshape(2, 3, 4)
    mask2d = num.ones((2, 3), dtype=bool)
    with pytest.raises(IndexError):
        _ = arr[mask2d, :, :]


def test_bare_bool_mask_too_many_dims_raises() -> None:
    # Bare a[mask3d] on a 2-D array now raises IndexError like the (mask3d,)
    # tuple spelling and NumPy (previously raised ValueError).
    arr = num.arange(12).reshape(3, 4)
    mask3d = num.ones((3, 4, 2), dtype=bool)
    with pytest.raises(IndexError):
        _ = arr[mask3d]


def test_bool_ellipsis_matches_numpy() -> None:
    # a[..., mask] must behave exactly like a[:, mask] — Ellipsis is normalized
    # to slice(None) co-keys in _prepare_boolean_array_indexing.
    np_a = np.arange(12, dtype=np.float64).reshape(3, 4)
    num_a = num.array(np_a)
    np_mask = np.array([True, False, True, False])
    num_mask = num.array(np_mask)

    # GET
    assert np.array_equal(np_a[..., np_mask], np.array(num_a[..., num_mask]))

    # SET (scalar rhs)
    np_a2, num_a2 = np_a.copy(), num.array(np_a.copy())
    np_a2[..., np_mask] = -1.0
    num_a2[..., num_mask] = -1.0
    assert np.array_equal(np_a2, np.array(num_a2))


def test_bool_ellipsis_multidim_mask_matches_numpy() -> None:
    # a[..., mask2d] == a[mask2d] (NumPy): a same-rank 2-D mask consumes both
    # axes, so the Ellipsis fills zero slices (not a[:, mask2d], which is an
    # IndexError).  Pins the multidimensional-mask branch of _expand_ellipsis.
    np_a = np.arange(6).reshape(2, 3)
    num_a = num.array(np_a)
    np_mask = np.array([[True, False, True], [False, True, False]])
    num_mask = num.array(np_mask)

    assert np.array_equal(np_a[..., np_mask], np.array(num_a[..., num_mask]))


@pytest.mark.parametrize(
    "flag",
    [True, False, np.bool_(True), np.bool_(False)],
    ids=["py_true", "py_false", "np_true", "np_false"],
)
def test_bool_scalar_get_matches_numpy(flag: bool | np.bool_) -> None:
    np_a = np.arange(24).reshape(2, 3, 4)
    num_a = num.array(np_a)

    keys = [
        flag,
        (..., flag),
        (slice(None), flag),
        (None, flag),
        (flag, None),
        (0, flag),
    ]
    for key in keys:
        np_result = np_a[key]
        num_result = num_a[key]
        assert np_result.shape == num_result.shape
        assert np.array_equal(np_result, np.array(num_result))


def test_bool_scalar_get_returns_copy() -> None:
    source = num.arange(6).reshape(2, 3)
    result = source[True]
    result[...] = -1
    assert np.array_equal(np.array(source), np.arange(6).reshape(2, 3))


@pytest.mark.parametrize("flag", [True, False], ids=["true", "false"])
def test_bool_scalar_mixed_advanced_get(flag: bool) -> None:
    np_a = np.arange(24).reshape(2, 3, 4)
    num_a = num.array(np_a)
    np_idx = np.array([0])
    num_idx = num.array(np_idx)

    key_pairs = [
        ((flag, np_idx), (flag, num_idx)),
        ((np_idx, flag), (num_idx, flag)),
        ((slice(None), flag, np_idx), (slice(None), flag, num_idx)),
        ((flag, slice(None), np_idx), (flag, slice(None), num_idx)),
    ]
    for np_key, num_key in key_pairs:
        np_result = np_a[np_key]
        num_result = num_a[num_key]
        assert np_result.shape == num_result.shape
        assert np.array_equal(np_result, np.array(num_result))


def test_false_scalar_mixed_advanced_broadcast_error() -> None:
    np_a = np.arange(24).reshape(2, 3, 4)
    num_a = num.array(np_a)
    np_idx = np.array([0, 1])
    num_idx = num.array(np_idx)

    with pytest.raises(IndexError):
        _ = np_a[False, np_idx]
    with pytest.raises((IndexError, ValueError)):
        _ = num_a[False, num_idx]


@pytest.mark.parametrize("flag", [True, False], ids=["true", "false"])
def test_bool_0d_array_get_matches_numpy(flag: bool) -> None:
    np_a = np.arange(6).reshape(2, 3)
    num_a = num.array(np_a)
    np_key = np.array(flag)
    num_key = num.array(flag)

    np_idx = np.array([0])
    num_idx = num.array(np_idx)
    key_pairs = (
        (np_key, num_key),
        ((..., np_key), (..., num_key)),
        ((np_key, np_idx), (num_key, num_idx)),
    )
    for np_index, num_index in key_pairs:
        np_result = np_a[np_index]
        num_result = num_a[num_index]
        assert np_result.shape == num_result.shape
        assert np.array_equal(np_result, np.array(num_result))


@pytest.mark.parametrize("flag", [True, False], ids=["true", "false"])
def test_bool_scalar_set_result_shaped_rhs(flag: bool) -> None:
    np_a = np.arange(6).reshape(2, 3)
    num_a = num.array(np_a)
    key = (..., flag)
    rhs = np.arange(np_a[key].size).reshape(np_a[key].shape)

    np_a[key] = rhs
    num_a[key] = num.array(rhs)
    assert np.array_equal(np_a, np.array(num_a))


def test_bool_scalar_set_excess_leading_singleton() -> None:
    np_a = np.arange(6).reshape(2, 3)
    num_a = num.array(np_a)
    key = (..., True)
    selection_shape = np_a[key].shape
    rhs = np.arange(np.prod(selection_shape)).reshape((1,) + selection_shape)

    np_a[key] = rhs
    num_a[key] = num.array(rhs)
    assert np.array_equal(np_a, np.array(num_a))


@pytest.mark.parametrize("flag", [True, False], ids=["true", "false"])
def test_bool_scalar_set_excess_leading_non_singleton_raises(
    flag: bool,
) -> None:
    np_a = np.arange(6).reshape(2, 3)
    num_a = num.array(np_a)
    key = (..., flag)
    selection_shape = np_a[key].shape
    rhs_shape = (2,) + selection_shape[:-1] + (1,)

    with pytest.raises(ValueError):
        np_a[key] = np.ones(rhs_shape)
    with pytest.raises(ValueError):
        num_a[key] = num.ones(rhs_shape)


@pytest.mark.parametrize("flag", [True, False], ids=["true", "false"])
def test_bool_0d_array_set_result_shaped_rhs(flag: bool) -> None:
    np_a = np.arange(6).reshape(2, 3)
    num_a = num.array(np_a)
    np_key = np.array(flag)
    num_key = num.array(flag)
    rhs = np.arange(np_a[np_key].size).reshape(np_a[np_key].shape)

    np_a[np_key] = rhs
    num_a[num_key] = num.array(rhs)
    assert np.array_equal(np_a, np.array(num_a))


@pytest.mark.parametrize("flag", [True, False], ids=["true", "false"])
def test_bool_scalar_indexes_0d_array(flag: bool) -> None:
    # Use distinct input scalars because scalar futures may be cached by value.
    initial = 7 if flag else 5
    np_a = np.array(initial)
    num_a = num.array(initial)

    np_result = np_a[flag]
    num_result = num_a[flag]
    assert np_result.shape == num_result.shape
    assert np.array_equal(np_result, np.array(num_result))

    rhs = np.arange(np_result.size).reshape(np_result.shape)
    np_a[flag] = rhs
    num_a[flag] = num.array(rhs)
    assert np.array_equal(np_a, np.array(num_a))


def test_bool_col_set_empty_selection_broadcast() -> None:
    # k > 0 empty-selection SET (a[:, empty_mask] = rhs): assignment
    # broadcasting is directional, so the RHS must broadcast *to* the indexed
    # shape (1, 0). rhs (2, 0) shares a symmetric broadcast result with (1, 0)
    # but cannot broadcast into it, so NumPy — and cuPyNumeric — must raise.
    arr = num.ones((1, 0))
    mask = num.zeros((0,), dtype=bool)  # empty mask over the size-0 axis

    # valid: scalar and (1, 0) rhs broadcast into (1, 0) -> no-op
    arr[:, mask] = 5
    arr[:, mask] = num.ones((1, 0))

    # invalid: (2, 0) cannot broadcast to (1, 0) — matches NumPy's ValueError.
    np_arr = np.ones((1, 0))
    np_mask = np.zeros((0,), dtype=bool)
    with pytest.raises(ValueError):
        np_arr[:, np_mask] = np.ones((2, 0))
    with pytest.raises(ValueError):
        arr[:, mask] = num.ones((2, 0))


def test_bool_whole_mask_set_requires_low_rank_rhs() -> None:
    # A whole-array boolean mask selects a 1-D run, so NumPy requires a 0- or
    # 1-D rhs and raises for a size-1 but multi-dim rhs like (1, 1) instead of
    # silently scattering.  Pins the rank guard before strategy selection.
    np_a = np.arange(6).reshape(2, 3)
    np_mask = np.array([[True, False, True], [False, True, False]])
    num_mask = num.array(np_mask)

    # valid: scalar and matching 1-D rhs broadcast/assign as NumPy does.
    np_a2, num_a2 = np_a.copy(), num.array(np_a.copy())
    np_a2[np_mask] = 7
    num_a2[num_mask] = 7
    np_a2[np_mask] = np.array([10, 20, 30])
    num_a2[num_mask] = num.array([10, 20, 30])
    assert np.array_equal(np_a2, np.array(num_a2))

    # invalid: size-1 but 2-D rhs -> raises like NumPy (a 0/1-D input is
    # required for a whole-mask assignment).
    with pytest.raises((TypeError, ValueError)):
        np_a.copy()[np_mask] = np.ones((1, 1))
    with pytest.raises((TypeError, ValueError)):
        num.array(np_a.copy())[num_mask] = num.ones((1, 1))


@pytest.mark.parametrize(
    "mask, rhs_shape",
    [
        ([True, True], (1, 2)),
        ([True, False], (1, 1)),
        ([False, False], (1, 0)),
    ],
    ids=["general", "putmask", "empty"],
)
def test_bool_whole_mask_set_checks_pre_squeeze_rank(
    mask: list[bool], rhs_shape: tuple[int, ...]
) -> None:
    np_a = np.zeros(2)
    num_a = num.zeros(2)
    np_mask = np.array(mask)
    num_mask = num.array(mask)

    with pytest.raises(TypeError):
        np_a[np_mask] = np.ones(rhs_shape)
    with pytest.raises(TypeError):
        num_a[num_mask] = num.ones(rhs_shape)


def test_int_array_set_keeps_leading_singleton_broadcast() -> None:
    np_a = np.zeros(2)
    num_a = num.zeros(2)
    np_idx = np.array([0, 1])
    num_idx = num.array(np_idx)
    rhs = np.ones((1, 2))

    np_a[np_idx] = rhs
    num_a[num_idx] = num.array(rhs)
    assert np.array_equal(np_a, np.array(num_a))


def test_bool_set_empty_selection_excess_leading_singleton() -> None:
    # A (2, 0) leading mask selects shape (0, 3) from a rank-3 target.  Keeping
    # target and RHS ranks equal prevents public preprocessing from squeezing
    # the RHS, so this reaches the excess-leading-singleton branch directly.
    np_a = np.ones((2, 0, 3))
    num_a = num.array(np_a)
    np_mask = np.zeros((2, 0), dtype=bool)
    num_mask = num.array(np_mask)

    # both the exact (0, 3) and the excess-leading-singleton (1, 0, 3) assign
    # (as empty no-ops) without raising, matching NumPy.
    np_a[np_mask] = np.ones((0, 3))
    num_a[num_mask] = num.ones((0, 3))
    np_a[np_mask] = np.ones((1, 0, 3))
    num_a[num_mask] = num.ones((1, 0, 3))
    assert np.array_equal(np_a, np.array(num_a))


def test_advanced_indexing_dimension_mismatch() -> None:
    arr = num.arange(6).reshape(2, 3)  # 2D array
    # Try to use too many index arrays
    idx1 = num.array([0])
    idx2 = num.array([1])
    idx3 = num.array([0])  # 3rd index for 2D array

    with pytest.raises((ValueError, IndexError)):
        # This should trigger dimension mismatch
        _ = arr[idx1, idx2, idx3]


def test_newaxis_in_boolean_indexing() -> None:
    np_arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    num_arr = num.arange(12, dtype=np.float32).reshape(3, 4)

    # Boolean mask + newaxis
    np_mask = np.array([True, False, True])
    num_mask = num.array([True, False, True])

    np_result = np_arr[np_mask, np.newaxis, :]
    num_result = num_arr[num_mask, np.newaxis, :]

    assert np_result.shape == num_result.shape
    assert np.allclose(np_result, np.array(num_result))


def test_newaxis_shifts_bool_to_mismatched_axis() -> None:
    np_arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    num_arr = num.array(np_arr)
    np_mask = np.array([True, False, True])
    num_mask = num.array(np_mask)
    np_result = np_arr[-1, np_mask]
    num_result = num_arr[-1, num_mask]
    assert np.array_equal(np_result, np.array(num_result))


def test_bool_index_with_newaxis_before_bool() -> None:
    arr = num.arange(6, dtype=np.float32).reshape(2, 3)
    bool_mask = num.array([True, False, True])
    with pytest.raises((ValueError, IndexError)):
        _ = arr[np.newaxis, bool_mask]


def test_bool_index_unsupported_type() -> None:
    arr = num.arange(6, dtype=np.float32).reshape(2, 3)
    bool_mask = num.array([True, False, True])
    with pytest.raises(TypeError, match="Unsupported entry type"):
        _ = arr[(0.5, bool_mask)]


def test_bool_shape_mismatch_in_mixed_indexing() -> None:
    arr = num.arange(12, dtype=np.float32).reshape(3, 4)
    int_idx = num.array([0, 1], dtype=np.int64)
    # shape (2,) doesn't match arr.shape[1] = 4
    wrong_bool = num.array([True, False], dtype=bool)
    with pytest.raises((ValueError, IndexError)):
        _ = arr[int_idx, wrong_bool]


def test_multiple_ellipses_raises() -> None:
    arr = num.arange(6).reshape(2, 3)
    with pytest.raises((ValueError, IndexError)):
        _ = arr[..., ...]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
