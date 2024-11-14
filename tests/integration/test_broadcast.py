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

import cupynumeric as num

DIM_CASES = [5, 40]


def _check_result(print_msg, err_arrs):
    if len(err_arrs) > 0:
        print_output = f"Failed, {print_msg}\n"
        for err_arr in err_arrs:
            print_output += (
                f"Attr, {err_arr[0]}\n"
                f"numpy result: {err_arr[1]}\n"
                f"cupynumeric_result: {err_arr[2]}\n"
            )
        assert False, (
            f"{print_output}"
            f"cupynumeric and numpy shows"
            f" different result\n"
        )
    else:
        print(f"Passed, {print_msg}")


def _broadcast_attrs(sizes):
    arr_np = list(np.arange(np.prod(size)).reshape(size) for size in sizes)
    arr_num = list(num.arange(np.prod(size)).reshape(size) for size in sizes)
    b = np.broadcast(*arr_np)
    c = num.broadcast(*arr_num)

    attrs = ["index", "nd", "ndim", "numiter", "shape", "size"]
    err_arrs = []
    # test attributes
    for attr in attrs:
        if getattr(b, attr) != getattr(c, attr):
            err_arrs.append([attr, getattr(b, attr), getattr(c, attr)])

    _check_result(f"np.broadcast({sizes})", err_arrs)


def _broadcast_value(sizes):
    arr_np = list(np.arange(np.prod(size)).reshape(size) for size in sizes)
    arr_num = list(num.arange(np.prod(size)).reshape(size) for size in sizes)
    b = np.broadcast(*arr_np)
    c = num.broadcast(*arr_num)

    err_arrs = []  # None

    # test elements in broadcasted array
    for each in zip(b, c):
        if each[0] != each[1]:
            err_arrs.append([("iters", b.index), each[0], each[1]])
    # test reset method
    b.reset()
    c.reset()
    if b.index != c.index:
        err_arrs.append([("reset", b.index), each[0], each[1]])

    _check_result(f"np.broadcast({sizes})", err_arrs)


def _broadcast_view(sizes):
    arr_num = list(num.arange(np.prod(size)).reshape(size) for size in sizes)
    c = num.broadcast(*arr_num)
    err_arrs = []  # None

    # test whether the broadcast provide views of the original array
    for i in range(len(arr_num)):
        arr_num[i][(0,) * arr_num[i].ndim] = 1
        if c.iters[i][0] != arr_num[i][(0,) * arr_num[i].ndim]:
            err_arrs.append(
                [
                    ("view", i),
                    c.iters[i][0],
                    arr_num[i][(0,) * arr_num[i].ndim],
                ]
            )

    _check_result(f"np.broadcast({sizes})", err_arrs)


def _broadcast_to_manipulation(arr, args):
    b = np.broadcast_to(*args).swapaxes(0, 1)
    c = num.broadcast_to(*args).swapaxes(0, 1)
    err_arrs = []  # [None, b, c]

    if not np.array_equal(b, c):
        err_arrs.append([None, b, c])

    _check_result(f"np.broadcast_to({args}).swapaxes(0,1)", err_arrs)


def _check(*args, params: list, routine: str):
    b = getattr(np, routine)(*args)
    c = getattr(num, routine)(*args)
    err_arrs = []  # None
    if isinstance(b, list):
        for each in zip(b, c):
            # Try to modify multiple elements in each broadcasted array
            if not np.array_equal(each[0], each[1]):
                err_arrs.append([("iters", b.index), each[0], each[1]])
    else:
        is_equal = True
        for each in zip(b, c):
            if isinstance(each[0], np.ndarray) and not np.array_equal(
                each[0], each[1]
            ):
                is_equal = False
            elif isinstance(each[0], tuple) and each[0] != each[1]:
                is_equal = False
            if not is_equal:
                err_arrs.append([("value", None), each[0], each[1]])

    _check_result(f"np.{routine}({params})", err_arrs)


def gen_shapes(dim):
    base = (dim,)
    result = [base]
    for i in range(1, LEGATE_MAX_DIM):
        base = base + (1,) if i % 2 == 0 else base + (dim,)
        result.append(base)
    return result


SHAPE_LISTS = {dim: gen_shapes(dim) for dim in DIM_CASES}


# test to run broadcast  w/ different size of arrays
@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_attrs(dim):
    _broadcast_attrs(SHAPE_LISTS[dim])


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_value(dim):
    _broadcast_value(SHAPE_LISTS[dim])


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_view(dim):
    _broadcast_view(SHAPE_LISTS[dim])


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_shapes(dim):
    shape_list = SHAPE_LISTS[dim]
    _check(*shape_list, params=shape_list, routine="broadcast_shapes")


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_to(dim):
    shape = SHAPE_LISTS[dim][-1]
    arr = np.arange(np.prod((dim,))).reshape((dim,))
    _check(arr, shape, params=(arr.shape, shape), routine="broadcast_to")


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_arrays(dim):
    shapes = SHAPE_LISTS[dim]
    arrays = list(np.arange(np.prod(shape)).reshape(shape) for shape in shapes)
    _check(*arrays, params=shapes, routine="broadcast_arrays")


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_broadcast_to_mainpulation(dim):
    shape = SHAPE_LISTS[dim][-1]
    arr = np.arange(np.prod((dim,))).reshape((dim,))
    _broadcast_to_manipulation(arr, (arr.shape, shape))


@pytest.mark.parametrize("dim", DIM_CASES, ids=str)
def test_iteration(dim):
    sizes = SHAPE_LISTS[dim]
    arr_np = list(np.arange(np.prod(size)).reshape(size) for size in sizes)
    arr_num = list(num.arange(np.prod(size)).reshape(size) for size in sizes)
    b = np.broadcast(*arr_np)
    c = num.broadcast(*arr_num)

    for item in b:
        assert item == next(c)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
