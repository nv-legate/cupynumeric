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

import itertools

import numpy as np
import pytest

import cupynumeric as num


def run_test(arr, routine, input_size):
    input_arr = [[arr]]
    if routine == "concatenate" or routine == "stack":
        # 'axis' options
        input_arr.append([axis for axis in range(arr[0].ndim)])
        # test axis == 'None' for concatenate
        if routine == "concatenate":
            # test axis == -1 if ndim > 0
            if arr[0].ndim > 0:
                input_arr[-1].append(-1)
            input_arr[-1].append(None)
        # 'out' argument
        input_arr.append([None])
    test_args = itertools.product(*input_arr)

    for args in test_args:
        b = getattr(np, routine)(*args)
        c = getattr(num, routine)(*args)
        is_equal = True
        err_arr = [b, c]

        if len(b) != len(c):
            is_equal = False
            err_arr = [b, c]
        else:
            for each in zip(b, c):
                if not np.array_equal(*each):
                    err_arr = each
                    is_equal = False
                    break
        shape_list = list(inp.shape for inp in arr)
        print_msg = f"np.{routine}(array({shape_list}), {args[1:]})"
        assert is_equal, (
            f"Failed, {print_msg}\n"
            f"numpy result: {err_arr[0]}, {b.shape}\n"
            f"cupynumeric_result: {err_arr[1]}, {c.shape}\n"
            f"cupynumeric and numpy shows"
            f" different result\n"
            f"array({arr}),"
            f"routine: {routine},"
            f"args: {args[1:]}"
        )
        print(
            f"Passed, {print_msg}, np: ({b.shape}, {b.dtype})"
            f", cupynumeric: ({c.shape}, {c.dtype}"
        )


DIM = 10

NUM_ARR = [1, 3]

SIZES = [
    (),
    (0,),
    (0, 10),
    (1,),
    (1, 1),
    (1, 1, 1),
    (1, DIM),
    (DIM, DIM),
    (DIM, DIM, DIM),
]

SCALARS = ((10,), (10, 20, 30))


@pytest.fixture(autouse=False)
def a(size, num):
    return [np.random.randint(low=0, high=100, size=size) for _ in range(num)]


@pytest.mark.parametrize("num", NUM_ARR, ids=str)
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_concatenate(size, num, a):
    run_test(tuple(a), "concatenate", size)


@pytest.mark.parametrize("arrays", SCALARS, ids=str)
def test_concatenate_scalar(arrays):
    res_np = np.concatenate(arrays, axis=None)
    res_num = num.concatenate(arrays, axis=None)
    assert np.array_equal(res_np, res_num)


def test_concatenate_with_out():
    a = [[1, 2], [3, 4]]
    b = [[5, 6]]
    axis = 0
    out_np = np.zeros((3, 2))
    out_num = num.array(out_np)

    np.concatenate((np.array(a), np.array(b)), axis=axis, out=out_np)
    result = num.concatenate(
        (num.array(a), num.array(b)), axis=axis, out=out_num
    )
    assert np.array_equal(out_np, out_num)
    assert np.array_equal(out_num, result)
    assert isinstance(result, num.ndarray)


@pytest.mark.parametrize(
    "dtype", (np.float32, np.int32), ids=lambda dtype: f"(dtype={dtype})"
)
def test_concatenate_dtype(dtype):
    a = [[1, 2], [3, 4]]
    b = [[5, 6]]
    axis = 0

    res_np = np.concatenate((np.array(a), np.array(b)), axis=axis, dtype=dtype)
    res_num = num.concatenate(
        (num.array(a), num.array(b)), axis=axis, dtype=dtype
    )
    assert np.array_equal(res_np, res_num)


def test_concatenate_return():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])

    res_num = num.concatenate((a, b))

    assert isinstance(res_num, num.ndarray)


@pytest.mark.parametrize(
    "casting",
    ("no", "equiv", "safe", "same_kind", "unsafe"),
    ids=lambda casting: f"(casting={casting})",
)
def test_concatenate_casting(casting):
    a = [[1, 2], [3, 4]]
    b = [[5, 6]]
    axis = 0

    res_np = np.concatenate(
        (np.array(a), np.array(b)), axis=axis, casting=casting
    )
    res_num = num.concatenate(
        (num.array(a), num.array(b)), axis=axis, casting=casting
    )
    assert np.array_equal(res_np, res_num)


class TestConcatenateErrors:
    def test_zero_arrays(self):
        expected_exc = ValueError
        arrays = ()
        axis = None
        with pytest.raises(expected_exc):
            np.concatenate(arrays, axis=axis)
        with pytest.raises(expected_exc):
            num.concatenate(arrays, axis=axis)

    @pytest.mark.parametrize(
        "arrays",
        ((1,), (1, 2), (1, [3, 4])),
        ids=lambda arrays: f"(arrays={arrays})",
    )
    def test_scalar_axis_is_not_none(self, arrays):
        expected_exc = ValueError
        axis = 0
        with pytest.raises(expected_exc):
            np.concatenate(arrays, axis=axis)
        with pytest.raises(expected_exc):
            num.concatenate(arrays, axis=axis)

    @pytest.mark.parametrize(
        "arrays",
        (
            ([[1, 2], [3, 4]], [5, 6]),
            ([[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]),
            ([[1, 2], [3, 4]], [[5, 6]]),
        ),
        ids=lambda arrays: f"(arrays={arrays})",
    )
    def test_arrays_mismatched_shape(self, arrays):
        expected_exc = ValueError
        axis = 1
        with pytest.raises(expected_exc):
            np.concatenate(arrays, axis=axis)
        with pytest.raises(expected_exc):
            num.concatenate(arrays, axis=axis)

    @pytest.mark.parametrize(
        "axis", (1, -2), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_out_of_bound(self, axis):
        expected_exc = ValueError
        a = [1, 2]
        b = [5, 6]
        with pytest.raises(expected_exc):
            np.concatenate((np.array(a), np.array(b)), axis=axis)
        with pytest.raises(expected_exc):
            num.concatenate((num.array(a), num.array(b)), axis=axis)

    def test_both_out_dtype_are_provided(self):
        expected_exc = TypeError
        a = [[1, 2], [3, 4]]
        b = [[5, 6]]
        axis = 0
        out_np = np.zeros((3, 2))
        out_num = num.array(out_np)
        dtype = np.float32

        with pytest.raises(expected_exc):
            np.concatenate(
                (np.array(a), np.array(b)), axis=axis, out=out_np, dtype=dtype
            )
        with pytest.raises(expected_exc):
            num.concatenate(
                (num.array(a), num.array(b)),
                axis=axis,
                out=out_num,
                dtype=dtype,
            )

    def test_invalid_casting(self):
        expected_exc = ValueError
        a = [[1, 2], [3, 4]]
        b = [[5, 6]]
        axis = 0
        casting = "unknown"
        with pytest.raises(expected_exc):
            np.concatenate(
                (np.array(a), np.array(b)), axis=axis, casting=casting
            )
        with pytest.raises(expected_exc):
            num.concatenate(
                (num.array(a), num.array(b)), axis=axis, casting=casting
            )


@pytest.mark.parametrize("num", NUM_ARR, ids=str)
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_stack(size, num, a):
    run_test(tuple(a), "stack", size)


@pytest.mark.parametrize("arrays", SCALARS, ids=str)
def test_stack_scalar(arrays):
    res_np = np.stack(arrays)
    res_num = num.stack(arrays)
    assert np.array_equal(res_np, res_num)


def test_stack_with_out():
    a = [1, 2]
    b = [3, 4]
    axis = 0
    out_np = np.zeros((2, 2))
    out_num = num.array(out_np)

    np.stack((np.array(a), np.array(b)), axis=axis, out=out_np)
    num.stack((num.array(a), num.array(b)), axis=axis, out=out_num)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("axis", (-3, -1), ids=lambda axis: f"(axis={axis})")
def test_stack_axis_is_negative(axis):
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    res_np = np.stack((np.array(a), np.array(b)), axis=axis)
    res_num = num.stack((num.array(a), num.array(b)), axis=axis)
    assert np.array_equal(res_np, res_num)


class TestStackErrors:
    def test_zero_arrays(self):
        expected_exc = ValueError
        arrays = ()
        with pytest.raises(expected_exc):
            np.stack(arrays)
        with pytest.raises(expected_exc):
            num.stack(arrays)

    @pytest.mark.parametrize(
        "arrays",
        (
            (1, []),
            ([1, 2], [3]),
            ([[1, 2], [3, 4]], [[5, 6, 7], [8, 9, 10]]),
            ([[1, 2], [3, 4]], [5, 6]),
        ),
        ids=lambda arrays: f"(arrays={arrays})",
    )
    def test_arrays_mismatched_shape(self, arrays):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.stack(arrays)
        with pytest.raises(expected_exc):
            num.stack(arrays)

    def test_axis_is_none(self):
        expected_exc = TypeError
        a = [1, 2]
        b = [5, 6]
        axis = None
        with pytest.raises(expected_exc):
            np.stack((np.array(a), np.array(b)), axis=axis)
        with pytest.raises(expected_exc):
            num.stack((num.array(a), num.array(b)), axis=axis)

    @pytest.mark.parametrize(
        "axis", (2, -3), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_out_of_bound(self, axis):
        expected_exc = ValueError
        a = [1, 2]
        b = [5, 6]
        with pytest.raises(expected_exc):
            np.stack((np.array(a), np.array(b)), axis=axis)
        with pytest.raises(expected_exc):
            num.stack((num.array(a), num.array(b)), axis=axis)

    @pytest.mark.parametrize(
        "out_shape",
        ((2,), (1, 2), (1, 2, 2)),
        ids=lambda out_shape: f"(out_shape={out_shape})",
    )
    def test_out_invalid_shape(self, out_shape):
        expected_exc = ValueError
        a = [1, 2]
        b = [3, 4]
        axis = 0
        out_np = np.zeros(out_shape)
        out_num = num.array(out_np)

        with pytest.raises(expected_exc):
            np.stack((np.array(a), np.array(b)), axis=axis, out=out_np)
        with pytest.raises(expected_exc):
            num.stack((num.array(a), num.array(b)), axis=axis, out=out_num)


@pytest.mark.parametrize("num", NUM_ARR, ids=str)
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_hstack(size, num, a):
    run_test(tuple(a), "hstack", size)


@pytest.mark.parametrize("arrays", SCALARS, ids=str)
def test_hstack_scalar(arrays):
    res_np = np.hstack(arrays)
    res_num = num.hstack(arrays)
    assert np.array_equal(res_np, res_num)


class TestHStackErrors:
    def test_zero_arrays(self):
        expected_exc = ValueError
        arrays = ()
        with pytest.raises(expected_exc):
            np.hstack(arrays)
        with pytest.raises(expected_exc):
            num.hstack(arrays)

    @pytest.mark.parametrize(
        "arrays",
        (([[1, 2], [3, 4]], [5, 6]), ([[1, 2], [3, 4]], [[5, 6]])),
        ids=lambda arrays: f"(arrays={arrays})",
    )
    def test_arrays_mismatched_shape(self, arrays):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.hstack(arrays)
        with pytest.raises(expected_exc):
            num.hstack(arrays)


@pytest.mark.parametrize("num", NUM_ARR, ids=str)
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_column_stack(size, num, a):
    run_test(tuple(a), "column_stack", size)


@pytest.mark.parametrize("arrays", SCALARS, ids=str)
def test_column_stack_scalar(arrays):
    res_np = np.column_stack(arrays)
    res_num = num.column_stack(arrays)
    assert np.array_equal(res_np, res_num)


class TestColumnStackErrors:
    def test_zero_arrays(self):
        expected_exc = ValueError
        arrays = ()
        with pytest.raises(expected_exc):
            np.column_stack(arrays)
        with pytest.raises(expected_exc):
            num.column_stack(arrays)

    @pytest.mark.parametrize(
        "arrays",
        ((1, []), ([1, 2], [3]), ([[1, 2]], [3, 4]), ([[1, 2]], [[3], [4]])),
        ids=lambda arrays: f"(arrays={arrays})",
    )
    def test_arrays_mismatched_shape(self, arrays):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.column_stack(arrays)
        with pytest.raises(expected_exc):
            num.column_stack(arrays)


@pytest.mark.parametrize("num", NUM_ARR, ids=str)
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_vstack(size, num, a):
    # exception for 1d array on vstack
    if len(size) == 2 and size == (1, DIM):
        a.append(np.random.randint(low=0, high=100, size=(DIM,)))
    run_test(tuple(a), "vstack", size)


@pytest.mark.parametrize("arrays", SCALARS, ids=str)
def test_vstack_scalar(arrays):
    res_np = np.vstack(arrays)
    res_num = num.vstack(arrays)
    assert np.array_equal(res_np, res_num)


class TestVStackErrors:
    def test_zero_arrays(self):
        expected_exc = ValueError
        arrays = ()
        with pytest.raises(expected_exc):
            np.vstack(arrays)
        with pytest.raises(expected_exc):
            num.vstack(arrays)

    @pytest.mark.parametrize(
        "arrays",
        (
            (1, []),
            ([1, 2], [3]),
            ([[1, 2], [3, 4]], [5]),
            ([[[1, 2], [3, 4]]], [5, 6]),
        ),
        ids=lambda arrays: f"(arrays={arrays})",
    )
    def test_arrays_mismatched_shape(self, arrays):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.vstack(arrays)
        with pytest.raises(expected_exc):
            num.vstack(arrays)


@pytest.mark.parametrize("num", NUM_ARR, ids=str)
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_rowstack(size, num, a):
    # exception for 1d array on rowstack
    if len(size) == 2 and size == (1, DIM):
        a.append(np.random.randint(low=0, high=100, size=(DIM,)))
    run_test(tuple(a), "row_stack", size)


@pytest.mark.parametrize("arrays", SCALARS, ids=str)
def test_row_stack_scalar(arrays):
    res_np = np.row_stack(arrays)
    res_num = num.row_stack(arrays)
    assert np.array_equal(res_np, res_num)


class TestRowStackErrors:
    def test_zero_arrays(self):
        expected_exc = ValueError
        arrays = ()
        with pytest.raises(expected_exc):
            np.row_stack(arrays)
        with pytest.raises(expected_exc):
            num.row_stack(arrays)

    @pytest.mark.parametrize(
        "arrays",
        (
            (1, []),
            ([1, 2], [3]),
            ([[1, 2], [3, 4]], [5]),
            ([[[1, 2], [3, 4]]], [5, 6]),
        ),
        ids=lambda arrays: f"(arrays={arrays})",
    )
    def test_arrays_mismatched_shape(self, arrays):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.row_stack(arrays)
        with pytest.raises(expected_exc):
            num.row_stack(arrays)


@pytest.mark.parametrize("num", NUM_ARR, ids=str)
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_dstack(size, num, a):
    # exception for 1d array on dstack
    if len(size) == 2 and size == (1, DIM):
        a.append(np.random.randint(low=0, high=100, size=(DIM,)))
    run_test(tuple(a), "dstack", size)


@pytest.mark.parametrize("arrays", SCALARS, ids=str)
def test_dstack_scalar(arrays):
    res_np = np.dstack(arrays)
    res_num = num.dstack(arrays)
    assert np.array_equal(res_np, res_num)


class TestDStackErrors:
    def test_zero_arrays(self):
        expected_exc = ValueError
        arrays = ()
        with pytest.raises(expected_exc):
            np.dstack(arrays)
        with pytest.raises(expected_exc):
            num.dstack(arrays)

    @pytest.mark.parametrize(
        "arrays",
        (
            (1, []),
            ([1, 2], [5]),
            ([[1, 2], [3, 4]], [5, 6]),
            ([[1, 2], [3, 4]], [[5], [6]]),
        ),
        ids=lambda arrays: f"(arrays={arrays})",
    )
    def test_arrays_mismatched_shape(self, arrays):
        expected_exc = ValueError
        with pytest.raises(expected_exc):
            np.dstack(arrays)
        with pytest.raises(expected_exc):
            num.dstack(arrays)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
