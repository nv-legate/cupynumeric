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
from utils.generators import mk_0to1_array

import cupynumeric as num


def _gen_array(n0, shape, dt, axis, outtype):
    range_lower = 0
    # range 1-3 for ints, avoid zeros for correct testing in prod case
    if np.issubdtype(dt, np.integer):
        range_lower = 1
    if dt == np.complex64:
        A = (
            (3 * np.random.random(shape) + range_lower)
            + (3 * np.random.random(shape) + range_lower) * 1j
        ).astype(np.complex64)
    elif dt == np.complex128:
        A = (
            (3 * np.random.random(shape) + range_lower)
            + (3 * np.random.random(shape) + range_lower) * 1j
        ).astype(np.complex128)
    else:
        A = (3 * np.random.random(shape) + range_lower).astype(dt)
    if n0 == "first_half":
        # second element along all axes is a NAN
        A[(1,) * len(shape)] = np.nan
    elif n0 == "second_half":
        # second from last element along all axes is a NAN
        A[tuple(map(lambda i, j: i - j, A.shape, (2,) * len(A.shape)))] = (
            np.nan
        )
    if outtype is None:
        B = None
        C = None
    else:
        if axis is None:
            B = np.zeros(shape=A.size, dtype=outtype)
            C = np.zeros(shape=A.size, dtype=outtype)
        else:
            B = np.zeros(shape=shape, dtype=outtype)
            C = np.zeros(shape=shape, dtype=outtype)
    return A, B, C


def _run_tests(op, n0, shape, dt, axis, out0, outtype):
    if (np.issubdtype(dt, np.integer) and n0 is not None) or (
        np.issubdtype(outtype, np.integer)
        and (op == "cumsum" or op == "cumprod")
        and n0 is not None
    ):
        return
    print(
        f"Running test: {op}, shape: {shape}, nan location: {n0}"
        f", axis: {axis}, in type: {dt}, out type: {outtype}"
        f", output array not provided: {out0}"
    )
    if out0 is True:
        A, B, C = _gen_array(n0, shape, dt, axis, None)
        B = getattr(num, op)(A, out=None, axis=axis, dtype=outtype)
        C = getattr(np, op)(A, out=None, axis=axis, dtype=outtype)
    else:
        A, B, C = _gen_array(n0, shape, dt, axis, outtype)
        getattr(num, op)(A, out=B, axis=axis, dtype=outtype)
        getattr(np, op)(A, out=C, axis=axis, dtype=outtype)

    print("Checking result...")
    if np.allclose(B, C, equal_nan=True):
        print("PASS!")
    else:
        print("FAIL!")
        print(f"INPUT    : {A}")
        print(f"CUPYNUMERIC: {B}")
        print(f"NUMPY    : {C}")
        assert False


ops = ["cumsum", "cumprod", "nancumsum", "nancumprod"]
ops_nan = ["nancumsum", "nancumprod"]
shapes = [[100], [4, 25], [4, 5, 6]]
axes = [None, 0, -1]
dtypes = [
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]
dtypes_simplified = [np.int32, np.float32, np.complex64]
n0s = [None, "first_half", "second_half"]


@pytest.mark.parametrize(
    "dtype, outtype",
    [
        pytest.param(np.int16, np.float64, marks=pytest.mark.xfail),
        # NumPy and cuPyNumeric produce different values
        # out_np: array([0., 0., 0., 0., 0., 0.])
        # out_num: array([0.16666667, 0.05555556, 0.02777778, 0.01851852,
        #                 0.0154321, 0.0154321 ]))
        (np.complex64, np.float64),
        (np.float32, np.int64),
    ],
    ids=str,
)
@pytest.mark.parametrize(
    "op",
    [
        pytest.param("cumsum", marks=pytest.mark.xfail),
        # cupynumeric.cumsum returns different value to numpy.cumsum
        # out_np: array([0., 0., 0., 0., 0., 0.])
        # out_num:
        # array([6.8983227e-310, 6.8983227e-310, 6.8983227e-310,
        #        6.8983227e-310, 6.8983227e-310, 6.8983227e-310])
        "cumprod",
        pytest.param("nancumsum", marks=pytest.mark.xfail),
        # dtype=np.float32, outtype=np.int64:
        # out_np: array([0, 0, 1, 1, 2, 3])
        # out_num: array([0, 0, 0, 0, 0, 1])
        "nancumprod",
    ],
    ids=str,
)
def test_scan_out_dtype_mismatch(dtype, outtype, op):
    shape = (1, 2, 3)
    out_shape = (6,)
    arr_np = mk_0to1_array(np, shape)
    arr_num = mk_0to1_array(num, shape)
    out_np = np.empty(out_shape, dtype=outtype)
    out_num = num.empty(out_shape, dtype=outtype)
    # NumPy ndarray doesn't have nancumprod, use module level API instead
    from cupynumeric.settings import settings

    if settings.force_thunk() != "eager":
        getattr(np, op)(arr_np, dtype=dtype, out=out_np)
        getattr(arr_num, op)(dtype=dtype, out=out_num)
        assert np.allclose(out_np, out_num)


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("axis", axes)
@pytest.mark.parametrize("outtype", dtypes_simplified)
@pytest.mark.parametrize("dt", dtypes_simplified)
def test_scan_out0_shape_ops(op, shape, axis, outtype, dt):
    out0 = True
    n0 = None
    _run_tests(op, n0, shape, dt, axis, out0, outtype)


@pytest.mark.parametrize("op", ops)
@pytest.mark.parametrize("outtype", dtypes_simplified)
@pytest.mark.parametrize("dt", dtypes)
@pytest.mark.parametrize("n0", n0s)
def test_scan_nan(op, outtype, dt, n0):
    shape = [100]
    axis = None
    out0 = False
    _run_tests(op, n0, shape, dt, axis, out0, outtype)


@pytest.mark.parametrize("op", ops)
def test_empty_inputs(op):
    in_np = np.ones(10)
    in_np[5:] = 0
    in_num = num.array(in_np)
    out_np = getattr(np, op)(in_np)
    out_num = getattr(num, op)(in_num)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("op", ops)
def test_empty_array(op):
    A = []
    out_np = getattr(np, op)(A)
    out_num = getattr(num, op)(A)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("op", ops)
def test_scalar(op):
    A = 1
    out_np = getattr(np, op)(A)
    out_num = getattr(num, op)(A)
    assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("op", ("nancumsum", "nancumprod"))
def test_nan_scalar_convert(op: str) -> None:
    arr_np = np.array([np.nan], dtype=np.float32)
    arr_num = num.array(arr_np)
    out_np = getattr(np, op)(arr_np, dtype=np.float64)
    out_num = getattr(num, op)(arr_num, dtype=np.float64)
    assert np.array_equal(out_np, out_num, equal_nan=True)


class TestScanErrors:
    @pytest.mark.parametrize("op", ("cumsum", "cumprod"))
    def test_array_with_nan(self, op):
        expected_exc = TypeError
        A = [1, 2, None, 4]
        with pytest.raises(expected_exc):
            getattr(np, op)(A)
        with pytest.raises(expected_exc):
            getattr(num, op)(A)

    @pytest.mark.parametrize(
        "axis", (-2, 1), ids=lambda axis: f"(axis={axis})"
    )
    @pytest.mark.parametrize("op", ops)
    def test_axis_out_of_bound(self, op, axis):
        expected_exc = ValueError
        A = [1, 2, 3, 4]
        with pytest.raises(expected_exc):
            getattr(np, op)(A, axis=axis)
        with pytest.raises(expected_exc):
            getattr(num, op)(A, axis=axis)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "out_shape",
        ((1,), (2, 3)),
        ids=lambda out_shape: f"(out_shape={out_shape})",
    )
    @pytest.mark.parametrize("op", ops)
    def test_out_invalid_shape(self, op, out_shape):
        # for all ops and all out_shape,
        # in Numpy, it raises ValueError
        # in cuPyNumeric, it raises NotImplementedError
        expected_exc = ValueError
        A = [1, 2, 3, 4]
        out_np = np.zeros(out_shape)
        out_num = num.zeros(out_shape)
        with pytest.raises(expected_exc):
            getattr(np, op)(A, out=out_np)
        with pytest.raises(expected_exc):
            getattr(num, op)(A, out=out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
