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

import argparse
import operator

import numpy as np
import pytest
from utils.comparisons import allclose

import cupynumeric as num


def check_result(op, in_np, out_np, out_num):
    # Allow all scalars (just for parametrizing over Python scalars as well)
    rtol = any(
        1e-02 if getattr(a, "dtype", None) == np.float16 else 1e-05
        for a in in_np
    )
    result = allclose(out_np, out_num, rtol=rtol) and (
        getattr(out_np, "dtype", None) == getattr(out_num, "dtype", None)
    )
    if not result:
        print(f"cupynumeric.{op} failed the test")
        print("Inputs:")
        for arr in in_np:
            print(arr)
            if hasattr(arr, "dtype"):
                print(f"dtype: {arr.dtype}")
        print("NumPy output:")
        print(out_np)
        print(f"dtype: {out_np.dtype}")
        print("cuPyNumeric output:")
        print(out_num)
        print(f"dtype: {out_num.dtype}")
        assert False


def check_op(op, in_np, out_dtype="D"):
    in_num = tuple(
        num.array(op) if not isinstance(op, int) else op for op in in_np
    )

    if op.isidentifier():
        op_np = getattr(np, op)
        op_num = getattr(num, op)
        assert op_np.nout == 1

        out_np = op_np(*in_np)
        out_num = op_num(*in_num)

        check_result(op, in_np, out_np, out_num)

        out_np = np.empty(out_np.shape, dtype=out_dtype)
        out_num = num.empty(out_num.shape, dtype=out_dtype)
        op_np(*in_np, out=out_np)
        op_num(*in_num, out=out_num)

        check_result(op, in_np, out_np, out_num)

        # Ask cuPyNumeric to produce outputs to NumPy ndarrays
        out_num = np.empty(out_np.shape, dtype=out_dtype)
        op_num(*in_num, out=out_num)

        check_result(op, in_np, out_np, out_num)

    else:
        # Doing it this way instead of invoking the dunders directly, to
        # avoid having to select the right version, __add__ vs __radd__,
        # when one isn't supported, e.g. for scalar.__add__(array)

        out_np = eval(f"in_np[0] {op} in_np[1]")
        out_num = eval(f"in_num[0] {op} in_num[1]")

        check_result(op, in_np, out_np, out_num)

        # Call asarray here to ensure out_np/out_num is not a python scalar
        out_np = np.ones_like(np.asarray(out_np))
        out_num = num.ones_like(num.asarray(out_num))
        exec(f"out_np {op}= in_np[0]")
        exec(f"out_num {op}= in_num[0]")

        check_result(op, in_np, out_np, out_num)

        out_num = np.ones_like(out_np)
        exec(f"out_num {op}= in_num[0]")

        check_result(op, in_np, out_np, out_num)


# TODO: right now we will simply check if the operations work
# for some boring inputs. For some of these, we will want to
# test corner cases in the future.

# TODO: matmul, @

# Math operations
math_ops = [
    "*",
    "+",
    "-",
    "/",
    "add",
    # "divmod",
    "equal",
    "fmax",
    "fmin",
    "greater",
    "greater_equal",
    # "heaviside",
    # "ldexp",
    "less",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "not_equal",
    "subtract",
    "true_divide",
]

# We want to test array-array, array-scalar, and scalar-array cases
arrs = (
    np.random.randint(3, 10, size=(4, 5)).astype("I"),
    np.random.uniform(size=(4, 5)).astype("e"),
    np.random.uniform(size=(4, 5)).astype("f"),
    np.random.uniform(size=(4, 5)).astype("d"),
    np.random.uniform(size=(4, 5)).astype("F"),
)

scalars = (
    2,
    np.uint64(2),
    np.int64(-3),
    np.random.randn(1)[0],
    np.complex64(1 + 1j),
)


@pytest.mark.parametrize("op", math_ops)
@pytest.mark.parametrize("arr1", arrs)
@pytest.mark.parametrize("arr2", arrs)
def test_math_ops_arr_arr(op, arr1, arr2) -> None:
    check_op(op, (arr1, arr2))


@pytest.mark.parametrize("op", math_ops)
@pytest.mark.parametrize("arr", arrs)
@pytest.mark.parametrize("scalar", scalars)
def test_math_ops_arr_scalar(op, arr, scalar) -> None:
    check_op(op, (arr, scalar))
    check_op(op, (scalar, arr))


@pytest.mark.parametrize("op", math_ops)
@pytest.mark.parametrize("scalar1", scalars)
@pytest.mark.parametrize("scalar2", scalars)
def test_math_ops_scalar_scalar(op, scalar1, scalar2) -> None:
    check_op(op, (scalar1, scalar2))


trig_ops = [
    "//",
    "arctan2",
    "copysign",
    "floor_divide",
    "mod",
    "fmod",
    "hypot",
    "logaddexp",
    "logaddexp2",
    "nextafter",
]


@pytest.mark.parametrize("op", trig_ops)
@pytest.mark.parametrize("arr1", arrs[:-1])
@pytest.mark.parametrize("arr2", arrs[:-1])
def test_trig_ops_arr_arr(op, arr1, arr2) -> None:
    check_op(op, (arr1, arr2))


@pytest.mark.parametrize("op", trig_ops)
@pytest.mark.parametrize("arr", arrs[:-1])
@pytest.mark.parametrize("scalar", scalars[:-1])
def test_trig_ops_arr_scalar(op, arr, scalar) -> None:
    check_op(op, (arr, scalar))
    check_op(op, (scalar, arr))


@pytest.mark.parametrize("op", trig_ops)
@pytest.mark.parametrize("scalar1", scalars[:-1])
@pytest.mark.parametrize("scalar2", scalars[:-1])
def test_trig_ops_scalar_scalar(op, scalar1, scalar2) -> None:
    check_op(op, (scalar1, scalar2))


power_ops = [
    "**",
    "power",
    "float_power",
]


@pytest.mark.parametrize("op", power_ops)
@pytest.mark.parametrize("arr1", arrs[:-1])
@pytest.mark.parametrize("arr2", arrs[:-1])
def test_power_ops_arr_arr(op, arr1, arr2) -> None:
    check_op(op, (arr1, arr2))


@pytest.mark.parametrize("op", power_ops)
@pytest.mark.parametrize("arr", arrs[:-1])
def test_power_ops_arr_scalar(op, arr) -> None:
    # NOTE: As of NumPy 2.0.1, NumPy has special paths for the operators
    # which means that **uint64(2) returns a wrong dtype (optimizing it to
    # no modify the input).
    # See https://github.com/numpy/numpy/issues/27082.
    check_op(op, (arr, scalars[0]))
    check_op(op, (scalars[0], arr))
    check_op(op, (arr, scalars[4]))
    check_op(op, (scalars[4], scalars[4]))


@pytest.mark.parametrize("op", power_ops)
def test_power_ops_scalar_scalar(op) -> None:
    check_op(op, (scalars[0], scalars[4]))
    check_op(op, (scalars[4], scalars[0]))


div_ops = [
    "%",
    "remainder",
]


@pytest.mark.parametrize("op", div_ops)
@pytest.mark.parametrize("arr1", arrs[:-1])
@pytest.mark.parametrize("arr2", arrs[:-1])
def test_div_ops_arr_arr(op, arr1, arr2) -> None:
    check_op(op, (arr1, arr2))


@pytest.mark.parametrize("op", div_ops)
@pytest.mark.parametrize("arr", arrs[:-1])
@pytest.mark.parametrize("scalar", scalars[:-2])
def test_div_ops_arr_scalar(op, arr, scalar) -> None:
    check_op(op, (arr, scalar))
    check_op(op, (scalar, arr))


@pytest.mark.parametrize("op", div_ops)
@pytest.mark.parametrize("scalar1", scalars[:-2])
@pytest.mark.parametrize("scalar2", scalars[:-2])
def test_div_ops_scalar_scalar(op, scalar1, scalar2) -> None:
    check_op(op, (scalar1, scalar2))


bit_ops = [
    "&",
    "<<",
    ">>",
    "^",
    "|",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "gcd",
    "lcm",
    "left_shift",
    "right_shift",
]


@pytest.mark.parametrize("op", bit_ops)
def test_bit_ops_arr_arr(op) -> None:
    check_op(op, (arrs[0], arrs[0]))


@pytest.mark.parametrize("op", bit_ops)
def test_bit_ops_arr_scalar(op) -> None:
    check_op(op, (arrs[0], scalars[0]))
    check_op(op, (arrs[0], scalars[1]))
    check_op(op, (arrs[0], scalars[2]))
    # cuPyNumeric << and >> have problems with python integers:
    # check_op(op, (scalars[0], arrs[0]))
    check_op(op, (scalars[1], arrs[0]))
    check_op(op, (scalars[2], arrs[0]))


@pytest.mark.parametrize("op", bit_ops)
def test_bit_ops_scalar_scalar(op) -> None:
    check_op(op, (scalars[1], scalars[1]))


@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "arr,val",
    [
        (num.asarray([0, 127, -128, 1], dtype="int8"), -1000),
        (num.asarray([0, 127, -128, 1], dtype="int8"), 1000),
        (num.asarray([0, 127, 255, 1], dtype="uint8"), -1),
        (num.asarray([0, -(2**62), 2**62 - 1, 1], dtype="int64"), 2**63),
    ],
)
def test_pyint_comparison(op, reverse, arr, val):
    if reverse:
        _op = op

        def op(x, y):
            return _op(y, x)

    # The expected result is the same as any number within range (i.e. 0)
    expected = op(0, val)
    res = op(arr, val)

    assert res.dtype == bool
    assert res.shape == arr.shape
    if expected:
        assert res.all()
    else:
        assert not res.any()

    # Check comparison to 0 as well
    assert op(arr, 0).tolist() == [op(v, 0) for v in arr.tolist()]


def parse_inputs(in_str, dtype_str):
    dtypes = tuple(np.dtype(dtype) for dtype in dtype_str.split(":"))
    tokens = in_str.split(":")
    inputs = []
    for token, dtype in zip(tokens, dtypes):
        split = token.split(",")
        if len(split) == 1:
            inputs.append(dtype.type(split[0]))
        else:
            inputs.append(np.array(split, dtype=dtype))
    return inputs


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opname",
        default=None,
        dest="op",
        help="the name of operation to test",
    )
    parser.add_argument(
        "--inputs",
        dest="inputs",
        default="1:1",
        help="input data",
    )
    parser.add_argument(
        "--dtypes",
        dest="dtypes",
        default="l:l",
        help="input data",
    )
    args, extra = parser.parse_known_args()

    sys.argv = sys.argv[:1] + extra

    if args.op is not None:
        in_np = parse_inputs(args.inputs, args.dtypes)
        check_op(args.op, in_np)
    else:
        sys.exit(pytest.main(sys.argv))
