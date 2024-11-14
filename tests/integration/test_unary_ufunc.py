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

import numpy as np
import pytest
from utils.comparisons import allclose

import cupynumeric as num


def deterministic_op_test(func):
    # Uses the op name to create a deterministic seed.
    # This enforces that inputs are always the same whether
    # running all tests or a single test with -k.
    def wrapper_set_seed(op, *args, **kwargs):
        func(op, *args, **kwargs)
        func(op, *args, **kwargs)

    return wrapper_set_seed


def check_result(op, in_np, out_np, out_num, **isclose_kwargs):
    if in_np.dtype == "e" or out_np.dtype == "e":
        # The mantissa is only 10 bits, 2**-10 ~= 10^(-4)
        # Gives 1e-3 as rtol to provide extra rounding error.
        f16_rtol = 1e-3
        rtol = isclose_kwargs.setdefault("rtol", f16_rtol)
        # make sure we aren't trying to fp16 compare with less precision
        assert rtol >= f16_rtol

    result = (
        allclose(out_np, out_num, **isclose_kwargs)
        and out_np.dtype == out_num.dtype
    )
    if not result:
        print(f"cupynumeric.{op} failed the test")
        print("Input:")
        print(in_np)
        print(f"dtype: {in_np.dtype}")
        print("NumPy output:")
        print(out_np)
        print(f"dtype: {out_np.dtype}")
        print("cuPyNumeric output:")
        print(out_num)
        print(f"dtype: {out_num.dtype}")
    return result


def check_op(op, in_np, out_dtype="d", **check_kwargs):
    in_num = num.array(in_np)

    if op.isidentifier():
        op_np = getattr(np, op)
        op_num = getattr(num, op)

        assert op_np.nout == 1

        out_np = op_np(in_np)
        out_num = op_num(in_num)

        assert check_result(op, in_np, out_np, out_num, **check_kwargs)

        out_np = np.empty(out_np.shape, dtype=out_dtype)
        out_num = num.empty(out_num.shape, dtype=out_dtype)

        op_np(in_np, out=out_np)
        op_num(in_num, out=out_num)

        assert check_result(op, in_np, out_np, out_num, **check_kwargs)

        out_np = np.empty(out_np.shape, dtype=out_dtype)
        out_num = num.empty(out_num.shape, dtype=out_dtype)

        op_np(in_np, out_np)
        op_num(in_num, out_num)

        assert check_result(op, in_np, out_np, out_num, **check_kwargs)

        # Ask cuPyNumeric to produce outputs to NumPy ndarrays
        out_num = np.ones(out_np.shape, dtype=out_dtype)
        op_num(in_num, out_num)

        assert check_result(op, in_np, out_np, out_num, **check_kwargs)

    else:
        out_np = eval(f"{op} in_np")
        out_num = eval(f"{op} in_num")

        assert check_result(op, in_np, out_np, out_num, **check_kwargs)


def check_ops(ops, in_np, out_dtype="d"):
    for op in ops:
        check_op(op, in_np, out_dtype)


def check_op_input(
    op,
    shape=(4, 5),
    a_min=None,
    a_max=None,
    randint=False,
    offset=None,
    astype=None,
    out_dtype="d",
    replace_zero=None,
    **check_kwargs,
):
    if randint:
        assert a_min is not None
        assert a_max is not None
        in_np = np.random.randint(a_min, a_max, size=shape)
    else:
        in_np = np.random.randn(*shape)
        if offset is not None:
            in_np = in_np + offset
        if a_min is not None:
            in_np = np.maximum(a_min, in_np)
        if a_max is not None:
            in_np = np.minimum(a_max, in_np)
        if astype is not None:
            in_np = in_np.astype(astype)

    if replace_zero is not None:
        in_np[in_np == 0] = replace_zero

    # converts to a scalar if shape is (1,)
    if in_np.ndim == 1 and in_np.shape[0] == 1:
        in_np = in_np[0]
    check_op(op, in_np, out_dtype=out_dtype, **check_kwargs)


# TODO: right now we will simply check if the operations work
# for some boring inputs. For some of these, we will want to
# test corner cases in the future.


@deterministic_op_test
def check_math_ops(op, **kwargs):
    check_op_input(op, **kwargs)
    check_op_input(op, astype="e", **kwargs)
    check_op_input(op, astype="f", **kwargs)
    check_op_input(op, astype="b", **kwargs)
    check_op_input(op, astype="B", **kwargs)
    check_op_input(op, randint=True, a_min=1, a_max=10, **kwargs)
    check_op_input(op, shape=(1,), **kwargs)


# Math operations
math_ops = (
    "+",
    "-",
    "absolute",
    "conjugate",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "logical_not",
    "negative",
    "positive",
    "rint",
    "sign",
    "square",
)


@pytest.mark.parametrize("op", math_ops)
def test_default_math_ops(op):
    check_math_ops(op)


special_math_ops = (
    # reciprocal is undefined on zero, replaces with 1
    ("reciprocal", dict(replace_zero=1)),
)


@pytest.mark.parametrize("op,kwargs", special_math_ops)
def test_special_math_ops(op, kwargs):
    check_math_ops(op, **kwargs)


log_ops = (
    "log",
    "log10",
    "log1p",
    "log2",
)


@pytest.mark.parametrize("op", log_ops)
@deterministic_op_test
def test_log_ops(op):
    # for real-valued log functions, requires inputs to be positive
    # since numpy does log(real) -> real and not log(real)->complex
    # for negative inputs
    check_op_input(op, offset=3, a_min=0.1)
    check_op_input(op, astype="e", offset=3, a_min=0.1)
    check_op_input(op, astype="f", offset=3, a_min=0.1)

    # for real-valued log functions, allows negative values and checks
    # that nans are returned appropriately for bad cases
    check_op_input(op, equal_nan=True)

    # for the complex case, this allows negative input values
    # in order to produce complex output values
    check_op_input(op, astype="F", out_dtype="D")

    check_op_input(op, randint=True, a_min=3, a_max=10)
    check_op_input(op, shape=(1,), a_min=0.1, offset=3)


even_root_ops = ("sqrt",)


@pytest.mark.parametrize("op", even_root_ops)
@deterministic_op_test
def test_even_root_ops(op):
    # Need to guarantee positive inputs with a_min # for float roots
    check_op_input(op, offset=3, a_min=0)
    check_op_input(op, astype="e", offset=3, a_min=0)
    check_op_input(op, astype="f", offset=3, a_min=0)
    # Complex inputs can be negative
    check_op_input(op, astype="F", out_dtype="D")
    check_op_input(op, randint=True, a_min=3, a_max=10)
    check_op_input(op, shape=(1,), a_min=0.1, offset=3)


odd_root_ops = ("cbrt",)


@pytest.mark.parametrize("op", odd_root_ops)
@deterministic_op_test
def test_odd_root_ops(op):
    check_op(op, np.random.randn(4, 5))
    check_op(op, np.random.randn(4, 5).astype("e"))
    check_op(op, np.random.randn(4, 5).astype("f"))
    check_op(op, np.random.randint(0, 10, size=(4, 5)))
    check_op(op, np.random.randn(1)[0] + 3)


trig_ops = (
    "arccos",
    "arcsin",
    "arctan",
    "arctanh",
    "cos",
    "cosh",
    "deg2rad",
    "rad2deg",
    "sin",
    "sinh",
    "tan",
    "tanh",
)


@pytest.mark.parametrize("op", trig_ops)
@deterministic_op_test
def test_trig_ops(op):
    check_op(op, np.random.uniform(low=-1, high=1, size=(4, 5)))
    check_op(op, np.random.uniform(low=-1, high=1, size=(4, 5)).astype("e"))
    check_op(op, np.array(np.random.uniform(low=-1, high=1)))


arc_hyp_trig_ops = (
    "arccosh",
    "arcsinh",
)


@pytest.mark.parametrize("op", arc_hyp_trig_ops)
@deterministic_op_test
def test_arc_hyp_trig_ops(op):
    check_op(op, np.random.uniform(low=1, high=5, size=(4, 5)))
    check_op(op, np.random.uniform(low=1, high=5, size=(4, 5)).astype("e"))
    check_op(op, np.array(np.random.uniform(low=1, high=5)))


bit_ops = ("invert", "~")


@pytest.mark.parametrize("op", bit_ops)
@deterministic_op_test
def test_bit_ops(op):
    check_op(op, np.random.randint(0, 2, size=(4, 5)))
    check_op(op, np.random.randint(0, 1, size=(4, 5), dtype="?"))


comparison_ops = ("logical_not",)


@pytest.mark.parametrize("op", comparison_ops)
def test_comparison_ops(op):
    check_op(op, np.random.randint(0, 2, size=(4, 5)))


floating_ops = (
    "ceil",
    "floor",
    "signbit",
    # "spacing",
    "trunc",
)


@pytest.mark.parametrize("op", floating_ops)
@deterministic_op_test
def test_floating_ops(op):
    check_op(op, np.random.randn(4, 5))
    check_op(op, np.random.randn(4, 5).astype("f"))
    check_op(op, np.random.randn(4, 5).astype("e"))
    check_op(op, np.random.randint(0, 10, size=(4, 5)))
    check_op(op, np.random.randint(0, 10, size=(4, 5), dtype="I"))
    check_op(op, np.random.randn(1)[0] + 3)


nan_ops = (
    "isfinite",
    "isinf",
    "isnan",
    # "isnat",
)


@pytest.mark.parametrize("op", nan_ops)
def test_nan_ops(op):
    check_op(op, np.array([-np.inf, 0.0, 1.0, np.inf, np.nan]))
    check_op(op, np.array([-np.inf, 0.0, 1.0, np.inf, np.nan], dtype="F"))
    check_op(op, np.array([-np.inf, 0.0, 1.0, np.inf, np.nan], dtype="e"))
    check_op(op, np.array(np.inf))


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
        default="1",
        help="input data",
    )
    parser.add_argument(
        "--dtypes",
        dest="dtypes",
        default="l",
        help="input data",
    )
    args, extra = parser.parse_known_args()

    sys.argv = sys.argv[:1] + extra

    if args.op is not None:
        in_np = parse_inputs(args.inputs, args.dtypes)
        check_ops([args.op], in_np)
    else:
        sys.exit(pytest.main(sys.argv))
