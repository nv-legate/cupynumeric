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

import functools

import numpy as np
import pytest
from utils.comparisons import allclose

import cupynumeric as num

np.random.seed(143)


def check_result(in_np, out_np, out_num, **isclose_kwargs):
    if in_np.dtype == "e" or out_np.dtype == "e":
        # The mantissa is only 10 bits, 2**-10 ~= 10^(-4)
        # Gives 1e-3 as rtol to provide extra rounding error.
        f16_rtol = 1e-2
        rtol = isclose_kwargs.setdefault("rtol", f16_rtol)
        # make sure we aren't trying to fp16 compare with less precision
        assert rtol >= f16_rtol

    if "negative_test" in isclose_kwargs:
        is_negative_test = isclose_kwargs["negative_test"]
    else:
        is_negative_test = False

    result = (
        allclose(out_np, out_num, equal_nan=True, **isclose_kwargs)
        and out_np.dtype == out_num.dtype
    )
    if not result and not is_negative_test:
        print("cupynumeric failed the test")
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


def check_op(op_np, op_num, in_np, out_dtype, **check_kwargs):
    in_num = num.array(in_np)

    out_np = op_np(in_np)
    out_num = op_num(in_num)

    assert check_result(in_np, out_np, out_num, **check_kwargs)

    out_np = np.empty(out_np.shape, dtype=out_dtype)
    out_num = num.empty(out_num.shape, dtype=out_dtype)

    op_np(in_np, out=out_np)
    op_num(in_num, out=out_num)

    assert check_result(in_np, out_np, out_num, **check_kwargs)


def get_op_input(
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

    return in_np


dtypes = ("e", "f", "d")


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_var_default_shape(dtype, ddof, axis, keepdims):
    np_in = get_op_input(astype=dtype)

    op_np = functools.partial(np.var, ddof=ddof, axis=axis, keepdims=keepdims)
    op_num = functools.partial(
        num.var, ddof=ddof, axis=axis, keepdims=keepdims
    )

    check_op(op_np, op_num, np_in, dtype)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_var_where(dtype, ddof, axis, keepdims):
    np_in = get_op_input(astype=dtype)
    where = (np_in.astype(int) % 2).astype(bool)

    op_np = functools.partial(
        np.var, ddof=ddof, axis=axis, keepdims=keepdims, where=where
    )
    op_num = functools.partial(
        num.var, ddof=ddof, axis=axis, keepdims=keepdims, where=where
    )

    check_op(op_np, op_num, np_in, dtype)


def _max_axis(obj) -> bool:
    if isinstance(obj, tuple):
        return max(obj)
    return obj


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize(
    "axis", [None, 0, 1, 2, pytest.param((0, 1), marks=pytest.mark.xfail)]
)
@pytest.mark.parametrize("shape", [(10,), (4, 5), (2, 3, 4)])
def test_var_w_shape(dtype, ddof, axis, shape):
    np_in = get_op_input(astype=dtype, shape=shape)

    if axis is not None and _max_axis(axis) >= len(shape):
        axis = None

    op_np = functools.partial(np.var, ddof=ddof, axis=axis)
    op_num = functools.partial(num.var, ddof=ddof, axis=axis)

    check_op(op_np, op_num, np_in, dtype)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("axis", [None])
@pytest.mark.parametrize("shape", [(10, 1)])
def test_var_corners(dtype, ddof, axis, shape):
    np_in = get_op_input(astype=dtype, shape=shape)

    if axis is not None and _max_axis(axis) >= len(shape):
        axis = None

    op_np = functools.partial(np.var, ddof=ddof, axis=axis)
    op_num = functools.partial(num.var, ddof=ddof, axis=axis)

    check_op(op_np, op_num, np_in, dtype)


@pytest.mark.xfail
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("axis", [None])
@pytest.mark.parametrize("shape", [(1,)])
def test_var_xfail(dtype, ddof, axis, shape):
    np_in = get_op_input(astype=dtype, shape=shape)

    op_np = functools.partial(np.var, ddof=ddof, axis=axis)
    op_num = functools.partial(num.var, ddof=ddof, axis=axis)

    check_op(op_np, op_num, np_in, dtype, negative_test=True)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("rowvar", [True, False])
@pytest.mark.parametrize("ddof", [None, 0, 1])
def test_cov(dtype, rowvar, ddof):
    np_in = get_op_input(astype=dtype)
    num_in = num.array(np_in)

    np_out = np.cov(np_in, rowvar=rowvar, ddof=ddof)
    num_out = num.cov(num_in, rowvar=rowvar, ddof=ddof)
    if dtype == dtypes[0]:
        assert allclose(np_out, num_out, atol=1e-2)
    else:
        assert allclose(np_out, num_out)


fweights_base = [[9, 2, 1, 2, 3], [1, 1, 3, 2, 4], None]
np_aweights_base = [
    np.abs(get_op_input(astype=dtype, shape=(5,))) for dtype in dtypes
] + [[0.03, 0.04, 01.01, 0.02, 0.08], None]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("ddof", [None, 0, 1])
@pytest.mark.parametrize("fweights", fweights_base)
@pytest.mark.parametrize("np_aweights", np_aweights_base)
def test_cov_full(dtype, bias, ddof, fweights, np_aweights):
    np_in = get_op_input(astype=dtype, shape=(4, 5))
    num_in = num.array(np_in)
    if fweights is not None:
        np_fweights = np.array(fweights)
        num_fweights = num.array(fweights)
    else:
        np_fweights = None
        num_fweights = None
    if isinstance(np_aweights, np.ndarray):
        num_aweights = num.array(np_aweights)
    else:
        num_aweights = np_aweights
    # num_aweights = None
    # np_aweights = None

    np_out = np.cov(
        np_in, bias=bias, ddof=ddof, fweights=np_fweights, aweights=np_aweights
    )
    num_out = num.cov(
        num_in,
        bias=bias,
        ddof=ddof,
        fweights=num_fweights,
        aweights=num_aweights,
    )
    # if dtype == dtypes[0]:
    #     assert allclose(np_out, num_out, atol=1e-2)
    # else:
    #     assert allclose(np_out, num_out)
    assert allclose(np_out, num_out, atol=1e-2)


@pytest.mark.parametrize("ddof", [None, 0, 1])
@pytest.mark.parametrize("fweights", fweights_base)
@pytest.mark.parametrize("np_aweights", np_aweights_base)
def test_cov_dtype_scaling(ddof, fweights, np_aweights):
    np_in = np.array(
        [
            [1 + 3j, 1 - 1j, 2 + 2j, 4 + 3j, -1 + 2j],
            [1 + 3j, 1 - 1j, 2 + 2j, 4 + 3j, -1 + 2j],
        ]
    )
    num_in = num.array(np_in)
    if fweights is not None:
        np_fweights = np.array(fweights)
        num_fweights = num.array(fweights)
    else:
        np_fweights = None
        num_fweights = None
    if isinstance(np_aweights, np.ndarray):
        num_aweights = num.array(np_aweights)
    else:
        num_aweights = np_aweights

    np_out = np.cov(
        np_in, ddof=ddof, fweights=np_fweights, aweights=np_aweights
    )
    num_out = num.cov(
        num_in, ddof=ddof, fweights=num_fweights, aweights=num_aweights
    )
    assert allclose(np_out, num_out, atol=1e-2)


def test_cov_invalid_ddof() -> None:
    np_in = get_op_input(astype="d")
    num_in = num.array(np_in)
    msg = "ddof must be integer"
    with pytest.raises(ValueError, match=msg):
        np.cov(np_in, ddof=2.5)
    with pytest.raises(ValueError, match=msg):
        num.cov(num_in, ddof=2.5)


def test_cov_invalid_input() -> None:
    np_in = np.random.randn(3, 3, 3)
    num_in = num.array(np_in)

    msg = "m has more than 2 dimensions"
    with pytest.raises(ValueError, match=msg):
        np.cov(np_in)
    with pytest.raises(ValueError, match=msg):
        num.cov(num_in)


def test_cov_invalid_input2() -> None:
    np_in = get_op_input(astype="d")
    num_in = num.array(np_in)

    np_in_y = np.random.randn(3, 3, 3)
    num_in_y = num.array(np_in_y)
    msg = "y has more than 2 dimensions"
    with pytest.raises(ValueError, match=msg):
        np.cov(np_in, np_in_y)
    with pytest.raises(ValueError, match=msg):
        num.cov(num_in, num_in_y)


def test_cov_dtype() -> None:
    np_in = get_op_input(astype="d")
    num_in = num.array(np_in)

    np_out = np.cov(np_in, dtype=np.float64)
    num_out = num.cov(num_in, dtype=np.float64)

    assert allclose(np_out, num_out, atol=1e-2)


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("ddof", [None, 0, 1])
def test_cov_input2(dtype: str, bias: bool, ddof: int) -> None:
    np_in = get_op_input(astype=dtype, shape=(4, 5))
    num_in = num.array(np_in)

    np_in_y = get_op_input(astype=dtype, shape=(4, 5))
    num_in_y = num.array(np_in_y)

    np_out = np.cov(np_in, np_in_y, bias=bias, ddof=ddof)
    num_out = num.cov(num_in, num_in_y, bias=bias, ddof=ddof)
    assert allclose(np_out, num_out, atol=1e-2)


def test_cov_shape0() -> None:
    np_in = np.random.randn(0, 3)
    num_in = num.array(np_in)

    np_out = np.cov(np_in)
    num_out = num.cov(num_in)
    assert allclose(np_out, num_out)


def test_cov_invalid_fweights() -> None:
    np_in = get_op_input(astype="d", shape=(4, 5))
    num_in = num.array(np_in)
    fweights = [[1, 2], [2, 3]]
    np_fweights = np.array(fweights)
    num_fweights = num.array(fweights)

    msg = "cannot handle multidimensional fweights"
    with pytest.raises(RuntimeError, match=msg):
        np.cov(np_in, fweights=np_fweights)
    with pytest.raises(RuntimeError, match=msg):
        num.cov(num_in, fweights=num_fweights)


def test_cov_invalidshape_fweights() -> None:
    np_in = get_op_input(astype="d", shape=(4, 5))
    num_in = num.array(np_in)
    fweights = [1, 2, 3]

    np_fweights = np.array(fweights)
    num_fweights = num.array(fweights)
    msg = "incompatible numbers of samples and fweights"
    with pytest.raises(RuntimeError, match=msg):
        np.cov(np_in, fweights=np_fweights)
    with pytest.raises(RuntimeError, match=msg):
        num.cov(num_in, fweights=num_fweights)


def test_cov_invalidsize_fweights() -> None:
    np_in = get_op_input(astype="d", shape=(4, 5))
    num_in = num.array(np_in)
    fweights = [1, 2, 3, -1, 3]

    np_fweights = np.array(fweights)
    num_fweights = num.array(fweights)
    msg = "fweights cannot be negative"
    with pytest.raises(ValueError, match=msg):
        np.cov(np_in, fweights=np_fweights)
    with pytest.raises(ValueError, match=msg):
        num.cov(num_in, fweights=num_fweights)


def test_cov_invalid_aweights() -> None:
    np_in = get_op_input(astype="d", shape=(4, 5))
    num_in = num.array(np_in)
    aweights = [[1, 2], [2, 3]]
    np_aweights = np.array(aweights)
    num_aweights = num.array(aweights)
    msg = "cannot handle multidimensional aweights"
    with pytest.raises(RuntimeError, match=msg):
        np.cov(np_in, aweights=np_aweights)
    with pytest.raises(RuntimeError, match=msg):
        num.cov(num_in, aweights=num_aweights)


def test_cov_invalidshape_aweights() -> None:
    np_in = get_op_input(astype="d", shape=(4, 5))
    num_in = num.array(np_in)
    aweights = [1, 2, 3]

    np_aweights = np.array(aweights)
    num_aweights = num.array(aweights)
    msg = "incompatible numbers of samples and aweights"
    with pytest.raises(RuntimeError, match=msg):
        np.cov(np_in, aweights=np_aweights)
    with pytest.raises(RuntimeError, match=msg):
        num.cov(num_in, aweights=num_aweights)


def test_cov_invalidsize_aweights() -> None:
    np_in = get_op_input(astype="d", shape=(4, 5))
    num_in = num.array(np_in)
    aweights = [1, 2, 3, -1, 3]

    np_aweights = np.array(aweights)
    num_aweights = num.array(aweights)
    msg = "aweights cannot be negative"
    with pytest.raises(ValueError, match=msg):
        np.cov(np_in, aweights=np_aweights)
    with pytest.raises(ValueError, match=msg):
        num.cov(num_in, aweights=num_aweights)


if __name__ == "__main__":
    import sys

    np.random.seed(12345)

    sys.exit(pytest.main(sys.argv))
