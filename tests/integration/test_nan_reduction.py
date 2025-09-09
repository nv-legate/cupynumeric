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

import os
from math import prod

import numpy as np
import pytest
from utils.comparisons import allclose
from utils.utils import MAX_DIM_RANGE, ONE_MAX_DIM_RANGE

import cupynumeric as num
from cupynumeric.settings import settings

NAN_FUNCS = ("nanmax", "nanmin", "nanprod", "nansum")

EAGER_TEST = os.environ.get("CUPYNUMERIC_FORCE_THUNK", None) == "eager"

NDIMS = MAX_DIM_RANGE

DTYPE = ["l", "L", "f", "d", "h", "i", "H", "I", "?", "b", "B"]


def to_dtype(s):
    return str(np.dtype(s))


class TestNanReductions:
    """
    These are positive cases compared with numpy
    """

    @pytest.mark.parametrize("func_name", ("nansum", "nanprod"))
    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic_nan_sum_prod(self, func_name, ndim, keepdims):
        """This test sets an element to NaN and checks if the output
        from cuPyNumeric and numpy match."""
        shape = (5,) * ndim
        size = prod(shape)
        in_np = np.random.random(shape)

        # set an element to nan
        index_nan = np.random.randint(low=0, high=size)
        index_nan = np.unravel_index(index_nan, shape)
        in_num = num.array(in_np)
        in_num[index_nan] = num.nan
        in_np[index_nan] = np.nan

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        out_np = func_np(in_np, keepdims=keepdims)
        out_num = func_num(in_num, keepdims=keepdims)

        assert allclose(out_num, out_np, rtol=1e-4)

    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic_nan_min_max(self, func_name, ndim, keepdims):
        """This test sets an element to NaN and checks if the output
        from cuPyNumeric and numpy match."""
        shape = (5,) * ndim
        size = prod(shape)
        in_np = np.random.random(shape)

        # set an element to nan
        index_nan = np.random.randint(low=0, high=size)
        index_nan = np.unravel_index(index_nan, shape)
        in_num = num.array(in_np)
        in_num[index_nan] = num.nan
        in_np[index_nan] = np.nan

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        out_np = func_np(in_np, keepdims=keepdims)
        out_num = func_num(in_num, keepdims=keepdims)

        assert np.array_equal(out_num, out_np)

    @pytest.mark.parametrize("func_name", NAN_FUNCS)
    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    def test_out(self, func_name, ndim):
        """This test checks that the out argument is updated with the
        output"""

        shape = (3,) * ndim
        in_np = np.random.random(shape)
        in_num = num.array(in_np)

        func_num = getattr(num, func_name)
        func_np = getattr(np, func_name)

        axes = list(range(0, ndim - 1))
        for axis in axes:
            _shape = list(shape)
            _shape[axis] = 1

            out_num = num.empty(_shape)
            func_num(in_num, out=out_num, axis=axis, keepdims=True)

            out_np = np.empty(_shape)
            func_np(in_np, out=out_np, axis=axis, keepdims=True)

            assert allclose(out_num, out_np, rtol=1e-4)

    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_complex_dtype_nansum(self, ndim, dtype, keepdims):
        """This test checks if nansum works as expected for complex
        datatypes. The parametrized datatype is that of real/complex
        component, so float32 would correspond to complex64 and
        float64 would correspond to complex128.
        """
        shape = (3,) * ndim
        size = prod(shape)

        r = np.random.random(shape).astype(dtype)
        c = np.random.random(shape).astype(dtype)

        # get index of nan
        index_nan = np.random.randint(low=0, high=size)
        index_nan = np.unravel_index(index_nan, shape)

        # set an element to nan
        r[index_nan] = np.nan
        c[index_nan] = np.nan

        in_np = r + 1j * c

        in_num = num.array(in_np)
        in_num[index_nan] = num.nan

        out_num = num.nansum(in_num, keepdims=keepdims)
        out_np = np.nansum(in_np, keepdims=keepdims)

        assert allclose(out_num, out_np, rtol=1e-4)

    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_complex_dtype_nanprod(self, ndim, keepdims):
        """This test checks if nanprod works as expected for complex
        datatypes. The parametrized datatype is that of real/complex
        component, so float32 would correspond to complex64.
        """
        shape = (3,) * ndim
        size = prod(shape)

        dtype = np.float32

        r = np.random.random(shape).astype(dtype)
        c = np.random.random(shape).astype(dtype)

        # get index of nan
        index_nan = np.random.randint(low=0, high=size)
        index_nan = np.unravel_index(index_nan, shape)

        # set just the real component to nan
        r[index_nan] = np.nan

        # set an element to nan
        in_np = r + 1j * c

        in_num = num.array(in_np)

        out_num = num.nanprod(in_num, keepdims=keepdims)
        out_np = np.nanprod(in_np, keepdims=keepdims)

        assert allclose(out_num, out_np, rtol=1e-4)

    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    def test_slice_nan_numpy_compat(self, func_name):
        """This test checks if nanmin and nanmax return nan for
        a slice that contains all-NaNs
        """
        settings.numpy_compat = True

        shape = (3, 3)
        in_num = num.random.random(shape)

        in_num[0, :] = num.nan
        func_num = getattr(num, func_name)
        out_num = func_num(in_num, axis=1)

        assert num.any(num.isnan(out_num))

        settings.numpy_compat.unset_value()

    @pytest.mark.skipif(
        EAGER_TEST,
        reason="Eager and Deferred mode will give different results",
    )
    @pytest.mark.parametrize(
        "identity, func_name",
        [
            (np.finfo(np.float64).min, "nanmax"),
            (np.finfo(np.float64).max, "nanmin"),
        ],
        ids=str,
    )
    def test_slice_nan_no_numpy_compat(self, identity, func_name):
        """This test checks if nanmin and nanmax return identity for
        a slice that contains all-NaNs
        """
        settings.numpy_compat = False

        in_num = num.random.random((3, 3))

        in_num[0, :] = num.nan
        func_num = getattr(num, func_name)
        out_num = func_num(in_num, axis=1)

        assert out_num[0] == identity

        settings.numpy_compat.unset_value()

    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    def test_all_nans_numpy_compat(self, ndim, func_name):
        """This test checks if we comply with the expected behavior when
        the array contains only NaNs.
        """
        settings.numpy_compat = True

        shape = (3,) * ndim
        in_num = num.random.random(shape)
        in_num[...] = np.nan

        func_num = getattr(num, func_name)
        out_num = func_num(in_num)

        assert num.isnan(out_num)

        settings.numpy_compat.unset_value()

    @pytest.mark.skipif(
        EAGER_TEST,
        reason="Eager and Deferred mode will give different results",
    )
    @pytest.mark.parametrize(
        "identity, func_name",
        [
            (np.finfo(np.float64).min, "nanmax"),
            (np.finfo(np.float64).max, "nanmin"),
        ],
        ids=str,
    )
    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    def test_all_nans_no_numpy_compat(self, ndim, identity, func_name):
        """This test checks if we comply with the expected behavior when
        the array contains only NaNs for nanmin and nanmax.
        """
        settings.numpy_compat = False

        shape = (3,) * ndim
        in_num = num.random.random(shape)
        in_num[...] = np.nan

        func_num = getattr(num, func_name)
        out_num = func_num(in_num)

        assert out_num == identity

        settings.numpy_compat.unset_value()

    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    def test_all_nans_nanprod(self, ndim):
        shape = (3,) * ndim
        in_num = num.random.random(shape)
        in_num[...] = np.nan

        out_num = num.nanprod(in_num)

        assert out_num == 1.0

    @pytest.mark.parametrize("dtype", DTYPE, ids=to_dtype)
    def test_dtype_nanprod(self, dtype) -> None:
        in_np = np.arange(1, 10)
        if dtype is bool:
            in_np %= 2
        in_np = in_np.astype(dtype)
        out_np = np.nanprod(in_np)
        in_num = num.asarray(in_np)
        out_num = num.nanprod(in_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("dtype", DTYPE, ids=to_dtype)
    def test_dtype_nansum(self, dtype) -> None:
        in_np = np.arange(1, 10)
        if dtype is bool:
            in_np %= 2
        in_np = in_np.astype(dtype)
        out_np = np.nansum(in_np)
        in_num = num.asarray(in_np)
        out_num = num.nansum(in_num)
        assert allclose(out_np, out_num)

    @pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
    def test_all_nans_nansum(self, ndim):
        shape = (3,) * ndim
        in_num = num.random.random(shape)
        in_num[...] = np.nan

        out_num = num.nansum(in_num)

        assert out_num == 0.0

    def test_where(self):
        arr = [[1, np.nan, 3], [2, np.nan, 4]]
        out_np = np.nansum(arr, where=[False, True, True])
        out_num = num.nansum(arr, where=[False, True, True])
        assert allclose(out_np, out_num)

        out_np = np.nanprod(arr, where=[False, True, True])
        out_num = num.nanprod(arr, where=[False, True, True])
        assert allclose(out_np, out_num)

        out_np = np.nanmax(
            arr, where=[[False, True, True], [False, False, True]], initial=-1
        )
        out_num = num.nanmax(
            arr, where=[[False, True, True], [False, False, True]], initial=-1
        )
        assert allclose(out_np, out_num)

        out_np = np.nanmin(
            arr, where=[[False, True, True], [False, True, True]], initial=10
        )
        out_num = num.nanmin(
            arr, where=[[False, True, True], [False, True, True]], initial=10
        )
        assert allclose(out_np, out_num)

        # where is a boolean
        out_np = np.nansum(arr, where=True)
        out_num = num.nansum(arr, where=True)
        assert allclose(out_np, out_num)

        out_np = np.nanprod(arr, where=False)
        out_num = num.nanprod(arr, where=False)
        assert allclose(out_np, out_num)

        out_np = np.nanmax(arr, where=True, initial=-1)
        out_num = num.nanmax(arr, where=True, initial=-1)
        assert allclose(out_np, out_num)

        out_np = np.nanmin(arr, where=True, initial=10)
        out_num = num.nanmin(arr, where=True, initial=10)
        assert allclose(out_np, out_num)

    def test_nanmin_non_nan(self) -> None:
        arr = np.array([3, 1, 2], dtype=np.int32)
        result_num = num.nanmin(arr)
        result_np = np.nanmin(arr)
        assert result_num == result_np


class TestCornerCases:
    """
    These are corner cases where we check with empty arrays
    """

    def test_empty_for_nanmin(self):
        expected_exp = ValueError
        with pytest.raises(expected_exp):
            num.nanmin([])

    def test_empty_for_nanmax(self):
        expected_exp = ValueError
        with pytest.raises(expected_exp):
            num.nanmax([])

    def test_empty_for_prod_sum(self):
        assert num.nanprod([]) == 1.0
        assert num.nansum([]) == 0.0


class TestXFail:
    """
    This class is to test negative cases
    """

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    @pytest.mark.parametrize("ndim", NDIMS)
    @pytest.mark.parametrize("disallowed_dtype", (np.complex64, np.complex128))
    def test_disallowed_dtypes(self, func_name, ndim, disallowed_dtype):
        """This test checks if we raise an error for types that are
        disallowed."""
        shape = (3,) * ndim
        in_num = num.random.random(shape).astype(disallowed_dtype)

        func_num = getattr(num, func_name)

        expected_exp = NotImplementedError
        msg = r"operation is not supported for complex64 and complex128 types"
        with pytest.raises(expected_exp, match=msg):
            func_num(in_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", ("nanmin", "nanmax"))
    def test_disallowed_dtype_nan_min_max(self, func_name):
        ndim = 1
        shape = (3,) * ndim
        dtype = np.float32

        r = num.random.random(shape).astype(dtype)
        c = num.random.random(shape).astype(dtype)
        r[0] = c[0] = num.nan
        in_num = r + 1j * c

        func_num = getattr(num, func_name)

        expected_exp = NotImplementedError
        with pytest.raises(expected_exp):
            func_num(in_num)

    @pytest.mark.xfail
    def test_disallowed_dtype_nanprod(self):
        ndim = 1
        shape = (3,) * ndim
        dtype = np.float64

        r = num.random.random(shape).astype(dtype)
        c = num.random.random(shape).astype(dtype)
        r[0] = c[0] = num.nan
        in_num = r + 1j * c

        expected_exp = NotImplementedError
        with pytest.raises(expected_exp):
            num.nanprod(in_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
