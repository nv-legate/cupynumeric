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
from legate.core import LEGATE_MAX_DIM

import cupynumeric as num
from cupynumeric.settings import settings

NAN_ARG_FUNCS = ("nanargmax", "nanargmin")

EAGER_TEST = os.environ.get("CUPYNUMERIC_FORCE_THUNK", None) == "eager"

DISALLOWED_DTYPES = (
    np.complex64,
    np.complex128,
)

# Note that when an element is repeated mulitple times in an array,
# the output from cuPyNumeric and numpy will vary. This is expected and
# is not a bug. So, whenever we compare with numpy, we try to make
# sure the elements in the array are unique. Another way to circumvent
# this problem would be to make sure that argument corresponding
# to the max/min is indeed the max/min element in the array


class TestNanArgReductions:
    """
    These are positive cases compared with numpy
    """

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize(
        "ndim",
        (
            1,
            2,
            3,
            4,
            LEGATE_MAX_DIM,
        ),
    )
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic(self, func_name, ndim, keepdims):
        """This test inserts a NaN in the array and checks if the
        output from cuPyNumeric and numpy match
        """
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

        # make sure numpy and cupynumeric give the same out array and max val
        out_np = np.unravel_index(func_np(in_np, keepdims=keepdims), shape)
        out_num = np.unravel_index(func_num(in_num, keepdims=keepdims), shape)

        index_array_np = in_np[out_np]
        index_array_num = in_num[out_num]

        assert np.array_equal(out_num, out_np)
        assert np.array_equal(index_array_num, index_array_np)

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize(
        "ndim",
        (
            1,
            2,
            3,
            4,
            LEGATE_MAX_DIM,
        ),
    )
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

            out_num = num.empty(_shape, dtype=int)
            func_num(in_num, out=out_num, axis=axis, keepdims=True)

            out_np = np.empty(_shape, dtype=int)
            func_np(in_np, out=out_np, axis=axis, keepdims=True)

            assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize(
        "ndim",
        (
            0,
            1,
            2,
            3,
            4,
            LEGATE_MAX_DIM,
        ),
    )
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_floating_point_types(self, func_name, ndim, dtype):
        """This test checks the most frequently used datatypes
        to make sure that the results match with numpy. The
        arrays may or may not contain NaNs.
        """
        shape = (4,) * ndim
        size = prod(shape)

        in_np = np.random.choice(size, size, replace=False) * np.random.rand(1)
        in_np = in_np.astype(dtype)

        in_num = num.array(in_np, dtype=dtype)

        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)

        out_np = func_np(in_np)
        out_num = func_num(in_num)

        assert np.array_equal(out_num, out_np)

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize(
        "ndim",
        (
            1,
            2,
            3,
            4,
            LEGATE_MAX_DIM,
        ),
    )
    def test_all_nan_numpy_compat(self, func_name, ndim):
        """This test checks if we comply with the expected behavior when
        the array contains only NaNs. The expected behavior is to
        raise a ValueError.
        """
        settings.numpy_compat = True

        shape = (3,) * ndim
        in_num = num.zeros(shape)
        in_num.fill(num.nan)

        func_num = getattr(num, func_name)

        expected_exp = ValueError
        with pytest.raises(expected_exp):
            func_num(in_num)

        settings.numpy_compat.unset_value()

    @pytest.mark.skipif(
        EAGER_TEST,
        reason="Eager and Deferred mode will give different results",
    )
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize(
        "ndim",
        (
            1,
            2,
            3,
            4,
            LEGATE_MAX_DIM,
        ),
    )
    def test_all_nan_no_numpy_compat(self, func_name, ndim):
        """This test checks that we return identity for all-NaN arrays.
        Note that scalar reductions (e.g., argmin/argmax) on arrays
        with identities will give different results for single and
        multiple processors.
        """

        settings.numpy_compat = False

        shape = (3,) * ndim
        in_num = num.zeros(shape)
        in_num.fill(num.nan)

        func_num = getattr(num, func_name)

        min_identity = np.iinfo(np.int64).min
        max_identity = np.iinfo(np.int64).max

        # this is a bit of a hack at this point
        assert min_identity == func_num(in_num) or max_identity == func_num(
            in_num
        )

        settings.numpy_compat.unset_value()

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    def test_slice_nan_numpy_compat(self, func_name):
        """This test checks if we comply with the numpy when
        a slice contains only NaNs and CUPYNUMERIC_NUMPY_COMPATABILITY
        is set to 1.
        """
        settings.numpy_compat = True

        in_num = num.random.random((3, 3))
        in_num[0, ...] = num.nan

        func_num = getattr(num, func_name)

        expected_exp = ValueError
        with pytest.raises(expected_exp):
            func_num(in_num, axis=1)

        settings.numpy_compat.unset_value()

    @pytest.mark.skipif(
        EAGER_TEST,
        reason="Eager and Deferred mode will give different results",
    )
    @pytest.mark.parametrize(
        "identity, func_name",
        [
            (np.iinfo(np.int64).min, "nanargmax"),
            (np.iinfo(np.int64).min, "nanargmin"),
        ],
        ids=str,
    )
    def test_slice_nan_no_numpy_compat(self, identity, func_name):
        """This test checks if we return identity for a slice that
        contains NaNs when CUPYNUMERIC_NUMPY_COMPATABILITY is set to 0.
        """
        settings.numpy_compat = False

        in_num = num.random.random((3, 3))
        in_num[0, ...] = num.nan

        func_num = getattr(num, func_name)
        out_num = func_num(in_num, axis=1)

        assert out_num[0] == identity

        settings.numpy_compat.unset_value()

    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    def test_empty_arr(self, func_name: str) -> None:
        a = []
        in_np = np.array(a)
        in_num = num.array(a)
        func_np = getattr(np, func_name)
        func_num = getattr(num, func_name)
        with pytest.raises(ValueError):
            func_np(in_np)
            # ValueError: All-NaN slice encountered
        with pytest.raises(ValueError):
            func_num(in_num)
            # ValueError: attempt to get nanargmax of an empty sequence
            # ValueError: attempt to get nanargmin of an empty sequence


class TestXFail:
    """
    This class is to test negative cases
    """

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    @pytest.mark.parametrize(
        "ndim",
        (
            0,
            1,
            2,
            3,
            4,
            LEGATE_MAX_DIM,
        ),
    )
    @pytest.mark.parametrize("disallowed_dtype", DISALLOWED_DTYPES)
    def test_disallowed_dtypes(self, func_name, ndim, disallowed_dtype):
        """This test checks if we raise an error for types that are
        disallowed."""
        shape = (2,) * ndim
        in_num = num.random.random(shape).astype(disallowed_dtype)

        func_num = getattr(num, func_name)

        expected_exp = ValueError
        with pytest.raises(expected_exp):
            func_num(in_num)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    def test_out_shape_mismatch(self, func_name):
        """This test checks if we raise an error when the output shape is
        incorrect"""

        in_num = np.random.random((3, 3))
        func_num = getattr(num, func_name)

        # shape
        index_array = np.random.randint(low=1, high=4, size=(2, 2))
        func_num(in_num, out=index_array)

    @pytest.mark.xfail
    @pytest.mark.parametrize("func_name", NAN_ARG_FUNCS)
    def test_out_dtype_mismatch(self, func_name):
        """This test checks if we raise an error when the output dtype is
        incorrect"""

        in_num = np.random.random((3, 3))
        func_num = getattr(num, func_name)

        # dtype
        index_array = np.random.random(2, 2)
        func_num(in_num, out=index_array)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
