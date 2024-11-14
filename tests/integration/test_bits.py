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


class TestPackbits(object):
    def test_none_arr(self):
        # Numpy raises "TypeError:
        # Expected an input array of integer or boolean data type"
        # For cuPyNumeric raises:
        #  > if a.dtype.kind not in ("u", "i", "b"):
        #  E AttributeError: 'NoneType' object has no attribute 'dtype'
        with pytest.raises(AttributeError):
            num.packbits(None)

    def test_dtype(self):
        shape = (3, 3)
        in_num = num.random.random(size=shape)
        # TypeError: Expected an input array of integer or boolean data type
        with pytest.raises(TypeError):
            num.packbits(in_num)

    def test_axis_outbound(self):
        shape = (3, 3)
        in_num = num.random.randint(low=0, high=2, size=shape)
        with pytest.raises(ValueError):
            num.packbits(in_num, axis=2)

    @pytest.mark.parametrize("bitorder", (1, True, "True", "BIG", "LITTLE"))
    def test_bitorder_negative(self, bitorder):
        shape = (3, 3)
        in_num = num.random.randint(low=0, high=2, size=shape, dtype="i")
        # when bitorder is 1 or True, Numpy raises
        # "TypeError: pack() argument 3 must be str".
        # while cuPyNumeric raises valueError.
        with pytest.raises(ValueError):
            num.packbits(in_num, bitorder=bitorder)

    @pytest.mark.parametrize("arr", ([], [[]]))
    @pytest.mark.parametrize("dtype", ("B", "i", "?"))
    @pytest.mark.parametrize("bitorder", ("little", "big"))
    def test_arr(self, arr, dtype, bitorder):
        in_np = np.array(arr, dtype=dtype)
        in_num = num.array(arr, dtype=dtype)
        out_np = np.packbits(in_np, bitorder=bitorder)
        out_num = num.packbits(in_num, bitorder=bitorder)
        assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("dtype", ("B", "i", "?"))
    @pytest.mark.parametrize("bitorder", ("little", "big"))
    def test_common(self, ndim, dtype, bitorder):
        shape = (3,) * ndim
        in_np = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
        in_num = num.array(in_np)

        out_np = np.packbits(in_np, bitorder=bitorder)
        out_num = num.packbits(in_num, bitorder=bitorder)
        assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("dtype", ("B", "i", "?"))
    @pytest.mark.parametrize("bitorder", ("little", "big"))
    def test_axis(self, ndim, dtype, bitorder):
        shape = (5,) * ndim
        in_np = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
        in_num = num.array(in_np)

        for axis in range(-ndim + 1, ndim):
            out_np = np.packbits(in_np, axis=axis, bitorder=bitorder)
            out_num = num.packbits(in_num, axis=axis, bitorder=bitorder)
            assert np.array_equal(out_np, out_num)


class TestUnpackbits(object):
    def test_none_arr(self):
        # Numpy raises "TypeError:
        # TypeError: Expected an input array of unsigned byte data type
        # For cuPyNumeric raises:
        # > if a.dtype != "B":
        # E AttributeError: 'NoneType' object has no attribute 'dtype'
        with pytest.raises(AttributeError):
            num.unpackbits(None)

    def test_dtype(self):
        shape = (3, 3)
        in_num = num.random.random(size=shape)
        # TypeError: Expected an input array of unsigned byte data type
        with pytest.raises(TypeError):
            num.unpackbits(in_num)

    def test_axis_outbound(self):
        shape = (3, 3)
        in_np = np.random.randint(low=0, high=255, size=shape, dtype="B")
        in_num = num.array(in_np)
        with pytest.raises(ValueError):
            num.unpackbits(in_num, axis=2)

    @pytest.mark.parametrize("bitorder", (1, True, "True", "BIG", "LITTLE"))
    def test_bitorder_negative(self, bitorder):
        shape = (3, 3)
        in_np = np.random.randint(low=0, high=255, size=shape, dtype="B")
        in_num = num.array(in_np)
        # when bitorder is 1 or True, Numpy raises
        # "TypeError: unpack() argument 4 must be str".
        # while cuPyNumeric raises valueError.
        with pytest.raises(ValueError):
            num.unpackbits(in_num, bitorder=bitorder)

    def test_count_type(self):
        expected_exc = TypeError
        shape = (3, 3)
        in_np = np.random.randint(low=0, high=255, size=shape, dtype="B")
        in_num = num.array(in_np)
        # count must be an integer or None
        with pytest.raises(expected_exc):
            np.unpackbits(in_np, count="1")
        with pytest.raises(expected_exc):
            num.unpackbits(in_num, count="1")

    @pytest.mark.parametrize("arr", ([], [[]]))
    @pytest.mark.parametrize("bitorder", ("little", "big"))
    def test_arr(self, arr, bitorder):
        in_np = np.array(arr, dtype="B")
        in_num = num.array(arr, dtype="B")
        out_np = np.unpackbits(in_np, bitorder=bitorder)
        out_num = num.unpackbits(in_num, bitorder=bitorder)
        assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("bitorder", ("little", "big"))
    def test_common(self, ndim, bitorder):
        shape = (5,) * ndim
        in_np = np.random.randint(low=0, high=255, size=shape, dtype="B")
        in_num = num.array(in_np)

        out_np = np.unpackbits(in_np, bitorder=bitorder)
        out_num = num.unpackbits(in_num, bitorder=bitorder)
        assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("count", (-9, 4, -1, 0, 4, 8, 9))
    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("bitorder", ("little", "big"))
    def test_count(self, ndim, count, bitorder):
        shape = (5,) * ndim
        in_np = np.random.randint(low=0, high=255, size=shape, dtype="B")
        in_num = num.array(in_np)

        out_np = np.unpackbits(in_np, count=count, bitorder=bitorder)
        out_num = num.unpackbits(in_num, count=count, bitorder=bitorder)
        assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("bitorder", ("little", "big"))
    def test_axis(self, ndim, bitorder):
        shape = (5,) * ndim
        in_np = np.random.randint(low=0, high=255, size=shape, dtype="B")
        in_num = num.array(in_np)

        for axis in range(-ndim + 1, ndim):
            out_np = np.unpackbits(in_np, axis=axis, bitorder=bitorder)
            out_num = num.unpackbits(in_num, axis=axis, bitorder=bitorder)
            assert np.array_equal(out_np, out_num)

    @pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
    @pytest.mark.parametrize("bitorder", ("little", "big"))
    @pytest.mark.parametrize("count", (-2, 0, 2, 5))
    def test_axis_count(self, ndim, bitorder, count):
        shape = (5,) * ndim
        in_np = np.random.randint(low=0, high=255, size=shape, dtype="B")
        in_num = num.array(in_np)

        for axis in range(-ndim + 1, ndim):
            out_np = np.unpackbits(
                in_np, count=count, axis=axis, bitorder=bitorder
            )
            out_num = num.unpackbits(
                in_num, count=count, axis=axis, bitorder=bitorder
            )
            assert np.array_equal(out_np, out_num)


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
@pytest.mark.parametrize("bitorder", ("little", "big"))
@pytest.mark.parametrize("dtype", ("B", "i", "?"))
def test_pack_unpack(ndim, bitorder, dtype):
    shape = (8,) * ndim
    in_np = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
    in_num = num.array(in_np)
    for axis in range(ndim):
        out_b = num.packbits(in_num, axis=axis)
        out_p = num.unpackbits(out_b, count=in_num.shape[0], axis=axis)
        assert np.array_equal(in_num, out_p)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
