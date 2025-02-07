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

import cupynumeric._utils.array as m  # module under test

EXPECTED_SUPPORTED_DTYPES = set(
    [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
)


class Test_is_advanced_indexing:
    def test_Ellipsis(self):
        assert not m.is_advanced_indexing(...)

    def test_None(self):
        assert not m.is_advanced_indexing(None)

    @pytest.mark.parametrize("typ", EXPECTED_SUPPORTED_DTYPES)
    def test_np_scalar(self, typ):
        assert not m.is_advanced_indexing(typ(10))

    def test_slice(self):
        assert not m.is_advanced_indexing(slice(None, 10))
        assert not m.is_advanced_indexing(slice(1, 10))
        assert not m.is_advanced_indexing(slice(None, 10, 2))

    def test_tuple_False(self):
        assert not m.is_advanced_indexing((..., None, np.int32()))

    def test_tuple_True(self):
        assert m.is_advanced_indexing(([1, 2, 3], np.array([1, 2])))

    def test_advanced(self):
        assert m.is_advanced_indexing([1, 2, 3])
        assert m.is_advanced_indexing(np.array([1, 2, 3]))


def test__SUPPORTED_DTYPES():
    assert set(m.SUPPORTED_DTYPES.keys()) == set(
        np.dtype(ty) for ty in EXPECTED_SUPPORTED_DTYPES
    )


class Test_is_supported_dtype:
    @pytest.mark.parametrize("value", ["foo", 10, 10.2, (), set()])
    def test_type_bad(self, value) -> None:
        with pytest.raises(TypeError):
            m.to_core_type(value)

    @pytest.mark.parametrize("value", EXPECTED_SUPPORTED_DTYPES)
    def test_supported(self, value) -> None:
        m.to_core_type(value)

    # This is just a representative sample, not exhasutive
    @pytest.mark.parametrize("value", [np.float128, np.datetime64, [], {}])
    def test_unsupported(self, value) -> None:
        with pytest.raises(TypeError):
            m.to_core_type(value)


@pytest.mark.parametrize(
    "shape, volume", [[(), 0], [(10,), 10], [(1, 2, 3), 6]]
)
def test_calculate_volume(shape, volume) -> None:
    assert m.calculate_volume(shape) == volume


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
