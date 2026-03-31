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


class Test_is_true_unoptimized_advanced_indexing:
    # --- basic indexing → False ---

    def test_scalar(self):
        assert not m.is_true_unoptimized_advanced_indexing(0, ndim=1)

    def test_slice(self):
        assert not m.is_true_unoptimized_advanced_indexing(
            slice(None, 10), ndim=1
        )

    def test_ellipsis(self):
        assert not m.is_true_unoptimized_advanced_indexing(..., ndim=2)

    def test_tuple_of_scalars(self):
        assert not m.is_true_unoptimized_advanced_indexing((0, 1), ndim=2)

    # --- single integer array, all slice(None), ndim < 5 → einsum → False ---

    def test_int_array_1d_low_ndim(self):
        idx = np.array([0, 2, 1])
        assert not m.is_true_unoptimized_advanced_indexing(idx, ndim=1)

    def test_int_array_row_select_ndim2(self):
        idx = np.array([0, 2])
        assert not m.is_true_unoptimized_advanced_indexing(
            (idx, slice(None)), ndim=2
        )

    def test_int_array_col_select_ndim2(self):
        idx = np.array([0, 2])
        assert not m.is_true_unoptimized_advanced_indexing(
            (slice(None), idx), ndim=2
        )

    def test_int_list_row_select_ndim4(self):
        assert not m.is_true_unoptimized_advanced_indexing(
            ([0, 1], slice(None), slice(None), slice(None)), ndim=4
        )

    # --- single integer array, ndim >= 5 → gather → True ---

    def test_int_array_ndim5(self):
        idx = np.array([0, 2])
        assert m.is_true_unoptimized_advanced_indexing(
            (idx, slice(None), slice(None), slice(None), slice(None)), ndim=5
        )

    # --- solo boolean array → ADVANCED_INDEXING task only, no gather/scatter → False ---

    def test_bool_array_1d_solo(self):
        mask = np.array([True, False, True])
        assert not m.is_true_unoptimized_advanced_indexing(mask, ndim=1)

    def test_bool_array_2d_solo(self):
        mask = np.ones((3, 3), dtype=bool)
        assert not m.is_true_unoptimized_advanced_indexing(mask, ndim=2)

    # --- boolean array with co-keys → nonzero + ZIP + gather/scatter → True ---

    def test_bool_array_row_with_slice(self):
        # a[mask, :] — falls through to nonzero + ZIP + gather
        mask = np.array([True, False, True])
        assert m.is_true_unoptimized_advanced_indexing(
            (mask, slice(None)), ndim=2
        )

    def test_bool_array_col_position(self):
        # a[:, mask] — same fallthrough
        mask = np.array([True, False, True])
        assert m.is_true_unoptimized_advanced_indexing(
            (slice(None), mask), ndim=2
        )

    # --- multiple advanced components → ZIP + gather/scatter → True ---

    def test_two_int_arrays(self):
        idx = np.array([0, 1])
        assert m.is_true_unoptimized_advanced_indexing((idx, idx), ndim=2)

    def test_int_array_and_bool_array(self):
        idx = np.array([0, 1])
        mask = np.array([True, False, True])
        assert m.is_true_unoptimized_advanced_indexing((idx, mask), ndim=2)

    # --- non-trivial slice alongside advanced component → gather/scatter → True ---

    def test_int_array_with_nontrivial_slice(self):
        idx = np.array([0, 1])
        assert m.is_true_unoptimized_advanced_indexing(
            (idx, slice(None, 5)), ndim=2
        )

    def test_int_array_with_step_slice(self):
        idx = np.array([0, 1])
        assert m.is_true_unoptimized_advanced_indexing(
            (idx, slice(None, None, 2)), ndim=2
        )


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
