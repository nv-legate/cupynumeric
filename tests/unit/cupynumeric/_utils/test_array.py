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

    @pytest.mark.parametrize("typ", EXPECTED_SUPPORTED_DTYPES - {np.bool_})
    def test_np_scalar(self, typ):
        assert not m.is_advanced_indexing(typ(10))

    @pytest.mark.parametrize(
        "scalar", [True, False, np.bool_(True), np.bool_(False)]
    )
    def test_bool_scalar(self, scalar):
        assert m.is_advanced_indexing(scalar)

    @pytest.mark.parametrize("scalar", [np.array(True), np.array(False)])
    def test_bool_0d_array(self, scalar):
        assert m.is_advanced_indexing(scalar)

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

    @pytest.mark.parametrize(
        "scalar", [True, False, np.bool_(True), np.bool_(False)]
    )
    def test_bool_scalar(self, scalar):
        assert m.is_true_unoptimized_advanced_indexing(scalar, ndim=1)

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

    # --- boolean array + all-slice(None) co-keys → BoolMask (ADVANCED_INDEXING task) → False ---

    def test_bool_array_row_with_slice(self):
        # a[mask, :] — BoolMask(transpose_index=0), no gather
        mask = np.array([True, False, True])
        assert not m.is_true_unoptimized_advanced_indexing(
            (mask, slice(None)), ndim=2
        )

    def test_bool_array_col_position(self):
        # a[:, mask] — BoolMask(transpose_index=1), no gather
        mask = np.array([True, False, True])
        assert not m.is_true_unoptimized_advanced_indexing(
            (slice(None), mask), ndim=2
        )

    # --- SET into a non-leading boolean mask is conservatively flagged as
    #     unoptimized (multi-process scalar `a[:, mask] = v` still runs
    #     nonzero + ZIP + scatter); the GET of the same key stays optimized ---

    def test_bool_col_position_set_is_unoptimized(self):
        # a[:, mask] = v — non-leading bool SET → conservatively unoptimized
        mask = np.array([True, False, True])
        assert m.is_true_unoptimized_advanced_indexing(
            (slice(None), mask), ndim=2, is_set=True
        )

    def test_bool_row_with_slice_set_stays_optimized(self):
        # a[mask, :] = v — leading bool mask uses putmask, stays optimized
        mask = np.array([True, False, True])
        assert not m.is_true_unoptimized_advanced_indexing(
            (mask, slice(None)), ndim=2, is_set=True
        )

    # --- Ellipsis co-keys are normalized to slice(None): a[..., mask] must
    #     classify identically to the explicit a[:, mask] spelling ---

    def test_bool_ellipsis_equivalent_to_slice_get(self):
        # a[..., mask] on a 2-D array == a[:, mask]: both optimized (GET).
        mask = np.array([True, False, True])
        got = m.is_true_unoptimized_advanced_indexing((..., mask), ndim=2)
        explicit = m.is_true_unoptimized_advanced_indexing(
            (slice(None), mask), ndim=2
        )
        assert got == explicit
        assert not got

    def test_bool_ellipsis_equivalent_to_slice_set(self):
        # a[..., mask] = v == a[:, mask] = v: both conservatively unoptimized.
        mask = np.array([True, False, True])
        got = m.is_true_unoptimized_advanced_indexing(
            (..., mask), ndim=2, is_set=True
        )
        explicit = m.is_true_unoptimized_advanced_indexing(
            (slice(None), mask), ndim=2, is_set=True
        )
        assert got == explicit
        assert got

    def test_bool_ellipsis_leading_when_no_free_dims(self):
        # a[..., mask] on a 1-D array: Ellipsis expands to zero slices, so the
        # mask is leading (== a[mask]) → optimized for both GET and SET.
        mask = np.array([True, False, True])
        assert not m.is_true_unoptimized_advanced_indexing((..., mask), ndim=1)
        assert not m.is_true_unoptimized_advanced_indexing(
            (..., mask), ndim=1, is_set=True
        )

    def test_bool_ellipsis_2d_mask_equals_solo(self):
        # a[..., mask2d] on a 2-D array == a[mask2d] (NumPy): the 2-D mask
        # consumes both axes, so the Ellipsis fills zero slices and it stays the
        # solo boolean-mask case → optimized.  (a[:, mask2d], with an explicit
        # slice, is the invalid/unoptimized case — see test_bool_2d_mask_*.)
        mask2d = np.ones((2, 3), dtype=bool)
        assert not m.is_true_unoptimized_advanced_indexing(
            (..., mask2d), ndim=2
        )
        assert not m.is_true_unoptimized_advanced_indexing(mask2d, ndim=2)

    # --- multidimensional boolean mask: leading axis stays on BoolMask (False),
    #     non-leading axis falls back to nonzero + ZIP + gather (True) ---

    def test_bool_2d_mask_leading_axis(self):
        # a[mask2d, :] — leading multidim mask → BoolMask(transpose_index=0)
        mask2d = np.zeros((3, 4), dtype=bool)
        assert not m.is_true_unoptimized_advanced_indexing(
            (mask2d, slice(None)), ndim=3
        )

    def test_bool_2d_mask_nonleading_axis(self):
        # a[:, mask2d] — non-leading multidim mask →
        # _prepare_boolean_array_indexing returns None → nonzero + ZIP + gather
        mask2d = np.zeros((3, 4), dtype=bool)
        assert m.is_true_unoptimized_advanced_indexing(
            (slice(None), mask2d), ndim=3
        )

    # --- boolean array with non-slice(None) co-keys → nonzero + ZIP + gather → True ---

    def test_bool_array_row_with_newaxis(self):
        # a[mask, np.newaxis] — newaxis co-key → General path
        mask = np.array([True, False, True])
        assert m.is_true_unoptimized_advanced_indexing((mask, None), ndim=2)

    def test_bool_array_with_scalar_co_key(self):
        # a[mask, 0] — scalar co-key → General path
        mask = np.array([True, False, True])
        assert m.is_true_unoptimized_advanced_indexing((mask, 0), ndim=2)

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


class Test_expand_ellipsis:
    # _expand_ellipsis counts the source axes consumed by each key entry so the
    # Ellipsis expands to exactly the remaining ones.

    @pytest.mark.parametrize("scalar", [True, False, np.bool_(True)])
    def test_bool_scalar_consumes_no_axis(self, scalar):
        # A plain Python bool / np.bool_ scalar adds a dimension like
        # np.newaxis, so it must NOT count against the axes the Ellipsis fills:
        # a[..., True] on a 2-D array expands the Ellipsis to both axes.
        result = m._expand_ellipsis((..., scalar), 2)
        assert len(result) == 3
        assert result[0] == slice(None) and result[1] == slice(None)
        assert result[2] is scalar

    def test_newaxis_consumes_no_axis(self):
        result = m._expand_ellipsis((..., np.newaxis), 2)
        assert len(result) == 3 and result[2] is np.newaxis

    def test_int_scalar_consumes_one_axis(self):
        assert m._expand_ellipsis((..., 5), 2) == (slice(None), 5)

    def test_bool_mask_consumes_ndim_axes(self):
        # A 2-D mask consumes both axes, so the Ellipsis fills zero slices
        # (a[..., mask2d] == a[mask2d], not a[:, mask2d]).
        mask2d = np.ones((2, 3), dtype=bool)
        result = m._expand_ellipsis((..., mask2d), 2)
        assert len(result) == 1 and result[0] is mask2d

    def test_no_ellipsis_returns_key_unchanged(self):
        key = (slice(None), True)
        assert m._expand_ellipsis(key, 2) is key


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
