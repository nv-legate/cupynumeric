# Copyright 2026 NVIDIA Corporation
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

from __future__ import annotations

import numpy as np
import pytest

import cupynumeric as num


class TestUfuncKindAndRepr:
    def test_get_kind_score_bool(self) -> None:
        x1_np = np.array([1.0, 2.0], dtype=np.float32)
        x2_np = np.array([True, False], dtype=np.bool_)
        x1 = num.array(x1_np)
        x2 = num.array(x2_np)

        out_np = np.ldexp(x1_np, x2_np)
        out = num.ldexp(x1, x2)
        assert np.array_equal(np.asarray(out), out_np)

    def test_get_kind_score_unknown_type(self) -> None:
        from cupynumeric._ufunc.ufunc import _get_kind_score

        assert _get_kind_score(object) == 3

    def test_ufunc_repr_is_string(self) -> None:
        from cupynumeric._ufunc.math import add as add_ufunc

        r = repr(add_ufunc)
        assert isinstance(r, str)


class TestUfuncNativeBridgeGuardrail:
    def test_public_ufuncs_do_not_use_native_array_bridge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_native_handle(self: num.ndarray) -> object:
            pytest.fail("public ufunc dispatch used the native array bridge")

        def fail_from_native_handle(handle: object) -> num.ndarray:
            pytest.fail("public ufunc dispatch wrapped a native array handle")

        monkeypatch.setattr(
            num.ndarray, "_native_array_handle", fail_native_handle
        )
        monkeypatch.setattr(
            num.ndarray,
            "_from_native_array_handle",
            staticmethod(fail_from_native_handle),
        )

        x_np = np.array([1, 2, 3], dtype=np.int32)
        y_np = np.array([4, 5, 6], dtype=np.int32)
        x = num.array(x_np)
        y = num.array(y_np)
        out = num.empty_like(x)

        neg = num.negative(x)
        add = num.add(x, y)
        add_out = num.add(x, y, out=out)
        reduced = num.add.reduce(x)

        assert np.array_equal(np.asarray(neg), np.negative(x_np))
        assert np.array_equal(np.asarray(add), np.add(x_np, y_np))
        assert add_out is out
        assert np.array_equal(np.asarray(out), np.add(x_np, y_np))
        assert np.array_equal(
            np.asarray(reduced), np.asarray(np.add.reduce(x_np))
        )


class TestUfuncPrepareOperandsErrors:
    def test_bad_out_type(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([4, 5, 6], dtype=np.int32)
        a = num.array(a_np)
        b = num.array(b_np)

        msg = r"return arrays must be of ArrayType"
        with pytest.raises(TypeError, match=msg):
            np.add(a_np, b_np, out=123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=msg):
            num.add(a, b, out=123)  # type: ignore[arg-type]

    def test_wrong_argcount(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        a = num.array(a_np)

        msg = r"takes from 2 to 3 positional arguments"
        with pytest.raises(TypeError, match=msg):
            np.add(a_np)  # type: ignore[call-arg]
        with pytest.raises(TypeError, match=msg):
            num.add(a)  # type: ignore[call-arg]

    def test_out_positional_and_kw(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([4, 5, 6], dtype=np.int32)
        out_np_1 = np.empty_like(a_np)
        out_np_2 = np.empty_like(a_np)

        a = num.array(a_np)
        b = num.array(b_np)
        out1 = num.empty_like(a)
        out2 = num.empty_like(a)

        msg = r"cannot specify 'out' as both a positional and keyword argument"
        with pytest.raises(TypeError, match=msg):
            np.add(a_np, b_np, out_np_1, out=out_np_2)
        with pytest.raises(TypeError, match=msg):
            num.add(a, b, out1, out=out2)

    def test_wrong_out_tuple_len(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([4, 5, 6], dtype=np.int32)
        out_np_1 = np.empty_like(a_np)
        out_np_2 = np.empty_like(a_np)

        a = num.array(a_np)
        b = num.array(b_np)
        out1 = num.empty_like(a)
        out2 = num.empty_like(a)

        msg = r"The 'out' tuple must have exactly one entry per ufunc output"
        with pytest.raises(ValueError, match=msg):
            np.add(a_np, b_np, out=(out_np_1, out_np_2))
        with pytest.raises(ValueError, match=msg):
            num.add(a, b, out=(out1, out2))

    def test_out_shape_mismatch(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([4, 5, 6], dtype=np.int32)
        out_np = np.empty((2,), dtype=np.int32)

        a = num.array(a_np)
        b = num.array(b_np)
        out = num.empty((2,), dtype=np.int32)

        msg = r"broadcast"
        with pytest.raises(ValueError, match=msg):
            np.add(a_np, b_np, out=out_np)
        with pytest.raises(ValueError, match=msg):
            num.add(a, b, out=out)

    def test_out_not_writeable(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([4, 5, 6], dtype=np.int32)
        a = num.array(a_np)
        b = num.array(b_np)
        out = num.empty_like(a)
        out.flags.writeable = False

        with pytest.raises(ValueError, match=r"array is not writeable"):
            num.add(a, b, out=out)

    def test_binary_out_variants(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([4, 5, 6], dtype=np.int32)
        a = num.array(a_np)
        b = num.array(b_np)
        expected = np.add(a_np, b_np)

        out_pos = num.empty_like(a)
        result_pos = num.add(a, b, out_pos)
        assert result_pos is out_pos
        assert np.array_equal(np.asarray(out_pos), expected)

        out_kw = num.empty_like(a)
        result_kw = num.add(a, b, out=out_kw)
        assert result_kw is out_kw
        assert np.array_equal(np.asarray(out_kw), expected)

        out_np = np.empty_like(a_np)
        result_np = num.add(a, b, out=out_np)
        assert np.array_equal(np.asarray(result_np), expected)
        assert np.array_equal(out_np, expected)

    def test_binary_broadcasted_output(self) -> None:
        a_np = np.array([[1, 2, 3]], dtype=np.int32)
        b_np = np.array([[10], [20]], dtype=np.int32)
        a = num.array(a_np)
        b = num.array(b_np)
        expected = np.add(a_np, b_np)

        out = num.empty(expected.shape, dtype=expected.dtype)
        result = num.add(a, b, out=out)
        assert result is out
        assert np.array_equal(np.asarray(out), expected)


class TestUfuncCastingBehavior:
    def test_binary_dtype_equal_fixed_precision(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.float32)
        b_np = np.array([4, 5, 6], dtype=np.float32)
        a = num.array(a_np)
        b = num.array(b_np)

        out_np = np.add(a_np, b_np, dtype=np.float32)
        out = num.add(a, b, dtype=np.float32)
        assert np.array_equal(np.asarray(out), out_np)

    def test_binary_same_dtype_arrays_preserve_dtype(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([4, 5, 6], dtype=np.int32)
        a = num.array(a_np)
        b = num.array(b_np)

        out_np = np.add(a_np, b_np)
        out = num.add(a, b)

        assert np.asarray(out).dtype == out_np.dtype
        assert np.array_equal(np.asarray(out), out_np)

    def test_binary_array_scalar_promotion(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        a = num.array(a_np)

        out_np = np.add(a_np, 2.0)
        out = num.add(a, 2.0)

        assert np.asarray(out).dtype == out_np.dtype
        assert np.array_equal(np.asarray(out), out_np)

    def test_binary_can_cast_input(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.int32)
        b_np = np.array([4, 5, 6], dtype=np.int32)
        a = num.array(a_np)
        b = num.array(b_np)

        out_np = np.add(a_np, b_np, dtype=np.float64, casting="safe")
        out = num.add(a, b, dtype=np.float64, casting="safe")
        assert np.array_equal(np.asarray(out), out_np)

    def test_binary_not_can_cast(self) -> None:
        a_np = np.array([1, 2, 3], dtype=np.float32)
        b_np = np.array([4, 5, 6], dtype=np.float32)
        a = num.array(a_np)
        b = num.array(b_np)

        with pytest.raises(TypeError):
            _ = np.add(a_np, b_np, dtype=np.int32, casting="same_kind")
        with pytest.raises(TypeError):
            _ = num.add(a, b, dtype=np.int32, casting="same_kind")

    def test_unary_call_full_dtype(self) -> None:
        from cupynumeric._ufunc.floating import floor as floor_ufunc

        x_np = np.array([1.25, 2.5], dtype=np.float32)
        x = num.array(x_np)

        out = floor_ufunc._call_full(x, dtype=np.float64)
        out_np = np.floor(x_np.astype(np.float64))
        assert np.array_equal(np.asarray(out), out_np)


class TestUfuncUnaryAndMultiout:
    def test_unary_resolve_dtype_no_match(self) -> None:
        # unary_ufunc.__call__ bypasses _resolve_dtype in cupynumeric, so call
        # _call_full to exercise the resolution path.
        from cupynumeric._ufunc.floating import floor as floor_ufunc

        x_np = np.array([1 + 2j], dtype=np.complex64)
        x = num.array(x_np)

        with pytest.raises(
            TypeError, match=r"ufunc 'floor' not supported for the input types"
        ):
            np.floor(x_np)
        with pytest.raises(
            TypeError, match=r"No matching signature of ufunc floor is found"
        ):
            floor_ufunc._call_full(x)

    def test_unary_out_positional_and_kw(self) -> None:
        x_np = np.array([1, 2, 3], dtype=np.int32)
        x = num.array(x_np)
        out1 = num.empty_like(x)
        out2 = num.empty_like(x)

        out_np_1 = np.empty_like(x_np)
        out_np_2 = np.empty_like(x_np)

        msg = r"cannot specify 'out' as both a positional and keyword argument"
        with pytest.raises(TypeError, match=msg):
            np.negative(x_np, out_np_1, out=out_np_2)
        with pytest.raises(TypeError, match=msg):
            num.negative(x, out1, out=out2)

    def test_unary_out_variants(self) -> None:
        x_np = np.array([1, 2, 3], dtype=np.int32)
        x = num.array(x_np)
        expected = np.negative(x_np)

        out_pos = num.empty_like(x)
        result_pos = num.negative(x, out_pos)
        assert result_pos is out_pos
        assert np.array_equal(np.asarray(out_pos), expected)

        out_kw = num.empty_like(x)
        result_kw = num.negative(x, out=out_kw)
        assert result_kw is out_kw
        assert np.array_equal(np.asarray(out_kw), expected)

        out_np = np.empty_like(x_np)
        result_np = num.negative(x, out=out_np)
        assert np.array_equal(np.asarray(result_np), expected)
        assert np.array_equal(out_np, expected)

    def test_multiout_resolve_dtype_no_match(self) -> None:
        x_np = np.array([1 + 2j], dtype=np.complex64)
        x = num.array(x_np)

        with pytest.raises(
            TypeError, match=r"ufunc 'frexp' not supported for the input types"
        ):
            np.frexp(x_np)
        with pytest.raises(
            TypeError, match=r"No matching signature of ufunc frexp is found"
        ):
            num.frexp(x)

    def test_multiout_dtype_semantics(self) -> None:
        x_np = np.array([1.25, 2.5], dtype=np.float32)
        x = num.array(x_np)

        mant_num, exp_num = num.frexp(x, dtype=np.float64)
        mant_np, exp_np = np.frexp(x_np.astype(np.float64))

        assert np.array_equal(np.asarray(exp_num), exp_np)
        assert np.allclose(np.asarray(mant_num), mant_np)

    def test_multiout_out_variants(self) -> None:
        x_np = np.array([1.25, 2.5], dtype=np.float32)
        x = num.array(x_np)
        expected_mant, expected_exp = np.frexp(x_np)

        mant_out = num.empty_like(x)
        exp_out = num.empty(x.shape, dtype=np.int32)
        mant, exp = num.frexp(x, out=(mant_out, exp_out))
        assert mant is mant_out
        assert exp is exp_out
        assert np.allclose(np.asarray(mant), expected_mant)
        assert np.array_equal(np.asarray(exp), expected_exp)

        mant_partial = num.empty_like(x)
        mant, exp = num.frexp(x, out=(mant_partial, None))
        assert mant is mant_partial
        assert np.allclose(np.asarray(mant), expected_mant)
        assert np.array_equal(np.asarray(exp), expected_exp)

        mant_pos = num.empty_like(x)
        exp_pos = num.empty(x.shape, dtype=np.int32)
        mant, exp = num.frexp(x, mant_pos, exp_pos)
        assert mant is mant_pos
        assert exp is exp_pos
        assert np.allclose(np.asarray(mant), expected_mant)
        assert np.array_equal(np.asarray(exp), expected_exp)

    def test_multiout_wrong_out_tuple_len(self) -> None:
        x_np = np.array([1.25, 2.5], dtype=np.float32)
        x = num.array(x_np)
        mant_out = num.empty_like(x)

        with pytest.raises(
            ValueError,
            match=r"The 'out' tuple must have exactly one entry per ufunc output",
        ):
            num.frexp(x, out=(mant_out,))


class TestUfuncBinaryResolution:
    def test_ldexp_scalar_float_exponent(self) -> None:
        x1_np = np.array([1.0, 2.0], dtype=np.float32)
        x1 = num.array(x1_np)

        with pytest.raises(TypeError):
            _ = np.ldexp(x1_np, 2.0)
        with pytest.raises(TypeError):
            _ = num.ldexp(x1, 2.0)

    def test_ldexp_float_exponent_array(self) -> None:
        x1_np = np.array([1.0, 2.0], dtype=np.float32)
        x2_np = np.array([3.0, 4.0], dtype=np.float32)
        x1 = num.array(x1_np)
        x2 = num.array(x2_np)

        with pytest.raises(TypeError):
            _ = np.ldexp(x1_np, x2_np)
        with pytest.raises(TypeError):
            _ = num.ldexp(x1, x2)

    def test_ldexp_int_exponent_array(self) -> None:
        x1_np = np.array([1.0, 2.0], dtype=np.float32)
        x2_np = np.array([3, 4], dtype=np.int64)
        x1 = num.array(x1_np)
        x2 = num.array(x2_np)

        out_np = np.ldexp(x1_np, x2_np)
        out = num.ldexp(x1, x2, casting="same_kind")
        assert np.array_equal(np.asarray(out), out_np)

    def test_ldexp_same_kind_rejects_float_exponent(self) -> None:
        x1_np = np.array([1.0, 2.0], dtype=np.float32)
        x2_np = np.array([3.0, 4.0], dtype=np.float32)
        x1 = num.array(x1_np)
        x2 = num.array(x2_np)

        with pytest.raises(TypeError):
            _ = np.ldexp(x1_np, x2_np, casting="same_kind")
        with pytest.raises(TypeError):
            _ = num.ldexp(x1, x2, casting="same_kind")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
