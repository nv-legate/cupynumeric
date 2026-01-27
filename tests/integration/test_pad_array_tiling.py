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

import cupynumeric as num


class TestAsPairsPadWidth:
    def test_len1(self) -> None:
        a_np = np.arange(4).reshape(2, 2)
        a = num.array(a_np)
        out_num = num.pad(a, pad_width=[1], mode="edge")
        out_np = np.pad(a_np, pad_width=((1, 1), (1, 1)), mode="edge")
        assert np.array_equal(out_num, out_np)
        assert out_num.shape == (4, 4)

    def test_row_pair(self) -> None:
        a_np = np.arange(4).reshape(2, 2)
        a = num.array(a_np)
        out_num = num.pad(a, pad_width=[[1, 2]], mode="edge")
        out_np = np.pad(a_np, pad_width=((1, 2), (1, 2)), mode="edge")
        assert np.array_equal(out_num, out_np)
        assert out_num.shape == (5, 5)

    def test_bad_len(self) -> None:
        a_np = np.arange(4).reshape(2, 2)
        a = num.array(a_np)
        match = r"sequence argument must be of length 1, 2, or ndim"
        with pytest.raises(ValueError, match=match):
            num.pad(a, pad_width=[1, 2, 3], mode="edge")
        with pytest.raises(ValueError):
            np.pad(a_np, pad_width=[1, 2, 3], mode="edge")

    def test_bad_shape(self) -> None:
        a_np = np.arange(4).reshape(2, 2)
        a = num.array(a_np)
        match = r"sequence argument must have shape"
        with pytest.raises(ValueError, match=match):
            num.pad(a, pad_width=[[1, 2, 3]], mode="edge")
        with pytest.raises(ValueError):
            np.pad(a_np, pad_width=[[1, 2, 3]], mode="edge")

    def test_bad_ndim(self) -> None:
        a_np = np.arange(4).reshape(2, 2)
        a = num.array(a_np)
        match = r"sequence argument must be 1- or 2-dimensional"
        pad_width = np.array([[[1, 1]]], dtype=np.int64)
        with pytest.raises(ValueError, match=match):
            num.pad(a, pad_width=pad_width, mode="edge")
        with pytest.raises(ValueError):
            np.pad(a_np, pad_width=pad_width, mode="edge")


class TestPadWidthDType:
    def test_non_integral_dtype(self) -> None:
        a_np = np.arange(3)
        a = num.array(a_np)
        pad_width = np.array([1.5], dtype=np.float64)
        with pytest.raises(
            TypeError, match=r"`pad_width` must be of integral type"
        ):
            num.pad(a, pad_width=pad_width, mode="edge")
        with pytest.raises(TypeError):
            np.pad(a_np, pad_width=pad_width, mode="edge")


class TestConstantValuesShape:
    def test_len1_scalar_like(self) -> None:
        a_np = np.zeros((2, 2, 2), dtype=np.int64)
        a = num.array(a_np)
        out_num = num.pad(a, pad_width=1, mode="constant", constant_values=[7])
        out_np = np.pad(
            a_np, pad_width=1, mode="constant", constant_values=[7]
        )
        assert np.array_equal(out_num, out_np)

    def test_1x1_scalar_like(self) -> None:
        a_np = np.zeros((2, 2), dtype=np.int64)
        a = num.array(a_np)
        out_num = num.pad(
            a, pad_width=1, mode="constant", constant_values=[[7]]
        )
        out_np = np.pad(
            a_np, pad_width=1, mode="constant", constant_values=[[7]]
        )
        assert np.array_equal(out_num, out_np)

    def test_1x2_pair(self) -> None:
        a_np = np.zeros((2, 2), dtype=np.int64)
        a = num.array(a_np)
        out_num = num.pad(
            a, pad_width=1, mode="constant", constant_values=[[1, 2]]
        )
        out_np = np.pad(
            a_np, pad_width=1, mode="constant", constant_values=[[1, 2]]
        )
        assert np.array_equal(out_num, out_np)

    def test_len_matches_ndim_per_axis(self) -> None:
        a_np = np.zeros((2, 2, 2), dtype=np.int64)
        a = num.array(a_np)
        out_num = num.pad(
            a, pad_width=1, mode="constant", constant_values=[1, 2, 3]
        )

        # NumPy does not accept the per-axis shorthand `constant_values=[1,2,3]`
        # for ndim>2; it raises while trying to broadcast to (ndim, 2).
        with pytest.raises(ValueError):
            np.pad(
                a_np, pad_width=1, mode="constant", constant_values=[1, 2, 3]
            )

        # Compare against an equivalent NumPy representation: shape (ndim, 1).
        out_np = np.pad(
            a_np,
            pad_width=1,
            mode="constant",
            constant_values=np.array([[1], [2], [3]], dtype=np.int64),
        )
        assert np.array_equal(out_num, out_np)

    def test_bad_len(self) -> None:
        a_np = np.zeros((2, 2, 2), dtype=np.int64)
        a = num.array(a_np)
        match = r"sequence argument must have length"
        with pytest.raises(ValueError, match=match):
            num.pad(
                a, pad_width=1, mode="constant", constant_values=[1, 2, 3, 4]
            )
        with pytest.raises(ValueError):
            np.pad(
                a_np,
                pad_width=1,
                mode="constant",
                constant_values=[1, 2, 3, 4],
            )

    def test_bad_shape(self) -> None:
        a_np = np.zeros((2, 2, 2), dtype=np.int64)
        a = num.array(a_np)
        bad = np.arange(6, dtype=np.int64).reshape(2, 3)
        match = r"constant_values argument must have shape"
        with pytest.raises(ValueError, match=match):
            num.pad(a, pad_width=1, mode="constant", constant_values=bad)
        with pytest.raises(ValueError):
            np.pad(a_np, pad_width=1, mode="constant", constant_values=bad)

    def test_bad_ndim(self) -> None:
        a_np = np.zeros((2, 2, 2), dtype=np.int64)
        a = num.array(a_np)
        bad = np.zeros((1, 1, 1))
        match = r"constant_values must be 0, 1, or 2-dimensional"
        with pytest.raises(ValueError, match=match):
            num.pad(a, pad_width=1, mode="constant", constant_values=bad)
        with pytest.raises(ValueError):
            np.pad(a_np, pad_width=1, mode="constant", constant_values=bad)


class TestPadPythonModes:
    def test_callable_mode(self) -> None:
        a_np = np.arange(6).reshape(2, 3)
        a = num.array(a_np)

        def noop_pad_func(
            vec: np.ndarray, pad_width: tuple[int, int], axis: int, kwargs
        ) -> None:
            vec[:] = vec

        out_num = num.pad(a, pad_width=1, mode=noop_pad_func)
        out_np = np.pad(a_np, pad_width=1, mode=noop_pad_func)
        assert np.array_equal(out_num, out_np)

    def test_empty_wrap_raises(self) -> None:
        a_np = np.empty((0, 2), dtype=np.float64)
        a = num.array(a_np)
        match = r"can't extend empty axis"
        with pytest.raises(ValueError, match=match):
            num.pad(a, pad_width=1, mode="wrap")
        with pytest.raises(ValueError, match=match):
            np.pad(a_np, pad_width=1, mode="wrap")

    def test_empty_zero_pad(self) -> None:
        a_np = np.empty((0, 2), dtype=np.float64)
        a = num.array(a_np)
        out_num = num.pad(a, pad_width=0, mode="wrap")
        out_np = np.pad(a_np, pad_width=0, mode="wrap")
        assert np.array_equal(out_num, out_np)

    def test_stat_length_zero(self) -> None:
        a_np = np.arange(3)
        a = num.array(a_np)
        out_num = num.pad(a, pad_width=1, mode="mean", stat_length=0)
        out_num_np = np.asarray(out_num)
        out_np = np.pad(a_np, pad_width=1, mode="mean", stat_length=0)
        assert out_num_np.shape == out_np.shape
        assert np.array_equal(out_num_np[1:-1], out_np[1:-1])
        assert out_num_np[0] == 0 and out_num_np[-1] == 0
        assert out_np[0] in (0, np.iinfo(out_np.dtype).min)
        assert out_np[-1] in (0, np.iinfo(out_np.dtype).min)

    def test_linear_ramp_2d(self) -> None:
        a_np = np.arange(4).reshape(2, 2)
        a = num.array(a_np)
        out_num = num.pad(a, pad_width=1, mode="linear_ramp", end_values=0)
        out_np = np.pad(a_np, pad_width=1, mode="linear_ramp", end_values=0)
        assert np.allclose(out_num, out_np)

    def test_bad_reflect_type(self) -> None:
        a_np = np.arange(3)
        a = num.array(a_np)
        out_num = num.pad(a, pad_width=1, mode="reflect", reflect_type="bogus")
        out_np = np.pad(
            a_np, pad_width=1, mode="reflect", reflect_type="bogus"
        )
        assert np.array_equal(out_num, out_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
