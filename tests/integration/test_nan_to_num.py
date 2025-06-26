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

from math import prod

import numpy as np
import pytest
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


class TestNanToNum:
    @pytest.mark.parametrize(
        "value",
        [
            np.inf,
            -np.inf,
            np.nan,
            42,
            -17,
            0,
        ],
    )
    def test_scalar(self, value):
        """Test scalar inputs match numpy's implementation."""
        result = num.nan_to_num(value)
        expected = np.nan_to_num(value)
        assert np.allclose(result, expected)

    @pytest.fixture
    def sample_array(self):
        """Sample array with special values."""
        return np.array([np.inf, -np.inf, np.nan, -128, 128])

    def test_array(self, sample_array):
        """Test array inputs match numpy's implementation."""
        sample_array_num = num.array(sample_array)
        result = num.nan_to_num(sample_array_num)
        expected = np.nan_to_num(sample_array)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"nan": -9999, "posinf": 33333333, "neginf": 33333333},
            {"nan": -9999},
            {"posinf": 33333333},
            {"neginf": -33333333},
            {"nan": -9999, "posinf": 33333333},
            {"nan": -9999, "neginf": -33333333},
            {"posinf": 33333333, "neginf": -33333333},
        ],
    )
    def test_custom_values(self, sample_array, kwargs):
        """Test custom replacement values match numpy's implementation."""
        sample_array_num = num.array(sample_array)
        result = num.nan_to_num(sample_array_num, **kwargs)
        expected = np.nan_to_num(sample_array, **kwargs)
        assert np.allclose(result, expected)

    @pytest.fixture
    def complex_array(self):
        """Sample complex array with special values."""
        return np.array(
            [
                complex(np.inf, np.nan),  # inf + nan*j
                np.nan,  # nan + 0j
                complex(np.nan, np.inf),  # nan + inf*j
                2 + 3j,  # regular complex number
                5 - 1j,  # negative imaginary part
                0 + 4j,  # zero real part
                complex(np.nan, 1),  # nan + 1j
                complex(0, np.inf),  # 0 + inf*j
            ]
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"nan": 111111, "posinf": 222222},
        ],
    )
    def test_complex(self, complex_array, kwargs):
        """Test complex number handling matches numpy's implementation."""
        complex_array_num = num.array(complex_array)
        result = num.nan_to_num(complex_array_num, **kwargs)
        expected = np.nan_to_num(complex_array, **kwargs)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize(
        "x",
        [
            [np.inf, -np.inf, np.nan],
            [complex(np.inf, np.nan), np.nan, 2 + 3j],
            [np.nan, 1, 2, -np.inf],
        ],
    )
    def test_copy(self, copy, x):
        """Test copy parameter behavior matches numpy's implementation."""
        x_cpn = num.array(x)
        x_numpy = np.array(x)
        result = num.nan_to_num(x_cpn, copy=copy)
        expected = np.nan_to_num(x_numpy, copy=copy)

        # Check if the copy behavior matches numpy
        if copy:
            assert result is not x
            assert expected is not x

        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "x",
        [
            np.array([True, False]),
            np.array([1, 2, 3]),
            np.array([1.5, np.inf, -np.inf, np.nan, 3.5], dtype=np.float32),
            np.array([1.5, np.inf, -np.inf, np.nan, 3.5], dtype=np.float64),
            np.array(
                [
                    1 + 2j,
                    complex(np.inf, np.nan),
                    complex(-np.inf, np.inf),
                    complex(np.nan, -np.inf),
                    5 + 6j,
                ],
                dtype=np.complex64,
            ),
            np.array(
                [
                    1 + 2j,
                    complex(np.inf, np.nan),
                    complex(-np.inf, np.inf),
                    complex(np.nan, -np.inf),
                    5 + 6j,
                ],
                dtype=np.complex128,
            ),
        ],
    )
    def test_dtypes(self, x):
        """Test behavior with different dtypes."""
        x_num = num.array(x)
        result = num.nan_to_num(x_num)
        res_np = np.nan_to_num(x)
        assert np.array_equal(result, res_np)
        assert result.dtype == res_np.dtype


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
@pytest.mark.parametrize("varargs", [0.5, 1, 2, 0.3, 0])
def test_scalar_varargs(ndim, varargs):
    shape = (5,) * ndim
    size = prod(shape)
    arr_np = np.random.randint(-100, 100, size, dtype=int)
    in_np = np.sort(arr_np).reshape(shape).astype(float)

    # Add special values in random positions
    special_idx = np.random.choice(size, 3, replace=False)
    in_np.flat[special_idx] = [np.nan, np.inf, -np.inf]

    in_cn = num.array(in_np)
    res_np = np.nan_to_num(in_np, varargs)
    res_cn = num.nan_to_num(in_cn, varargs)
    assert np.allclose(res_np, res_cn, equal_nan=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
