#!/usr/bin/env python
# Copyright 2025 NVIDIA Corporation
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

import pytest

import cupynumeric as np


class TestRoundMethod:
    def test_round_scalar(self):
        arr = np.array(3.14159)
        assert round(arr) == 3
        assert round(arr, 2) == 3.14
        assert round(arr, 0) == 3.0
        assert round(arr, 3) == 3.142

    def test_round_integer_scalar(self):
        arr = np.array(5)
        assert round(arr) == 5
        assert round(arr, 2) == 5

    def test_round_negative_number(self):
        arr = np.array(-2.71828)
        assert round(arr) == -3
        assert round(arr, 3) == -2.718

    def test_round_non_numeric_raises(self):
        arr = np.array(0.0 + 0.0j)
        msg = "Rounding not supported for type"
        with pytest.raises(TypeError, match=msg):
            round(arr)

    def test_round_empty_array_raises(self):
        arr = np.array([])
        msg = "Python's round method can be called only on scalars"
        with pytest.raises(ValueError, match=msg):
            round(arr)

    def test_round_non_scalar_raises(self):
        arr = np.array([1.1, 2.2, 3.3])
        msg = "Python's round method can be called only on scalars"
        with pytest.raises(ValueError, match=msg):
            round(arr)

        arr_2d = np.array([[1.1, 2.2], [3.3, 4.4]])
        msg = "Python's round method can be called only on scalars"
        with pytest.raises(ValueError, match=msg):
            round(arr_2d)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
