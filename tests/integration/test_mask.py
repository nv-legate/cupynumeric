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


def test_lhs():
    x = num.array([1, 2, -3])
    mask = x > 0
    w = mask * x
    assert np.array_equal(w, [1, 2, 0])


def test_rhs():
    x = num.array([1, 2, -3])
    mask = x > 0
    w = x * mask
    assert np.array_equal(w, [1, 2, 0])


def test_inverted_rhs():
    x = num.array([1, 2, -3])
    mask = x > 0
    w = x * ~mask
    assert np.array_equal(w, np.array([0, 0, -3]))


def test_inverted_lhs():
    x = num.array([1, 2, -3])
    mask = x > 0
    w = ~mask * x
    assert np.array_equal(w, [0, 0, -3])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
