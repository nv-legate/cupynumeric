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
import numpy as np
import pytest
from utils.comparisons import allclose

from cupynumeric._module.creation_shape import zeros


def setup_arrays():
    a = zeros((3, 3), dtype=int)
    a[:2] = [[100000, 200000, 300000], [400000, 500000, 600000]]
    a[2:3] = [99999, 99999, 99999]
    a_np = np.zeros((3, 3), dtype=int)
    a_np[:2] = [[100000, 200000, 300000], [400000, 500000, 600000]]
    a_np[2:3] = [99999, 99999, 99999]
    return a, a_np


def test_unchanged():
    a, a_np = setup_arrays()
    before = a.copy()
    before_np = a_np.copy()
    a[3:] = a[2:]
    a_np[3:] = a_np[2:]
    assert allclose(a, a_np)
    assert allclose(a, before)
    assert allclose(a_np, before_np)


def test_sliceobj():
    a, a_np = setup_arrays()
    a[slice(3, None)] = a[slice(2, None)]
    a_np[slice(3, None)] = a_np[slice(2, None)]
    assert allclose(a, a_np)


def test_assign_empty_slice_to_itself():
    a, a_np = setup_arrays()
    a[3:] = a[3:]
    a_np[3:] = a_np[3:]
    assert allclose(a, a_np)


def test_assign_to_regular_slice():
    a = zeros((3, 3), dtype=int)
    a_np = np.zeros((3, 3), dtype=int)
    a[1:3] = [[700000, 800000, 900000], [1000000, 1100000, 1200000]]
    a_np[1:3] = [[700000, 800000, 900000], [1000000, 1100000, 1200000]]
    assert allclose(a, a_np)


def test_assign_regular_array_to_empty_slice():
    a, a_np = setup_arrays()
    a[3:] = [[100000, 200000, 300000]]
    a_np[3:] = [[100000, 200000, 300000]]
    assert allclose(a, a_np)


def test_assign_out_of_bounds_high_index():
    a, a_np = setup_arrays()
    a[100:] = a[2:]
    a_np[100:] = a_np[2:]
    assert allclose(a, a_np)


def test_assign_negative_zero_start():
    a, a_np = setup_arrays()
    a[-0:] = a[2:]
    a_np[-0:] = a_np[2:]
    assert allclose(a, a_np)


def test_assign_out_of_bounds_low_index():
    a, a_np = setup_arrays()
    a[:-100] = a[2:]
    a_np[:-100] = a_np[2:]
    assert allclose(a, a_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
