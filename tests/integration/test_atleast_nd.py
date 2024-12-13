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
from utils.utils import MAX_DIM_RANGE, check_module_function

import cupynumeric as num

DIM = 10

SIZE_CASES = list((DIM,) * ndim for ndim in MAX_DIM_RANGE)

SIZE_CASES += [
    (0,),  # empty array
    (1,),  # singlton array
]


# test to run atleast_nd w/ a single array
@pytest.mark.parametrize("size", SIZE_CASES, ids=str)
def test_atleast_1d(size):
    a = [np.arange(np.prod(size)).reshape(size)]
    print_msg = f"np & cupynumeric.atleast_1d(size={size})"
    check_module_function("atleast_1d", a, {}, print_msg)


def test_atleast_1d_scalar():
    a = 1.0
    assert np.array_equal(np.atleast_1d(a), num.atleast_1d(a))


@pytest.mark.parametrize("size", SIZE_CASES, ids=str)
def test_atleast_2d(size):
    a = [np.arange(np.prod(size)).reshape(size)]
    print_msg = f"np & cupynumeric.atleast_2d(size={size})"
    check_module_function("atleast_2d", a, {}, print_msg)


def test_atleast_2d_scalar():
    a = 1.0
    assert np.array_equal(np.atleast_2d(a), num.atleast_2d(a))


@pytest.mark.parametrize("size", SIZE_CASES, ids=str)
def test_atleast_3d(size):
    a = [np.arange(np.prod(size)).reshape(size)]
    print_msg = f"np & cupynumeric.atleast_3d(size={size})"
    check_module_function("atleast_3d", a, {}, print_msg)


def test_atleast_3d_scalar():
    a = 1.0
    assert np.array_equal(np.atleast_2d(a), num.atleast_2d(a))


# test to run atleast_nd w/ list of arrays
@pytest.mark.parametrize("dim", range(1, 4))
def test_atleast_nd(dim):
    a = list(np.arange(np.prod(size)).reshape(size) for size in SIZE_CASES)
    scalar = 10.0
    a.append(scalar)
    print_msg = f"np & cupynumeric.atleast_{dim}d(size={SIZE_CASES})"
    check_module_function(f"atleast_{dim}d", a, {}, print_msg)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
