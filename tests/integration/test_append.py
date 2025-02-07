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
from utils.utils import check_module_function

import cupynumeric as num

DIM = 10

# test append w/ 1D, 2D and 3D arrays
SIZES = [
    (0,),
    (0, 10),
    (1,),
    (1, 1),
    (1, 1, 1),
    (1, DIM),
    (1, DIM, 1),
    (DIM, DIM),
    (DIM, DIM, DIM),
]


@pytest.mark.parametrize("size", SIZES, ids=str)
def test_append(size):
    a = np.random.randint(low=0, high=100, size=size)
    test_args = [-1] + list(range(a.ndim))

    for axis in test_args:
        size_b = list(size)
        size_b[axis] = size[axis] + 10
        b = np.random.randint(low=0, high=100, size=size_b)
        print_msg = f"np.append(array({a.shape}), array({b.shape}), {axis})"
        check_module_function("append", [a, b], {"axis": axis}, print_msg)


@pytest.mark.parametrize("size_b", SIZES, ids=str)
@pytest.mark.parametrize("size_a", SIZES, ids=str)
def test_append_axis_none(size_a, size_b):
    axis = None
    a = np.random.randint(low=0, high=100, size=size_a)
    b = np.random.randint(low=0, high=100, size=size_b)
    print_msg = f"np.append(array({a.shape}), array({b.shape}), {axis})"
    check_module_function("append", [a, b], {"axis": axis}, print_msg)


class TestAppendErrors:
    def setup_method(self):
        size_a = (1, DIM)
        self.a = np.random.randint(low=0, high=100, size=size_a)

    def test_bad_dimension(self):
        size_b = (1, DIM, 1)
        b = np.random.randint(low=0, high=100, size=size_b)

        msg = (
            "All arguments to concatenate must have the "
            "same number of dimensions"
        )
        with pytest.raises(ValueError, match=msg):
            num.append(self.a, b, axis=1)

    def test_bad_index(self):
        with pytest.raises(IndexError):
            num.append(self.a, self.a, axis=5)

    def test_bad_shape(self):
        size_c = (10, DIM)
        c = np.random.randint(low=0, high=100, size=size_c)
        with pytest.raises(ValueError):
            num.append(self.a, c, axis=1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
