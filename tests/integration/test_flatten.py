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
from utils.utils import check_array_method

import cupynumeric as num

DIM = 10

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


# test ndarray.flatten w/ 1D, 2D and 3D arrays
@pytest.mark.parametrize("order", ("C", "F", "A"))
@pytest.mark.parametrize("size", SIZES, ids=str)
def test_basic(order, size):
    a = np.random.randint(low=0, high=100, size=size)
    print_msg = f"np & cupynumeric.ndarray.flatten({order})"
    check_array_method(a, "flatten", [order], {}, print_msg)


class TestNdarrayFlattenErrors:
    def setup_method(self):
        size_a = (1, DIM)
        anp = np.random.randint(low=0, high=100, size=size_a)
        self.anum = num.array(anp)

    def test_non_string_order(self):
        order = 0
        msg = "order must be str, not int"
        with pytest.raises(TypeError, match=msg):
            self.anum.flatten(order)

        order = 1
        msg = "order must be str, not int"
        with pytest.raises(TypeError, match=msg):
            self.anum.flatten(order)

        order = -1
        msg = "order must be str, not int"
        with pytest.raises(TypeError, match=msg):
            self.anum.flatten(order)

        order = 1.0
        msg = "order must be str, not float"
        with pytest.raises(TypeError, match=msg):
            self.anum.flatten(order)

        order = ["C"]
        msg = "order must be str, not list"
        with pytest.raises(TypeError, match=msg):
            self.anum.flatten(order)

    def test_bad_string_order(self):
        order = "Z"
        msg = "order must be one of 'C', 'F', 'A', or 'K'"
        with pytest.raises(ValueError, match=msg):
            self.anum.flatten(order)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
