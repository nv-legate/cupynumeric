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

from functools import reduce

import pytest
from utils.generators import mk_seq_array

import cupynumeric as num

DIM = 128
NO_EMPTY_SIZES = [
    (DIM,),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]


@pytest.mark.parametrize("size", NO_EMPTY_SIZES)
def test_int(size):
    arr = mk_seq_array(num, shape=size)
    max_data = reduce(lambda x, y: x * y, size)
    assert -1 not in arr
    assert 0 not in arr
    assert 1 in arr
    assert max_data // 2 in arr
    assert max_data in arr
    assert max_data + 1 not in arr


@pytest.mark.parametrize("size", NO_EMPTY_SIZES)
def test_complex(size):
    arr = mk_seq_array(num, shape=size) + mk_seq_array(num, shape=size) * 1.0j
    max_data = reduce(lambda x, y: x * y, size)
    assert -1 not in arr
    assert 0 not in arr
    assert 1 + 1.0j in arr
    assert (max_data // 2) + (max_data // 2) * 1.0j in arr
    assert max_data + max_data * 1.0j in arr
    assert (max_data + 1) + (max_data + 1) * 1.0j not in arr


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
