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


def test_negative():
    a = num.array([0, 1, 2])
    with pytest.raises(ValueError):
        num.tile(a, -4)


def test_float():
    a = num.array([0, 1, 2])
    msg = r"float"
    with pytest.raises(TypeError, match=msg):
        num.tile(a, 2.2)


def test_list():
    a = num.array([0, 1, 2])
    msg = r"1d sequence"
    with pytest.raises(TypeError, match=msg):
        num.tile(a, [[1, 2], [3, 4]])


def test_tuple():
    a = num.array([0, 1, 2])
    msg = r"1d sequence"
    with pytest.raises(TypeError, match=msg):
        num.tile(a, ((1, 2), (3, 4)))


DIM = 5
SIZES = [
    (0,),
    (1),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 1, 1),
    (DIM, DIM, DIM),
]


@pytest.mark.parametrize("size", SIZES, ids=str)
@pytest.mark.parametrize("value", (0, DIM, (DIM, DIM), (DIM, DIM, DIM)))
def test_basic(size, value):
    a = np.random.randint(low=-10.0, high=10, size=size)
    res_np = np.tile(a, value)
    res_num = num.tile(a, value)
    assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
