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


def test_basic():
    a = num.array([[1, 2, 3], [4, 5, 6]])
    an = np.array([[1, 2, 3], [4, 5, 6]])
    print(a.shape)
    print(an.shape)
    assert a.shape == an.shape
    assert a.flat[2] == 3
    a.flat[2] = 4
    assert a.flat[2] != 3

    r = a.sum(0)
    rn = an.sum(0)
    assert r.shape == rn.shape

    y = num.random.random((5, 6, 7))
    yn = np.random.random((5, 6, 7))
    assert y.shape == yn.shape

    zn = yn[:, 3:5]
    z = y[:, 3:5]
    assert z.shape == zn.shape

    print(type(y.shape[1]))
    d = y.shape[1] / 3
    assert d == 2.0


def test_reshape():
    x = num.random.random((2, 3, 4))
    y = x.reshape((4, 3, 2))
    assert y.shape == (4, 3, 2)
    assert y.size == x.size
    pos = 0
    for a in range(0, y.size):
        print(pos, y.flat[pos], x.flat[pos])
        assert y.flat[pos] == x.flat[pos]
        pos = pos + 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
