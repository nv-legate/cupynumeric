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
    x = num.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x[0:5] = num.array([11, 12, 13, 14, 15])
    x[5:10] = num.array([16, 17, 18, 19, 20])
    x[4:8] = num.array([21, 22, 23, 24])
    assert np.array_equal(x[5:10], [22, 23, 24, 19, 20])
    assert np.array_equal(x, [11, 12, 13, 14, 21, 22, 23, 24, 19, 20])

    anp = np.zeros((5, 6))
    bnp = np.random.random((5, 4))
    cnp = np.random.random((5, 2))
    a = num.zeros((5, 6))
    b = num.array(bnp)
    c = num.array(cnp)
    a[:, :4] = b
    a[:, 0] = 1
    a[:, 3:5] = c
    anp[:, :4] = bnp
    anp[:, 0] = 1
    anp[:, 3:5] = cnp
    assert np.array_equal(a, anp)

    dnp = np.random.random((2, 3, 4))
    enp = np.random.random((2, 3, 4))
    fnp = np.random.random((3, 2))
    d = num.array(dnp)
    e = num.array(enp)
    f = num.array(fnp)
    d[1, :, 0] = 1
    d[1, :, 1:3] = f
    d[0] = e[1]
    dnp[1, :, 0] = 1
    dnp[1, :, 1:3] = fnp
    dnp[0] = enp[1]
    assert np.array_equal(d, dnp)

    anp = num.random.random((3, 3))
    a = anp.__array__()

    anp[:, 0] = anp[:, 1]
    a[:, 0] = a[:, 1]
    assert np.array_equal(anp, a)

    anp = num.random.random((3, 3))
    a = anp.__array__()

    anp[:, 0:2] = anp[:, 1:3]
    a[:, 0:2] = a[:, 1:3]
    assert np.array_equal(anp, a)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
