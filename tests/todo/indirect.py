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

import cupynumeric as num


def test():
    x = num.array([[1, 2], [3, 4], [5, 6]])
    assert num.array_equal(x[[0, 1, 2], [0, 1, 0]], [1, 4, 5])

    x = num.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    rows = num.array([0, 3])
    columns = num.array([0, 2])
    assert num.array_equal(x[rows[:, np.newaxis], columns], [[0, 2], [9, 11]])

    zg = num.array([[-1.2 + 0.5j, 1.2 - 2j], [-2.2 + 3.5j, 4.2 - 6.2j]])
    m = num.array([[True, False], [False, True]])
    assert num.array_equal(zg[m], [-1.2 + 0.5j, 4.2 - 6.2j])

    anp = np.array([[[2, 1], [3, 2]], [[2, 4], [4, 1]]])
    a = num.array(anp)
    nznp = anp < 3
    nzgp = a < 3
    assert num.array_equal(anp[nznp], a[nzgp])

    y = num.array(
        [[[True, True], [False, True]], [[True, False], [False, True]]]
    )
    z = num.nonzero(y)
    assert num.array_equal(a[z], num.array([2, 1, 2, 2, 1]))

    np.random.seed(42)
    anp = np.random.randn(10, 10, 4)
    a = num.array(anp)
    bnp = np.array([3, 4, 6])
    cnp = np.array([1, 4, 5])
    b = num.array(bnp)
    c = num.array(cnp)

    assert num.array_equal(a[b], anp[bnp])
    assert num.array_equal(a[(b, c)], anp[(b, c)])

    onesnp = np.zeros(10, int)
    ones = num.zeros(10, int)

    dnp = np.random.randn(20, 4)
    d = num.array(dnp)
    assert num.array_equal(dnp[np.where(onesnp)], d[num.where(ones)])


if __name__ == "__main__":
    test()
