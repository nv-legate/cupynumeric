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
    a = num.array([[1, 2, 3, 4, 5, 6], [4, 5, 6, 7, 8, 9]], dtype=np.float64)
    b = num.array(
        [[10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]],
        dtype=np.float64,
    )
    c = a.dot(b)
    assert num.array_equal(c, [[350, 371], [620, 659]])

    d = num.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    e = num.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    f = d.dot(e)
    assert f == 91

    # This test does not work ATM. It seems that setting random seed to
    # be the same is not sufficient to make the inputs the same.

    # num.random.seed(42)
    # a = num.random.randn(1, 3, 15)
    # b = num.random.randn(15, 16)
    # c = a[0].dot(b)

    # np.random.seed(42)
    # an = np.random.randn(1, 3, 15)
    # bn = np.random.randn(15, 16)
    # cn = an[0].dot(bn)

    # assert num.allclose(c, cn)

    return


if __name__ == "__main__":
    test()
