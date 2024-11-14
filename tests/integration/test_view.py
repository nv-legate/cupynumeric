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


def test_update_orig():
    orig = num.random.randn(2, 3, 4)
    update = num.random.randn(3, 15)
    view = orig[0]

    orig[0, :] += update[:, 11:]

    assert num.array_equal(orig[0, 0, :], view[0, :])


def test_update_slice():
    orig = num.random.randn(2, 3, 4)
    view = orig[0]

    view[1, :] = num.random.randn(4)

    assert num.array_equal(orig[0], view)


def test_mixed():
    xnp = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    ynp = xnp[2:, 2:]
    x = num.array(xnp)

    y = x[2:, 2:]

    assert np.array_equal(ynp, y)


@pytest.mark.parametrize("value", (2, 3))
def test_scalar(value):
    x = num.ones((1,))
    y = x.reshape((1, 1))

    y[:] = value

    assert np.array_equal(x, [value])
    assert np.array_equal(y, [[value]])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
