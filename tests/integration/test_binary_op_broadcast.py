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

N = 20

SHAPES = [
    (N, N + 1, N + 2),
    (1, N + 1, N + 2),
]

DIMS = list(range(2, 4))


@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("ndim", DIMS)
def test_random(shape, ndim):
    local_shape = shape[:ndim]
    x = num.random.random(local_shape)
    a = x.__array__()

    for dim in range(1, ndim):
        y = num.random.random(local_shape[-dim:])
        b = y.__array__()
        print(f"  {a.shape} x {b.shape}")
        assert num.array_equal(x + y, a + b)

    for dim in range(ndim):
        rhs_shape = list(local_shape)
        rhs_shape[dim] = 1
        if (np.array(rhs_shape) == 1).all():
            y = np.random.random(tuple(rhs_shape))
            b = num.array(y)
        else:
            y = num.random.random(tuple(rhs_shape))
            b = y.__array__()
        print(f"  {a.shape} x {b.shape}")
        assert num.array_equal(x + y, a + b)

    if ndim > 2:
        for dim in range(ndim):
            rhs_shape = [1] * ndim
            rhs_shape[dim] = local_shape[dim]
            if (np.array(rhs_shape) == 1).all():
                y = np.random.random(tuple(rhs_shape))
                b = num.array(y)
            else:
                y = num.random.random(tuple(rhs_shape))
                b = y.__array__()
            print(f"  {a.shape} x {b.shape}")
            assert num.array_equal(x + y, a + b)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
