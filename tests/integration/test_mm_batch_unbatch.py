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
from utils.comparisons import allclose

import cupynumeric as num


@pytest.mark.parametrize("a_shape", ((4, 5), (100, 5)))
@pytest.mark.parametrize("b_shape", ((5, 6),))
def test_batched(a_shape, b_shape):
    np_a = np.random.random(a_shape)
    np_b = np.random.random(b_shape)
    num_a = num.array(np_a)
    num_b = num.array(np_b)

    num_res = num.matmul(num_a, num_b)
    np_res = np.matmul(np_a, np_b)

    assert allclose(np_res, num_res)


@pytest.mark.parametrize("a_shape", ((4, 5), (100, 5)))
@pytest.mark.parametrize("b_shape", ((5, 6),))
def test_unbatched(a_shape, b_shape):
    np_a = np.random.random(a_shape)
    np_b = np.random.random(b_shape)
    num_a = num.array(np_a)
    num_b = num.array(np_b)

    res_shape = (a_shape[0], b_shape[1])
    np_res = np.zeros(res_shape, dtype=np_a.dtype)

    num_res = num.array(np_res)

    num_res._thunk.ts_matmul(num_a._thunk, num_b._thunk)
    np_res = np.matmul(np_a, np_b)

    # print("A @ B = %s"%(str(num_res)))
    assert allclose(np_res, num_res)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
