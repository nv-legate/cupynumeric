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
from utils.generators import mk_seq_array
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num


@pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 1, 2)])
def test_logical_reductions(axis):
    input = [[[12, 0, 1, 2], [9, 0, 0, 1]], [[0, 0, 0, 5], [1, 1, 1, 1]]]
    in_num = num.array(input)
    in_np = np.array(input)

    out_num = num.logical_and.reduce(in_num, axis=axis)
    out_np = np.logical_and.reduce(in_np, axis=axis)
    assert num.array_equal(out_num, out_np)

    out_num = num.logical_or.reduce(in_num, axis=axis)
    out_np = np.logical_or.reduce(in_np, axis=axis)
    assert num.array_equal(out_num, out_np)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE[:-1])
@pytest.mark.parametrize(
    "axis",
    [
        None,
        0,
        -1,
    ],
)
def test_logical_reductions_over_cupynumeric_arrays(ndim, axis):
    shape = (5,) * ndim
    np_arr = mk_seq_array(np, shape)
    in_np = tuple(np_arr % 2 for dim in range(ndim))
    in_num = tuple(num.array(np_arr % 2) for dim in range(ndim))
    out_num = num.logical_and.reduce(in_num, axis=axis)
    out_np = np.logical_and.reduce(in_np, axis=axis)
    assert num.array_equal(out_num, out_np)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
