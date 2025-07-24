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

axes = [None, 0, 1, 2, (0, 1, 2)]


input = [[[12, 3, 1, 2], [9, 1, 6, 1]], [[7, 9, 11, 50], [31, 5, 3, 2]]]
in_num = num.array(input)
in_np = np.array(input)


@pytest.mark.parametrize("axis", axes)
def test_no_mask(axis):
    out_num, scl_num = num.average(in_num, axis=axis, returned=True)
    out_num_no_scl = num.average(in_num, axis=axis, returned=False)
    out_np, scl_np = np.average(in_np, axis=axis, returned=True)
    assert allclose(out_num, out_np)
    assert allclose(scl_num, scl_np)
    assert allclose(out_num, out_num_no_scl)


@pytest.mark.parametrize("axis", axes)
def test_full_weights(axis):
    weight_input = [[[1, 2, 3, 4], [3, 3, 7, 1]], [[2, 2, 3, 3], [4, 1, 0, 1]]]
    weights_np = np.array(weight_input)
    weights_num = num.array(weight_input)

    out_num, scl_num = num.average(
        in_num, weights=weights_num, axis=axis, returned=True
    )
    out_num_no_scl = num.average(
        in_num, weights=weights_num, axis=axis, returned=False
    )
    out_np, scl_np = np.average(
        in_np, weights=weights_np, axis=axis, returned=True
    )
    assert allclose(out_num, out_np)
    assert allclose(scl_num, scl_np)
    assert allclose(out_num, out_num_no_scl)


single_dimension_weights = [[3, 4], [1, 2], [4, 1, 2, 1]]
single_dimension_axis = [0, 1, 2]


@pytest.mark.parametrize(
    "weights,axis", zip(single_dimension_weights, single_dimension_axis)
)
def test_single_axis_weights(weights, axis):
    weights_np = np.array(weights)
    weights_num = num.array(weights)

    out_num, scl_num = num.average(
        in_num, weights=weights_num, axis=axis, returned=True
    )
    out_num_no_scl = num.average(
        in_num, weights=weights_num, axis=axis, returned=False
    )
    out_np, scl_np = np.average(
        in_np, weights=weights_np, axis=axis, returned=True
    )
    assert allclose(out_num, out_np)
    assert allclose(scl_num, scl_np)
    assert allclose(out_num, out_num_no_scl)


def test_exception_raising():
    with pytest.raises(ValueError):
        num.average(in_num, weights=[0, 2])
    with pytest.raises(ValueError):
        num.average(in_num, axis=2, weights=[0, 2])
    with pytest.raises(ValueError):
        num.average(in_num, axis=0, weights=[[0, 2]])
    with pytest.raises(ZeroDivisionError):
        num.average(in_num, axis=0, weights=[0, 0])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
