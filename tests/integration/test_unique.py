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
from legate.core import LEGATE_MAX_DIM

import cupynumeric as num


def test_with_nonzero():
    (a,) = num.nonzero(num.array([1, 1, 0, 0]))
    a_np = a.__array__()

    b = num.unique(a)
    b_np = np.unique(a_np)

    assert np.array_equal(b, b_np)


@pytest.mark.parametrize(
    "ndim",
    (
        1,
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
def test_ndim(ndim):
    shape = (4,) * ndim
    a = num.random.randint(0, 3, size=shape)
    a_np = np.array(a)

    b = np.unique(a)
    b_np = num.unique(a_np)

    assert np.array_equal(b, b_np)


@pytest.mark.xfail
@pytest.mark.parametrize("return_index", (True, False))
@pytest.mark.parametrize("return_inverse", (True, False))
@pytest.mark.parametrize("return_counts", (True, False))
@pytest.mark.parametrize("axis", (0, 1))
def test_parameters(return_index, return_inverse, return_counts, axis):
    arr_num = num.random.randint(0, 3, size=(3, 3))
    arr_np = np.array(arr_num)
    res_num = num.unique(
        arr_num,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    # cuPyNumeric raises NotImplementedError: Keyword arguments
    # for `unique` are not yet supported
    res_np = np.unique(
        arr_np,
        return_index=return_index,
        return_inverse=return_inverse,
        return_counts=return_counts,
        axis=axis,
    )
    assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
