# Copyright 2025 NVIDIA Corporation
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

import cupynumeric as cn

# Basic functionality


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (3, 3),
        (3, 3, 3),
    ],
)
def test_ellipsis_multiply(shape):
    a = mk_seq_array(cn, shape)
    b = mk_seq_array(cn, shape)
    out = cn.empty(shape)

    cn0 = out[..., 0]
    a0 = a[..., 0]
    b0 = b[..., 0]

    cn.multiply(a0, b0, out=cn0)
    assert np.array_equal(out[..., 0], cn.multiply(a0, b0))


def test_inplace_op():
    a = mk_seq_array(cn, (3, 3))
    a[..., 0] += 10
    assert np.array_equal(a[:, 0], mk_seq_array(cn, (3, 3))[:, 0] + 10)


# Different indices


def test_middle():
    a = mk_seq_array(cn, (3, 4, 5))
    a[1, ..., 2] = 77
    assert np.array_equal(a[1, :, 2], cn.full((4,), 77))


def test_negative_index():
    a = mk_seq_array(cn, (3, 3))
    a[..., -1] = 5
    assert np.all(a[:, -1] == 5)


# Other


def test_mask_indexing():
    a = mk_seq_array(cn, (3, 3, 3))
    mask = np.array([True, False, True])
    a[..., mask] = 7
    assert np.all(a[:, :, mask] == 7)


@pytest.mark.parametrize(
    "assign_value, expected",
    [
        ([1, 2, 3], cn.array([[1, 2, 3], [1, 2, 3]])),
        (cn.array([[10], [20]]), cn.array([[10, 10, 10], [20, 20, 20]])),
        (5, cn.full((2, 3), 5)),
    ],
)
def test_broadcast(assign_value, expected):
    a = mk_seq_array(cn, (2, 3, 4))
    a[..., 2] = assign_value
    assert np.array_equal(a[:, :, 2], expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
