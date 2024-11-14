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
from numpy.lib import NumpyVersion
from utils.comparisons import allclose
from utils.generators import mk_seq_array

import cupynumeric as num


def setitem(lib, a, slice_lhs, slice_rhs):
    a[slice_lhs] = a[slice_rhs]


def dot(lib, a, slice_lhs, slice_rhs):
    modes = "".join([chr(ord("a") + m) for m in range(len(a.shape))])
    expr = f"{modes},{modes}->{modes}"
    lib.einsum(expr, a[slice_lhs], a[slice_rhs], out=a[slice_lhs])


def unary_arith(lib, a, slice_lhs, slice_rhs):
    lib.sin(a[slice_rhs], out=a[slice_lhs])


def binary_arith(lib, a, slice_lhs, slice_rhs):
    a[slice_lhs] += a[slice_rhs]


def put(lib, a, slice_lhs, slice_rhs):
    indices = lib.flip(lib.arange(a[slice_rhs].size))
    a[slice_lhs].put(indices, a[slice_rhs])


def putmask(lib, a, slice_lhs, slice_rhs):
    mask = (mk_seq_array(lib, a[slice_rhs].shape) % 2).astype(bool)
    lib.putmask(a[slice_lhs], mask, a[slice_rhs])


SHAPES = ((4,), (4, 5), (4, 5, 6))
OPERATIONS = (setitem, dot, unary_arith, binary_arith, put, putmask)


@pytest.mark.parametrize("partial", (True, False))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_partial(partial, shape, operation):
    if partial:
        # e.g. for shape = (4,5) and setitem: lhs[1:,:] = rhs[:-1,:]
        slice_lhs = (slice(1, None),) + (slice(None),) * (len(shape) - 1)
        slice_rhs = (slice(None, -1),) + (slice(None),) * (len(shape) - 1)
    else:
        # e.g. for shape = (4,5) and setitem: lhs[:,:] = rhs[:,:]
        slice_lhs = (slice(None),) * len(shape)
        slice_rhs = (slice(None),) * len(shape)

    a_np = mk_seq_array(np, shape).astype(np.float64)
    a_num = mk_seq_array(num, shape).astype(np.float64)

    operation(np, a_np, slice_lhs, slice_rhs)
    operation(num, a_num, slice_lhs, slice_rhs)

    assert allclose(a_np, a_num)


@pytest.mark.skipif(
    NumpyVersion(np.__version__) < "2.1.0", reason="old np broken"
)
def test_advanced_indexing_setitem():
    arr = num.array([0, 1, 2])
    idx = num.array([2, 1, 0])
    arr[idx] = arr

    assert np.array_equal(arr, [2, 1, 0])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
