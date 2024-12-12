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
from utils.generators import mk_0to1_array

import cupynumeric as num


def random_array(lib, ndim):
    return mk_0to1_array(lib, ndim * (5,))


def nd_view_of_1d(lib, ndim):
    return lib.arange(5**ndim).reshape(ndim * (5,))


def tuple_set(tup, idx, val):
    lis = list(tup)
    lis[idx] = val
    return tuple(lis)


def no_overlap(lib, ndim, input_gen):
    # slices disjoint on all dimensions
    x = input_gen(lib, ndim)
    x[ndim * (slice(0, 2),)] = x[ndim * (slice(2, 4),)]
    yield x
    # slices are parallel (full) hyperplanes
    for d in range(ndim):
        x = input_gen(lib, ndim)
        lhs = tuple_set(ndim * (slice(None),), d, 1)
        rhs = tuple_set(ndim * (slice(None),), d, 2)
        x[lhs] = x[rhs]
        yield x
    # slices are parallel (partial) hyperplanes
    for d in range(ndim):
        x = input_gen(lib, ndim)
        lhs = tuple_set(ndim * (slice(3, 5),), d, 1)
        rhs = tuple_set(ndim * (slice(3, 5),), d, 2)
        x[lhs] = x[rhs]
        yield x
    # slices disjoint on one dimension
    for d in range(ndim):
        x = input_gen(lib, ndim)
        rhs = ndim * (slice(2, 4),)
        lhs = tuple_set(rhs, d, slice(0, 2))
        x[lhs] = x[rhs]
        yield x


def partial_overlap(lib, ndim):
    # slices partially overlap on one dimension, all elements on others
    for d in range(ndim):
        x = random_array(lib, ndim)
        lhs = tuple_set(ndim * (slice(None),), d, slice(0, 2))
        rhs = tuple_set(ndim * (slice(None),), d, slice(1, 3))
        x[lhs] = x[rhs]
        yield x
    # slices partially overlap on one dimension, some elements on others
    for d in range(ndim):
        x = random_array(lib, ndim)
        lhs = tuple_set(ndim * (slice(3, 5),), d, slice(0, 2))
        rhs = tuple_set(ndim * (slice(3, 5),), d, slice(1, 3))
        x[lhs] = x[rhs]
        yield x
    # slices partially overlap on all dimensions
    x = random_array(lib, ndim)
    x[ndim * (slice(0, 2),)] = x[ndim * (slice(1, 3),)]
    yield x


def full_overlap(lib, ndim):
    # overlap on full array
    x = random_array(lib, ndim)
    x[:] = x
    yield x
    x = random_array(lib, ndim)
    y = lib.zeros(ndim * (5,))
    x[:] = y[:]
    yield x
    # special cases of full-overlap self-copy occuring because Python
    # translates expressions like x[:] += y into x[:] = x[:].__iadd__(y)
    # which eventually executes x[:] = x[:]
    x = random_array(lib, ndim)
    x[ndim * (slice(0, 3),)] += 2
    yield x
    x = random_array(lib, ndim)
    x[ndim * (slice(0, 3),)] = x[ndim * (slice(0, 3),)]
    yield x
    x = random_array(lib, ndim)
    x[:] += 2
    yield x
    # overlap on a (full) hyperplane
    for d in range(ndim):
        x = random_array(lib, ndim)
        sl = tuple_set(ndim * (slice(None),), d, 1)
        x[sl] = x[sl]
        yield x
    # overlap on a (partial) hyperplane
    for d in range(ndim):
        x = random_array(lib, ndim)
        sl = tuple_set(ndim * (slice(3, 5),), d, 1)
        x[sl] = x[sl]
        yield x
    # overlap on a block
    for d in range(ndim):
        x = random_array(lib, ndim)
        sl = ndim * (slice(2, 4),)
        x[sl] = x[sl]
        yield x


def array_gen(lib, ndim):
    # no overlap between source and destination slice
    yield from no_overlap(lib, ndim, random_array)
    # no overlap at view level, but possible overlap on underlying array
    # TODO: disable until legate#40 is fixed
    # yield from no_overlap(lib, ndim, nd_view_of_1d)
    # partial overlap between source and destination slice
    yield from partial_overlap(lib, ndim)
    # full overlap between source and destination slice
    yield from full_overlap(lib, ndim)


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
def test_overlap(ndim):
    for np_arr, num_arr in zip(array_gen(np, ndim), array_gen(num, ndim)):
        assert np.array_equal(np_arr, num_arr)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
