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

from itertools import permutations, product

import numpy as np
from legate.core import LEGATE_MAX_DIM


def scalar_gen(lib, val):
    """
    Generates different kinds of scalar-like arrays that contain the given
    value.
    """
    # pure scalar values
    yield lib.array(val)
    # ()-shape arrays
    yield lib.full((), val)
    for ndim in range(1, LEGATE_MAX_DIM + 1):
        # singleton arrays
        yield lib.full(ndim * (1,), val)
        # singleton slices of larger arrays
        yield lib.full(ndim * (5,), val)[ndim * (slice(1, 2),)]


def mk_0to1_array(lib, shape, dtype=np.float64):
    """
    Constructs an array of the required shape, containing (in C order)
    sequential real values uniformly spaced in the range (0,1].
    """
    size = np.prod(shape)
    if size == 1:
        # Avoid zeros, since those are more likely to cause arithmetic issues
        # or produce degenerate outputs.
        return lib.full(shape, 0.5, dtype=dtype)
    return (mk_seq_array(lib, shape) / size).astype(dtype)


def mk_seq_array(lib, shape):
    """
    Constructs an array of the required shape, containing (in C order)
    sequential integer values starting from 1.
    """
    arr = lib.zeros(shape, dtype=int)
    size = np.prod(shape)
    # Don't return the reshaped array directly, instead use it to update
    # the contents of an existing array of the same shape, thus producing a
    # Store without transformations, that has been tiled in the natural way
    arr[:] = lib.arange(1, size + 1).reshape(shape)
    return arr


def broadcasts_to(tgt_shape):
    """
    Generates all shapes that broadcast to `tgt_shape`.
    """
    for mask in product([True, False], repeat=len(tgt_shape)):
        yield tuple(d if keep else 1 for (d, keep) in zip(tgt_shape, mask))


def permutes_to(tgt_shape):
    """
    Generates all the possible `(axes, src_shape)` pairs for which
    `x.transpose(axes).shape == tgt_shape`, where `x.shape == src_shape`.
    """
    for axes in permutations(range(len(tgt_shape))):
        src_shape = [-1] * len(tgt_shape)
        for i, j in enumerate(axes):
            src_shape[j] = tgt_shape[i]
        yield (axes, tuple(src_shape))


def broadcasts_to_along_axis(tgt_shape, axis, values):
    """
    Generates all shapes that broadcast to `tgt_shape` along axis for
    each value.
    """
    axis = axis % (len(tgt_shape))
    tgt_shape_axis_removed = tgt_shape[:axis] + tgt_shape[axis + 1 :]

    for s in broadcasts_to(tgt_shape_axis_removed):
        for v in values:
            shape = s[:axis] + (v,) + s[axis:]
            yield shape


def generate_item(ndim):
    """
    Generates item location for ndarray.item and ndarray.itemset
    """
    max_index = pow(4, ndim) - 1
    random_index = np.random.randint(-1, max_index)
    random_tuple = tuple(np.random.randint(0, 3) for i in range(0, ndim))
    return [random_index, max_index, random_tuple]
