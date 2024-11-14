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
from legate.core import LEGATE_MAX_DIM
from legate.core.utils import OrderedSet

import cupynumeric as num

from .comparisons import allclose
from .generators import mk_0to1_array


def gen_inputs_default(lib, modes):
    # 5x5x...x5
    (a_modes, b_modes, out_modes) = modes
    a_shape = (5,) * len(a_modes)
    b_shape = (5,) * len(b_modes)
    yield (mk_0to1_array(lib, a_shape), mk_0to1_array(lib, b_shape))


def gen_output_default(lib, modes, a, b):
    (a_modes, b_modes, out_modes) = modes
    out_shape = (5,) * len(out_modes)
    yield lib.zeros(out_shape)


def gen_shapes(a_modes, b_modes):
    yield ((5,) * len(a_modes), (5,) * len(b_modes))
    for mode_to_squeeze in OrderedSet(a_modes + b_modes):
        a_shape = tuple((1 if m == mode_to_squeeze else 5) for m in a_modes)
        b_shape = tuple((1 if m == mode_to_squeeze else 5) for m in b_modes)
        yield (a_shape, b_shape)


def gen_inputs_of_various_shapes(lib, modes):
    # 5x5x...x5, with up to one dimension set to 1
    # making sure common modes appear with the same extent on both arrays
    (a_modes, b_modes, out_modes) = modes
    for a_shape, b_shape in gen_shapes(a_modes, b_modes):
        if lib == num:
            print(f"  {a_shape} x {b_shape}")
        yield (mk_0to1_array(lib, a_shape), mk_0to1_array(lib, b_shape))


def gen_permutations(ndim):
    yield tuple(range(ndim))  # e.g. (0, 1, 2, 3)
    if ndim > 1:
        yield tuple(range(ndim - 1, -1, -1))  # e.g. (3, 2, 1, 0)
    if ndim > 2:
        yield (
            tuple(range(ndim // 2)) + tuple(range(ndim - 1, ndim // 2 - 1, -1))
        )  # e.g. (0, 1, 3, 2)


def gen_permuted_inputs(lib, modes):
    (a_modes, b_modes, out_modes) = modes
    a = mk_0to1_array(lib, (5,) * len(a_modes))
    b = mk_0to1_array(lib, (5,) * len(b_modes))
    for a_axes in gen_permutations(len(a_modes)):
        for b_axes in gen_permutations(len(b_modes)):
            if lib == num:
                print(f"  transpose{a_axes} x transpose{b_axes}")
            yield (a.transpose(a_axes), b.transpose(b_axes))


def gen_inputs_of_various_types(lib, modes):
    (a_modes, b_modes, out_modes) = modes
    a_shape = (5,) * len(a_modes)
    b_shape = (5,) * len(b_modes)
    for a_dtype, b_dtype in [
        (np.float16, np.float16),
        (np.float16, np.float32),
        (np.float32, np.float32),
        (np.complex64, np.complex64),
    ]:
        if lib == num:
            print(f"  {a_dtype} x {b_dtype}")
        yield (
            mk_0to1_array(lib, a_shape, a_dtype),
            mk_0to1_array(lib, b_shape, b_dtype),
        )


def gen_output_of_various_types(lib, modes, a, b):
    (a_modes, b_modes, out_modes) = modes
    out_shape = (5,) * len(out_modes)
    for out_dtype in [np.float16, np.complex64]:
        if lib == num:
            print(f"  -> {out_dtype}")
        yield lib.zeros(out_shape, out_dtype)


def _test(name, modes, operation, gen_inputs, gen_output=None, **kwargs):
    (a_modes, b_modes, out_modes) = modes
    if (
        len(OrderedSet(a_modes) | OrderedSet(b_modes) | OrderedSet(out_modes))
        > LEGATE_MAX_DIM
    ):
        # Total number of distinct modes can't exceed maximum Legion dimension,
        # because we may need to promote arrays so that one includes all modes.
        return
    print(name)
    for np_inputs, num_inputs in zip(
        gen_inputs(np, modes), gen_inputs(num, modes)
    ):
        np_res = operation(np, *np_inputs, **kwargs)
        num_res = operation(num, *num_inputs, **kwargs)
        rtol = (
            1e-02
            if any(x.dtype == np.float16 for x in np_inputs)
            or kwargs.get("dtype") == np.float16
            else 1e-05
        )
        assert allclose(np_res, num_res, rtol=rtol)
        if gen_output is not None:
            for num_out in gen_output(num, modes, *num_inputs):
                operation(num, *num_inputs, out=num_out, **kwargs)
                rtol_out = 1e-02 if num_out.dtype == np.float16 else rtol
                assert allclose(
                    num_out, num_res, rtol=rtol_out, check_dtype=False
                )


def check_default(name, modes, operation):
    name = f"{name}"
    _test(name, modes, operation, gen_inputs_default, gen_output_default)


def check_shapes(name, modes, operation):
    name = f"{name} -- various shapes"
    _test(name, modes, operation, gen_inputs_of_various_shapes)


def check_permutations(name, modes, operation):
    name = f"{name} -- permutations"
    _test(name, modes, operation, gen_permuted_inputs)


def check_types(name, modes, operation):
    name = f"{name} -- various types"
    _test(
        name,
        modes,
        operation,
        gen_inputs_of_various_types,
        gen_output_of_various_types,
        casting="unsafe",
    )
    name = f"{name} -- various types, dtype=np.float32"
    _test(
        name,
        modes,
        operation,
        gen_inputs_of_various_types,
        gen_output_of_various_types,
        dtype=np.float32,
        casting="unsafe",
    )
