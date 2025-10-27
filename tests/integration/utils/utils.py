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

import cupynumeric as num

from numpy.exceptions import AxisError  # noqa: F401


def compare_array(a, b, check_type=True):
    """
    Compare two array using zip method.
    """
    if check_type:
        if a.dtype != b.dtype:
            return False, [a, b]

    if len(a) != len(b):
        return False, [a, b]
    else:
        for each in zip(a, b):
            if not np.array_equal(*each):
                return False, each
    return True, None


def compare_array_and_print_results(a, b, print_msg, check_type=True):
    """
    Compare two arrays and print results.
    """
    if isinstance(a, list) or isinstance(a, tuple):
        is_equal, err_arr = compare_array(a, b, check_type=False)
        assert is_equal, (
            f"Failed, {print_msg}\n"
            f"numpy result: {err_arr[0]}\n"
            f"cupynumeric_result: {err_arr[1]}\n"
            f"cupynumeric and numpy shows"
            f" different result\n"
        )
        print(f"Passed, {print_msg}")

    else:
        is_equal, err_arr = compare_array(a, b, check_type=check_type)
        assert is_equal, (
            f"Failed, {print_msg}\n"
            f"numpy result: {err_arr[0]}, {a.shape}\n"
            f"cupynumeric_result: {err_arr[1]}, {b.shape}\n"
            f"cupynumeric and numpy shows"
            f" different result\n"
        )
        print(
            f"Passed, {print_msg}, np: ({a.shape}, {a.dtype})"
            f", cupynumeric: ({b.shape}, {b.dtype})"
        )


def check_array_method(
    ndarray_np, fn, args, kwargs, print_msg, check_type=True
):
    """
    Run np_array.func and num_array.func respectively and compare results
    """

    ndarray_num = num.array(ndarray_np)
    a = getattr(ndarray_np, fn)(*args, **kwargs)
    b = getattr(ndarray_num, fn)(*args, **kwargs)
    compare_array_and_print_results(a, b, print_msg, check_type=check_type)


def check_module_function(fn, args, kwargs, print_msg, check_type=True):
    """
    Run np.func and num.func respectively and compare results
    """

    a = getattr(np, fn)(*args, **kwargs)
    b = getattr(num, fn)(*args, **kwargs)
    compare_array_and_print_results(a, b, print_msg, check_type=check_type)


# MAX_DIM_RANGE is a list of array dimensions, that is used to test APIs
# on different array dims. We reduce this list to a sub-set of possible
# dimensions to reduce walltime for testing
MAX_DIM_RANGE = list(range(min(4, LEGATE_MAX_DIM)))
if LEGATE_MAX_DIM > MAX_DIM_RANGE[-1]:
    MAX_DIM_RANGE.append(LEGATE_MAX_DIM)
ONE_MAX_DIM_RANGE = MAX_DIM_RANGE[1:]
TWO_MAX_DIM_RANGE = MAX_DIM_RANGE[2:]
