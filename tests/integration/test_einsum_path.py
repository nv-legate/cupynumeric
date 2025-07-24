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

import cupynumeric as num

expr = "ij,jk,kl->il"
np_a = np.empty((2, 2))
np_b = np.empty((2, 5))
np_c = np.empty((5, 2))
num_a = num.empty((2, 2))
num_b = num.empty((2, 5))
num_c = num.empty((5, 2))

OPTIMIZE = [
    True,
    False,
    "optimal",
    "greedy",
    ("optimal", 4),
    ["einsum_path", (0, 1), (0, 1)],
]


@pytest.mark.parametrize("optimize", OPTIMIZE)
def test_einsum_path(optimize):
    path_np, _ = np.einsum_path(expr, np_a, np_b, np_c, optimize=optimize)
    path_num, _ = num.einsum_path(expr, num_a, num_b, num_c, optimize=optimize)
    assert path_np == path_num


OPTIMIZE_OPPOSITE = [2, 2.2, ("optimal", 4, 5)]


@pytest.mark.xfail
@pytest.mark.parametrize("optimize", OPTIMIZE_OPPOSITE)
def test_einsum_path_optimize_opposite(optimize):
    expected_exc = TypeError
    with pytest.raises(expected_exc):
        path_np, _ = np.einsum_path(expr, np_a, np_b, np_c, optimize=optimize)
        # Numpy raises: TypeError: object of type 'int' has no len()
    with pytest.raises(expected_exc):
        path_num, _ = num.einsum_path(
            expr, num_a, num_b, num_c, optimize=optimize
        )
        # cuPyNumeric raises ValueError: einsum_path: unexpected value
        # for optimize: 2


@pytest.mark.xfail
def test_einsum_path_optimize_none():
    optimize = None
    path_np, _ = np.einsum_path(expr, np_a, np_b, np_c, optimize=optimize)
    # Numpy returns results
    path_num, _ = num.einsum_path(expr, num_a, num_b, num_c, optimize=optimize)
    # cupynumeric raises ValueError: einsum_path: unexpected value
    # for optimize: None
    assert path_np == path_num


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
