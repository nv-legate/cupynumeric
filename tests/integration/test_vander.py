# Copyright 2026 NVIDIA Corporation
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
from utils.utils import check_module_function

import cupynumeric as num

NS = (None, 0, 1, 3, 6)

XS = ([1, 2, 3, 5], [2], [0, 1, 2, 3, 4, 5, 6, 7], [1.0, 2.5, -3.0, 4.0], [])

DTYPES = (np.int32, np.int64, np.float32, np.float64, np.complex128, bool)


@pytest.mark.parametrize(
    "increasing", (False, True), ids=lambda v: f"(inc={v})"
)
@pytest.mark.parametrize("N", NS, ids=lambda n: f"(N={n})")
@pytest.mark.parametrize("x", XS, ids=str)
def test_vander_basic(x, N, increasing):
    print_msg = f"np & cupynumeric.vander({x}, N={N}, increasing={increasing})"
    check_module_function(
        "vander", [x], {"N": N, "increasing": increasing}, print_msg
    )


@pytest.mark.parametrize(
    "increasing", (False, True), ids=lambda v: f"(inc={v})"
)
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_vander_dtype(dtype, increasing):
    x = np.array([1, 2, 3, 4], dtype=dtype)
    print_msg = f"np & cupynumeric.vander({x}, dtype={dtype})"
    check_module_function(
        "vander", [x], {"N": 3, "increasing": increasing}, print_msg
    )


def test_vander_default_square():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    print_msg = f"np & cupynumeric.vander({x})"
    check_module_function("vander", [x], {}, print_msg)


class TestVanderErrors:
    def test_2d_input(self) -> None:
        msg = "one-dimensional"
        with pytest.raises(ValueError, match=msg):
            num.vander([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            np.vander([[1, 2], [3, 4]])

    def test_negative_N(self) -> None:
        with pytest.raises(ValueError):
            num.vander([1, 2, 3], N=-1)
        with pytest.raises(ValueError):
            np.vander([1, 2, 3], N=-1)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
