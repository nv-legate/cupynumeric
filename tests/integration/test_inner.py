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

import pytest
from legate.core import LEGATE_MAX_DIM
from utils.contractions import check_default
from utils.generators import mk_0to1_array

import cupynumeric as num
from cupynumeric._utils.linalg import inner_modes


@pytest.mark.parametrize(
    "b_ndim",
    (
        0,
        1,
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
@pytest.mark.parametrize(
    "a_ndim",
    (
        0,
        1,
        2,
        3,
        4,
        LEGATE_MAX_DIM,
    ),
)
def test_inner(a_ndim, b_ndim):
    name = f"inner({a_ndim} x {b_ndim})"
    modes = inner_modes(a_ndim, b_ndim)

    def operation(lib, *args, **kwargs):
        return lib.inner(*args, **kwargs)

    check_default(name, modes, operation)


class TestInnerErrors:
    def setup_method(self):
        self.A = mk_0to1_array(num, (5, 3))
        self.B = mk_0to1_array(num, (2, 3))

    @pytest.mark.parametrize(
        "shapeA",
        ((3,), (4, 3), (5, 4, 3)),
        ids=lambda shapeA: f"(shapeA={shapeA})",
    )
    def test_a_b_invalid_shape(self, shapeA):
        A = mk_0to1_array(num, shapeA)
        B = mk_0to1_array(num, (3, 2))
        with pytest.raises(ValueError):
            num.inner(A, B)

    @pytest.mark.parametrize(
        "shape", ((5,), (2,), (5, 3)), ids=lambda shape: f"(shape={shape})"
    )
    def test_out_invalid_shape(self, shape):
        out = num.zeros(shape)
        with pytest.raises(ValueError):
            num.inner(self.A, self.B, out=out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
