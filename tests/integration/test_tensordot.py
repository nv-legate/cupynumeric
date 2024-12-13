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
from utils.contractions import check_default
from utils.generators import mk_0to1_array
from utils.utils import MAX_DIM_RANGE

import cupynumeric as num
from cupynumeric._utils.linalg import tensordot_modes


def gen_axes(a_ndim, b_ndim):
    yield from range(min(a_ndim, b_ndim, 2) + 1)
    if a_ndim >= 2 and b_ndim >= 2:
        yield ([0, 1], [0, 1])
        yield ([0, 1], [1, 0])


@pytest.mark.parametrize("b_ndim", MAX_DIM_RANGE)
@pytest.mark.parametrize("a_ndim", MAX_DIM_RANGE)
def test_tensordot(a_ndim, b_ndim):
    for axes in gen_axes(a_ndim, b_ndim):
        name = f"tensordot({a_ndim} x {b_ndim}, axes={axes})"
        modes = tensordot_modes(a_ndim, b_ndim, axes)

        def operation(lib, *args, **kwargs):
            return lib.tensordot(*args, **kwargs, axes=axes)

        check_default(name, modes, operation)


class TestTensorDotErrors:
    def setup_method(self):
        self.A = mk_0to1_array(num, (2, 3, 4))
        self.B = mk_0to1_array(num, (3, 2, 4))

    @pytest.mark.parametrize(
        "axis",
        (
            1,
            2,
            [],
            [0],
            [0, 0],
            ([0, 1], [0, 1]),
            ([0, 1], [1, 0], [0, 1]),
            ([0, 0], [0, 0]),
        ),
        ids=lambda axis: f"(axis={axis})",
    )
    def test_axis_invalid_value(self, axis):
        with pytest.raises(ValueError):
            num.tensordot(self.A, self.B, axis)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "axis", (4, ([0, 3], [1, 3])), ids=lambda axis: f"(axis={axis})"
    )
    def test_axis_invalid_index(self, axis):
        # In Numpy, for both cases, it raises IndexError
        # In cuPyNumeric, for both cases, it raises ValueError
        with pytest.raises(IndexError):
            num.tensordot(self.A, self.B, axis)

    @pytest.mark.parametrize(
        "shape", ((4,), (4, 3)), ids=lambda shape: f"(shape={shape})"
    )
    def test_out_invalid_shape(self, shape):
        out = num.zeros(shape)
        with pytest.raises(ValueError):
            num.tensordot(self.A, self.B, out=out)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
