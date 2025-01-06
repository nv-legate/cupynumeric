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

import cupynumeric as num


@pytest.mark.parametrize(
    "in_shape, out_shape",
    [
        ((1, 1, 10), (10,)),
        ((6, 1, 1), (1, 1, 6)),
        ((12, 1), (1, 3, 4)),
        ((12, 1, 4), (2, 3, 2, 4)),
    ],
)
def test_reshape_no_copy(in_shape: tuple, out_shape: tuple) -> None:
    x = num.zeros(in_shape, dtype=num.int32)
    y = num.reshape(x, out_shape)
    x.fill(1)

    assert y.shape == out_shape
    assert num.sum(y) != 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
