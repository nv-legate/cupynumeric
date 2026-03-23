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


@pytest.mark.parametrize(
    "slices",
    [
        slice(2, None, 1),
        slice(2, 4, None),
        slice(None, 4, 0.75),
        [slice(0, 4, 1), slice(3, 20, 4)],
        [slice(None, 8, None), slice(10, 25, None), slice(None, 4, None)],
        [slice(None, 8, 0.25), slice(10, 25, 15.5), slice(None, 4, 1)],
        [slice(0, 4, 1), slice(3, 3, None), slice(5, 10, 2)],
        [slice(0, 4, 1.5), slice(3, 3, None), slice(5, 10, 2)],
    ],
)
def test_mgrid(slices):
    a_np = np.mgrid[slices]
    a_num = num.mgrid[slices]
    assert np.array_equal(a_np, a_num)


class TestMGridErrors:
    def testBadSlice(self):
        msg = "slice stop cannot be None for mgrid"
        with pytest.raises(ValueError, match=msg):
            num.mgrid[slice(None, 4, None), slice(5, None, 1)]

        with pytest.raises(ValueError, match=msg):
            num.mgrid[slice(None, None, 2.0), slice(5, 10, None)]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
