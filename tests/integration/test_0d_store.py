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

from itertools import product

import pytest

import cupynumeric as num

SIZE = 3


def test_0d_region_backed_stores():
    arr = num.arange(9).reshape(3, 3)

    for i, j in product(range(SIZE), range(SIZE)):
        i_ind = num.array(i)
        j_ind = num.array(j)
        v = arr[i_ind, j_ind]
        assert int(v) == i * SIZE + j


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
