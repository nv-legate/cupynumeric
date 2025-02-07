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


def test_basic():
    x = num.array([[1, 2, 3], [4, 5, 6]])
    y = num.array([[7, 8, 9], [10, 11, 12]])
    xc = x.copy()
    yc = y.copy()
    x[0, :] = [7, 8, 9]
    y = num.array([[10, 12, 13], [25, 26, 27]])
    w = x + y
    wc = xc + yc
    assert np.array_equal(w, [[17, 20, 22], [29, 31, 33]])
    assert np.array_equal(wc, [[8, 10, 12], [14, 16, 18]])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
