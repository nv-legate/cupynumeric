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
from utils.comparisons import allclose

import cupynumeric as num

window_functions = ("bartlett", "blackman", "hamming", "hanning")


@pytest.mark.parametrize("M", (-1, 0, 1, 10, 100))
@pytest.mark.parametrize("fn", window_functions)
def test_basic_window(fn, M):
    out_np = getattr(np, fn)(M)
    out_num = getattr(num, fn)(M)

    assert allclose(out_np, out_num)


@pytest.mark.parametrize("beta", (-1.0, 0, 5, 6, 8.6))
@pytest.mark.parametrize("M", (-1, 0, 1, 10, 100))
def test_kaiser_window(M, beta):
    out_np = np.kaiser(M, beta)
    out_num = num.kaiser(M, beta)

    assert allclose(out_np, out_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
