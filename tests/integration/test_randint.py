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


class TestRandint:
    @pytest.mark.parametrize("size", (1, 8000, (8000, 2)))
    def test_randint(self, size: int | tuple[int, ...]) -> None:
        L1 = num.random.randint(8000, size=size)
        L2 = np.random.randint(8000, size=size)
        assert L1.ndim == L2.ndim
        assert L1.dtype.kind == "i"

    def test_randint_0(self):
        L1 = num.random.randint(8000, size=0)
        L2 = np.random.randint(8000, size=0)
        assert np.array_equal(L1, L2)

    def test_low(self):
        L1 = num.random.randint(500)
        L2 = np.random.randint(500)
        assert L1 < 500
        assert L2 < 500

    def test_high(self):
        L1 = num.random.randint(500, 800)
        L2 = np.random.randint(500, 800)
        assert 500 < L1 < 800
        assert 500 < L2 < 800

    @pytest.mark.xfail(
        reason="https://github.com/nv-legate/cunumeric.internal/issues/199"
    )
    def test_same_seed(self) -> None:
        num.random.seed(13)
        L1 = num.random.randint(100)
        num.random.seed(13)
        L2 = num.random.randint(100)
        assert np.array_equal(L1, L2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
