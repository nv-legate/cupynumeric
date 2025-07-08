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
import re

import numpy as np
import pytest

import cupynumeric as num


class TestRand:
    def test_rand_null(self) -> None:
        L1 = num.random.rand()
        assert L1.dtype.kind == "f"
        assert L1.ndim == 0

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    @pytest.mark.parametrize("size", (0, 1, 3))
    def test_rand(self, size: int) -> None:
        L1 = num.random.rand(size)
        L2 = np.random.rand(size)
        assert L1.ndim == L2.ndim == 1

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    def test_rand_2d(self) -> None:
        L1 = num.random.rand(3, 3)
        L2 = np.random.rand(3, 3)
        assert L1.ndim == L2.ndim == 2

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    def test_float(self) -> None:
        msg = r"expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.random.rand(1.5)
        msg = r"'float' object cannot be interpreted as an integer"
        with pytest.raises(TypeError, match=msg):
            np.random.rand(1.5)

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    def test_negative_value(self) -> None:
        msg = r"Extent must be a positive number"
        with pytest.raises(ValueError, match=msg):
            num.random.rand(-2, -2)
        msg = r"negative dimensions are not allowed"
        with pytest.raises(ValueError, match=msg):
            np.random.rand(-2, -2)

    @pytest.mark.xfail(
        reason="https://github.com/nv-legate/cunumeric.internal/issues/199"
    )
    def test_same_seed(self) -> None:
        num.random.seed(10)
        L1 = num.random.rand(3, 3)
        num.random.seed(10)
        L2 = num.random.rand(3, 3)
        assert np.array_equal(L1, L2)


class TestRandn:
    def test_randn_null(self) -> None:
        L1 = num.random.randn()
        assert L1.dtype.kind == "f"
        assert L1.ndim == 0

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    @pytest.mark.parametrize("size", (0, 1, 3))
    def test_randn(self, size: int) -> None:
        L1 = num.random.randn(size)
        L2 = np.random.randn(size)
        assert L1.ndim == L2.ndim == 1

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    def test_2d(self) -> None:
        L1 = num.random.randn(3, 3)
        L2 = np.random.randn(3, 3)
        assert L1.ndim == L2.ndim == 2

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    def test_float(self) -> None:
        msg = r"expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.random.randn(1.5)
        msg = r"'float' object cannot be interpreted as an integer"
        with pytest.raises(TypeError, match=msg):
            np.random.randn(1.5)

    def test_negative_value(self) -> None:
        with pytest.raises(ValueError):
            num.random.randn(-2, -2)
        msg = r"negative dimensions are not allowed"
        with pytest.raises(ValueError, match=msg):
            np.random.randn(-2, -2)

    @pytest.mark.xfail(
        reason="https://github.com/nv-legate/cunumeric.internal/issues/199"
    )
    def test_same_seed(self) -> None:
        num.random.seed(10)
        L1 = num.random.randn(3, 3)
        num.random.seed(10)
        L2 = num.random.randn(3, 3)
        assert np.array_equal(L1, L2)


class TestRandom:
    def test_random_null(self) -> None:
        L1 = num.random.random()
        assert L1.dtype.kind == "f"
        assert L1.ndim == 1

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    @pytest.mark.parametrize("size", (0, 1, 3))
    def test_random(self, size: int) -> None:
        L1 = num.random.random(size)
        L2 = np.random.random(size)
        assert L1.ndim == L2.ndim == 1

    def test_float(self) -> None:
        msg = r"expected a sequence of integers or a single integer"
        with pytest.raises(TypeError, match=msg):
            num.random.random(1.5)
        msg = r"expected a sequence of integers or a single integer, got '1.5'"
        with pytest.raises(TypeError, match=msg):
            np.random.random(1.5)

    def test_negative_value(self) -> None:
        with pytest.raises(ValueError):
            num.random.random(-2)
        msg = r"negative dimensions are not allowed"
        with pytest.raises(ValueError, match=msg):
            np.random.random(-2)

    @pytest.mark.xfail(
        reason="https://github.com/nv-legate/cunumeric.internal/issues/199"
    )
    def test_same_seed(self) -> None:
        num.random.seed(10)
        L1 = num.random.random(3)
        num.random.seed(10)
        L2 = num.random.random(3)
        assert np.array_equal(L1, L2)


class TestRandomSeed:
    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    def test_none(self) -> None:
        num.random.seed()
        L1 = num.random.randn(3, 3)
        np.random.seed()
        L2 = np.random.randn(3, 3)
        assert L1.ndim == L2.ndim

    @pytest.mark.xfail(
        reason="numpy failures in random.mtrand.RandomState.standard_normal"
    )
    @pytest.mark.parametrize("seed", (None, 1, 100, 20000))
    def test_seed(self, seed: int | None) -> None:
        num.random.seed(seed)
        L1 = num.random.randn(3, 3)
        np.random.seed(seed)
        L2 = np.random.randn(3, 3)
        assert L1.ndim == L2.ndim

    def test_invalid_type_seed(self) -> None:
        msg = re.escape(
            "Cannot cast scalar from dtype('<U3') to dtype('int64') "
            "according to the rule 'safe'"
        )
        with pytest.raises(TypeError, match=msg):
            np.random.seed("123")
        with pytest.raises(TypeError, match=msg):
            np.random.seed("abc")

        msg = re.escape(
            "Cannot cast scalar from dtype('float64') to dtype('int64') "
            "according to the rule 'safe'"
        )
        with pytest.raises(TypeError, match=msg):
            np.random.seed(10.5)

        msg_num = re.escape("seed must be an integer or None")
        with pytest.raises(TypeError, match=msg_num):
            num.random.seed("123")
        with pytest.raises(TypeError, match=msg_num):
            num.random.seed("abc")
        with pytest.raises(TypeError, match=msg_num):
            num.random.seed(10.5)

    def test_invalid_value_seed(self) -> None:
        msg = re.escape("Seed must be between 0 and 2**32 - 1")
        with pytest.raises(ValueError, match=msg):
            np.random.seed(-10)
        with pytest.raises(ValueError, match=msg):
            np.random.seed(2**32)
        with pytest.raises(ValueError, match=msg):
            num.random.seed(-10)
        with pytest.raises(ValueError, match=msg):
            num.random.seed(2**32)


def test_RandomState() -> None:
    rdm_num = num.random.RandomState(10)
    L1 = rdm_num.randn(3, 3)
    rdm_np = np.random.RandomState(10)
    L2 = rdm_np.randn(3, 3)
    assert np.array_equal(L1, L2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
