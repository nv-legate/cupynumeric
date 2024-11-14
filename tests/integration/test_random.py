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
import numpy as np
import pytest

import cupynumeric as num


@pytest.mark.xfail(
    reason="https://github.com/nv-legate/cupynumeric.internal/issues/199"
)
def test_basic_num() -> None:
    num.random.seed(10)
    L1 = num.random.randn(3, 3)
    num.random.seed(10)
    L2 = num.random.randn(3, 3)
    assert np.array_equal(L1, L2)


@pytest.mark.xfail(
    reason="numpy failures in random.mtrand.RandomState.standard_normal"
)
def test_basic_np() -> None:
    np.random.seed(10)
    L1 = np.random.randn(3, 3)
    np.random.seed(10)
    L2 = np.random.randn(3, 3)
    assert np.array_equal(L1, L2)

    np.random.seed(10)
    L1 = np.random.randn(3, 3)
    L2 = np.random.randn(3, 3)
    assert not np.array_equal(L1, L2)


@pytest.mark.xfail(
    reason="https://github.com/nv-legate/cupynumeric.internal/issues/199"
)
def test_none_num() -> None:
    num.random.seed()
    L1 = num.random.randn(3, 3)
    num.random.seed()
    L2 = num.random.randn(3, 3)
    assert np.array_equal(L1, L2)

    num.random.seed()
    L1 = num.random.randn(3, 3)
    L2 = num.random.randn(3, 3)
    assert not np.array_equal(L1, L2)


@pytest.mark.xfail(
    reason="numpy failures in random.mtrand.RandomState.standard_normal"
)
def test_none_np() -> None:
    np.random.seed()
    L1 = np.random.randn(3, 3)
    np.random.seed()
    L2 = np.random.randn(3, 3)
    assert not np.array_equal(L1, L2)

    np.random.seed()
    L1 = np.random.randn(3, 3)
    L2 = np.random.randn(3, 3)
    assert not np.array_equal(L1, L2)


@pytest.mark.xfail(
    reason="numpy failures in random.mtrand.RandomState.standard_normal"
)
def test_basic_num_np() -> None:
    np.random.seed(10)
    L1 = np.random.randn(3, 3)
    num.random.seed(10)
    L2 = num.random.randn(3, 3)
    assert not np.array_equal(L1, L2)


def test_RandomState() -> None:
    rdm_num = num.random.RandomState(10)
    L1 = rdm_num.randn(3, 3)
    rdm_np = np.random.RandomState(10)
    L2 = rdm_np.randn(3, 3)
    assert np.array_equal(L1, L2)


def test_float() -> None:
    with pytest.raises(TypeError):
        np.random.seed(10.5)
        # TypeError: 'float' object cannot be interpreted as an integer
    num.random.seed(10.5)
    # cuPyNumeric passed with float


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
