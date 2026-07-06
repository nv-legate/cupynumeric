# Copyright 2026 NVIDIA Corporation
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

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import cupynumeric as num

from _benchmark.get_numpy import get_numpy


def test_get_numpy_with_numpy_returns_copy() -> None:
    source = np.array([1, 2, 3], dtype=np.int32)
    result = get_numpy(np, source)

    assert isinstance(result, np.ndarray)
    assert result.dtype == source.dtype
    np.testing.assert_array_equal(result, source)


def test_get_numpy_with_numpy_accepts_python_sequence() -> None:
    result = get_numpy(np, [1.0, 2.0, 3.0])

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_get_numpy_with_cupy_calls_get() -> None:
    expected = np.array([4, 5, 6])
    array = MagicMock()
    array.get.return_value = expected
    cupy_module = SimpleNamespace(__name__="cupy")

    result = get_numpy(cupy_module, array)  # type: ignore[arg-type]

    array.get.assert_called_once_with()
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


def test_get_numpy_with_cupynumeric_round_trip() -> None:
    expected = np.array([7.0, 8.0, 9.0])
    array = num.array(expected)
    result = get_numpy(num, array)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


def test_get_numpy_rejects_unknown_module() -> None:
    other = SimpleNamespace(__name__="jax")
    with pytest.raises(RuntimeError, match="Unsupported module jax"):
        get_numpy(other, np.array([1, 2, 3]))  # type: ignore[arg-type]


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
