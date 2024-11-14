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

ARRAYS = (
    [1, 2, 3],
    [4j, 5j, 6j],
    [3 + 6j],
    [[1 + 4j, 2 + 5j, 3 + 6j]],
    [],
)


def strict_type_equal_array(a, b):
    return np.array_equal(a, b) and a.dtype == b.dtype


@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("arr", ARRAYS)
def test_complex_array(arr, dtype):
    # If val has complex elements, the returned type is float.
    x_np = np.array(arr, dtype)
    x_num = num.array(x_np)

    assert strict_type_equal_array(np.real(x_np), num.real(x_num))
    assert strict_type_equal_array(np.imag(x_np), num.imag(x_num))

    assert strict_type_equal_array(x_np.conj(), x_num.conj())
    assert strict_type_equal_array(x_np.real, x_num.real)
    assert strict_type_equal_array(x_np.imag, x_num.imag)


@pytest.mark.parametrize("dtype", (np.int32, np.float64))
def test_non_complex_array(dtype):
    # If val is real, the type of val is used for the output.
    arr = [1, 2, 3]
    x_np = np.array(arr, dtype)
    x_num = num.array(x_np)

    assert strict_type_equal_array(np.real(x_np), num.real(x_num))
    assert strict_type_equal_array(np.imag(x_np), num.imag(x_num))

    assert strict_type_equal_array(x_np.conj(), x_num.conj())
    assert strict_type_equal_array(x_np.real, x_num.real)
    assert strict_type_equal_array(x_np.imag, x_num.imag)


SCALARS = (1, 0.0, 1 + 1j, 1.1 + 1j, 0j)


@pytest.mark.diff
@pytest.mark.parametrize("val", SCALARS)
def test_scalar(val):
    # e.g., np.array_equal(1.1, array(1.1))
    # In numpy, it returns val as a scalar
    # In cupynumeric, it returns a 0-dim array(val)
    assert np.array_equal(np.real(val), num.real(val))
    assert np.array_equal(np.imag(val), num.imag(val))


@pytest.mark.xfail
@pytest.mark.parametrize("imag_val", ([10, 11, 12], 12))
@pytest.mark.parametrize("real_val", ([7, 8, 9], 9))
def test_assignment(real_val, imag_val):
    # In numpy, x_np.real = real_val pass
    # In cupynumeric, it rasies AttributeError: can't set attribute
    arr = [1 + 4j, 2 + 5j, 3 + 6j]
    x_np = np.array(arr)
    x_num = num.array(x_np)

    x_np.real = real_val
    x_np.imag = imag_val
    x_num.real = real_val
    x_num.imag = imag_val

    assert np.array_equal(x_np, x_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
