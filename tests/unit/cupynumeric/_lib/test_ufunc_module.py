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

import gc
import sys
from typing import Any

import numpy as np
import pytest

import cupynumeric as num


def import_ufunc_module() -> Any:
    return pytest.importorskip("cupynumeric._lib.ufunc")


def test_private_ufunc_module_importable() -> None:
    module = import_ufunc_module()
    assert module._is_available() is True


def test_native_array_handle_is_private_boundary_type() -> None:
    module = import_ufunc_module()

    assert module._native_array_handle_kind() == "NativeArrayHandle"
    assert module._NativeArrayHandle.__name__ == "_NativeArrayHandle"
    with pytest.raises(TypeError):
        module._NativeArrayHandle()


def test_native_array_handle_capabilities_are_registered() -> None:
    module = import_ufunc_module()

    assert module._native_array_handle_capabilities() == (
        "shape",
        "dtype",
        "writeability",
        "store",
        "result_allocation",
        "conversion",
        "task_launch",
        "ndarray_extraction",
        "ndarray_wrapping",
    )


def test_ndarray_extracts_native_array_handle() -> None:
    module = import_ufunc_module()

    array = num.zeros((2, 3))
    handle = array._native_array_handle()

    assert isinstance(handle, module._NativeArrayHandle)
    assert handle.dim() == 2
    assert handle.shape() == [2, 3]
    assert handle.writeable() is True


def test_ndarray_extraction_preserves_writeability() -> None:
    module = import_ufunc_module()

    array = num.zeros((2, 3))
    array.setflags(write=False)

    handle = array._native_array_handle()

    assert isinstance(handle, module._NativeArrayHandle)
    assert handle.writeable() is False


def test_ndarray_wraps_native_array_handle() -> None:
    module = import_ufunc_module()

    array = num.zeros((2, 3))
    handle = array._native_array_handle()

    wrapped = num.ndarray._from_native_array_handle(handle)

    assert isinstance(handle, module._NativeArrayHandle)
    assert isinstance(wrapped, num.ndarray)
    assert wrapped is not array
    assert wrapped.shape == (2, 3)
    assert wrapped.flags["W"] is True
    assert wrapped._thunk.base.equal_storage(array._thunk.base)


def test_ndarray_wrapping_preserves_writeability() -> None:
    module = import_ufunc_module()

    array = num.zeros((2, 3))
    array.setflags(write=False)
    handle = array._native_array_handle()

    wrapped = num.ndarray._from_native_array_handle(handle)

    assert isinstance(handle, module._NativeArrayHandle)
    assert isinstance(wrapped, num.ndarray)
    assert wrapped.flags["W"] is False
    assert wrapped._thunk.base.equal_storage(array._thunk.base)
    with pytest.raises(ValueError, match="not writeable"):
        wrapped[0, 0] = 1


def test_native_array_handle_keeps_store_alive_after_ndarray_release() -> None:
    module = import_ufunc_module()

    array = num.array([[1, 2, 3], [4, 5, 6]])
    expected = array.__array__()
    handle = array._native_array_handle()

    del array
    gc.collect()

    wrapped = num.ndarray._from_native_array_handle(handle)

    assert isinstance(handle, module._NativeArrayHandle)
    assert wrapped.shape == (2, 3)
    assert np.array_equal(wrapped.__array__(), expected)


def test_wrapped_ndarray_keeps_store_alive_after_handle_release() -> None:
    import_ufunc_module()

    array = num.array([[1, 2, 3], [4, 5, 6]])
    expected = array.__array__()
    handle = array._native_array_handle()
    wrapped = num.ndarray._from_native_array_handle(handle)

    del handle
    del array
    gc.collect()

    assert isinstance(wrapped, num.ndarray)
    assert wrapped.shape == (2, 3)
    assert np.array_equal(wrapped.__array__(), expected)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
