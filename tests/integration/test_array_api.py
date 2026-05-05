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

import re

import numpy as np
import pytest

from legate.core import LEGATE_MAX_DIM

import cupynumeric as num

try:
    import array_api_compat
except ModuleNotFoundError as exc:
    array_api_compat = None
    array_api_compat_import_error = exc
else:
    array_api_compat_import_error = None


@pytest.mark.parametrize(
    ("alias", "target"),
    (
        ("acos", "arccos"),
        ("acosh", "arccosh"),
        ("asin", "arcsin"),
        ("asinh", "arcsinh"),
        ("atan", "arctan"),
        ("atanh", "arctanh"),
        ("concat", "concatenate"),
        ("permute_dims", "transpose"),
    ),
)
def test_array_api_alias_exports(alias: str, target: str) -> None:
    assert getattr(num, alias) is getattr(num, target)


def test_array_namespace_default() -> None:
    arr = num.asarray([1.0, 2.0, 3.0])

    xp = arr.__array_namespace__()

    assert xp is num


def test_array_api_version() -> None:
    assert num.__array_api_version__ == "2025.12"


def test_array_namespace_accepts_supported_api_version() -> None:
    arr = num.asarray([1.0, 2.0, 3.0])

    xp = arr.__array_namespace__(api_version="2025.12")

    assert xp is num


def test_array_namespace_rejects_unsupported_api_version() -> None:
    arr = num.asarray([1.0, 2.0, 3.0])

    with pytest.raises(
        NotImplementedError,
        match=(
            "currently only implements the "
            f"{re.escape(num.__array_api_version__)} Array API namespace, "
            "got api_version='2024.12'"
        ),
    ):
        arr.__array_namespace__(api_version="2024.12")


def test_array_namespace_smoke_functions_return_cupynumeric_arrays() -> None:
    arr_np = np.asarray([[0.0, 0.5], [1.0, -0.5]])
    arr = num.asarray(arr_np)
    xp = arr.__array_namespace__()

    mean_result = xp.mean(arr)
    sin_result = xp.sin(arr)
    acos_result = xp.acos(arr)
    concat_result = xp.concat((arr, arr), axis=0)
    permute_result = xp.permute_dims(arr, axes=(1, 0))

    assert isinstance(mean_result, num.ndarray)
    assert np.allclose(mean_result.__array__(), np.mean(arr_np))

    assert isinstance(sin_result, num.ndarray)
    assert np.allclose(sin_result.__array__(), np.sin(arr_np))

    assert isinstance(acos_result, num.ndarray)
    assert np.allclose(acos_result.__array__(), np.arccos(arr_np))

    assert isinstance(concat_result, num.ndarray)
    assert np.array_equal(
        concat_result.__array__(), np.concatenate((arr_np, arr_np), axis=0)
    )

    assert isinstance(permute_result, num.ndarray)
    assert np.array_equal(
        permute_result.__array__(), np.transpose(arr_np, axes=(1, 0))
    )


def test_array_api_compat_namespace_dispatch() -> None:
    if array_api_compat_import_error is not None:
        pytest.fail(
            "array_api_compat must be installed to exercise Array API "
            "namespace dispatch in this test environment"
        )

    assert array_api_compat is not None
    arr = num.asarray([1.0, 2.0, 3.0])

    xp = array_api_compat.array_namespace(arr)

    assert xp is num


def test_array_namespace_info_capabilities() -> None:
    info = num.__array_namespace_info__()

    assert isinstance(info, num.ArrayNamespaceInfo)
    assert info.capabilities() == {
        "boolean indexing": False,
        "data-dependent shapes": False,
        "max dimensions": LEGATE_MAX_DIM,
    }


def test_array_namespace_info_devices() -> None:
    info = num.__array_namespace_info__()

    assert info.default_device() is None
    assert info.devices() == (None,)


def test_array_namespace_info_default_dtypes() -> None:
    info = num.__array_namespace_info__()

    assert info.default_dtypes() == {
        "real floating": np.float64,
        "complex floating": np.complex128,
        "integral": np.int64,
        "indexing": np.int64,
    }
    assert info.default_dtypes(device=None)["indexing"] is np.int64


def test_array_namespace_info_dtypes() -> None:
    info = num.__array_namespace_info__()

    assert info.dtypes() == {
        "bool": np.bool_,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }
    assert info.dtypes(kind="bool") == {"bool": np.bool_}
    assert info.dtypes(kind="integral") == {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
    }
    assert info.dtypes(kind=("real floating", "complex floating")) == {
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }
    assert info.dtypes(kind="numeric") == {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }

    with pytest.raises(ValueError, match="unrecognized Array API dtype kind"):
        info.dtypes(kind="invalid")


def test_array_api_empty_accepts_default_device() -> None:
    arr = num.empty(1, dtype=np.int64, device=None)

    assert isinstance(arr, num.ndarray)
    assert arr.dtype == np.dtype(np.int64)


def test_array_api_device_is_none() -> None:
    arr = num.asarray([1, 2, 3])

    assert arr.device is None


def test_array_api_to_device_returns_self() -> None:
    arr = num.asarray([1, 2, 3])

    assert arr.to_device(None) is arr
    assert arr.to_device(arr.device) is arr
    assert arr.to_device(None, stream=None) is arr


@pytest.mark.parametrize("stream", (0, 1, "stream-0"))
def test_array_api_to_device_rejects_stream(stream: object) -> None:
    arr = num.asarray([1, 2, 3])

    with pytest.raises(NotImplementedError, match="stream argument"):
        arr.to_device(None, stream=stream)


def test_array_api_rejects_non_default_device() -> None:
    info = num.__array_namespace_info__()

    with pytest.raises(
        ValueError, match="only supports device=None, got device='gpu'"
    ):
        info.default_dtypes(device="gpu")
    with pytest.raises(
        ValueError, match="only supports device=None, got device='gpu'"
    ):
        info.dtypes(device="gpu")
    with pytest.raises(
        ValueError, match="only supports device=None, got device='gpu'"
    ):
        num.empty(1, device="gpu")


def test_array_api_to_device_rejects_non_default_device() -> None:
    arr = num.asarray([1, 2, 3])

    with pytest.raises(
        ValueError, match="only supports device=None, got device='gpu'"
    ):
        arr.to_device("gpu")


@pytest.mark.parametrize(
    "shape", ((3, 4), (2, 3, 4), (2, 3, 4, 5), (0, 5), (2, 0, 3), (2, 3, 0))
)
def test_array_api_mT_swaps_last_two_axes(shape: tuple[int, ...]) -> None:
    arr_np = np.arange(int(np.prod(shape))).reshape(shape)
    arr = num.asarray(arr_np)

    result = arr.mT

    expected = np.swapaxes(arr_np, -1, -2)
    assert isinstance(result, num.ndarray)
    assert result.shape == expected.shape
    assert np.array_equal(result.__array__(), expected)

    round_trip = result.mT
    assert round_trip.shape == arr.shape
    assert np.array_equal(round_trip.__array__(), arr.__array__())


def test_array_api_mT_preserves_writeability() -> None:
    arr = num.asarray(np.arange(24).reshape(2, 3, 4))
    arr.flags["W"] = False

    result = arr.mT

    assert not result.flags["W"]


@pytest.mark.parametrize("obj", (1, [1, 2, 3]))
def test_array_api_mT_rejects_less_than_two_dimensions(obj: object) -> None:
    arr = num.asarray(obj)

    with pytest.raises(ValueError, match="at least two dimensions"):
        arr.mT


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
