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

import numpy as np
import pytest

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


def test_array_namespace_rejects_api_version() -> None:
    arr = num.asarray([1.0, 2.0, 3.0])

    with pytest.raises(
        NotImplementedError,
        match="versioned Array API negotiation is not implemented yet",
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
