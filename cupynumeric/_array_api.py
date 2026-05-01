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

from collections.abc import Iterable
from typing import Any

import numpy as np

from legate.core import LEGATE_MAX_DIM

__array_api_version__ = "2025.12"

_SUPPORTED_DEVICE = None

_BOOL_DTYPES: dict[str, type[np.generic]] = {"bool": np.bool_}
_SIGNED_INTEGER_DTYPES: dict[str, type[np.generic]] = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}
_UNSIGNED_INTEGER_DTYPES: dict[str, type[np.generic]] = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}
_REAL_FLOATING_DTYPES: dict[str, type[np.generic]] = {
    "float32": np.float32,
    "float64": np.float64,
}
_COMPLEX_FLOATING_DTYPES: dict[str, type[np.generic]] = {
    "complex64": np.complex64,
    "complex128": np.complex128,
}

_KIND_TO_DTYPES = {
    "bool": _BOOL_DTYPES,
    "signed integer": _SIGNED_INTEGER_DTYPES,
    "unsigned integer": _UNSIGNED_INTEGER_DTYPES,
    "integral": _SIGNED_INTEGER_DTYPES | _UNSIGNED_INTEGER_DTYPES,
    "real floating": _REAL_FLOATING_DTYPES,
    "complex floating": _COMPLEX_FLOATING_DTYPES,
    "numeric": (
        _SIGNED_INTEGER_DTYPES
        | _UNSIGNED_INTEGER_DTYPES
        | _REAL_FLOATING_DTYPES
        | _COMPLEX_FLOATING_DTYPES
    ),
}
_ALL_DTYPES: dict[str, type[np.generic]] = (
    _BOOL_DTYPES
    | _SIGNED_INTEGER_DTYPES
    | _UNSIGNED_INTEGER_DTYPES
    | _REAL_FLOATING_DTYPES
    | _COMPLEX_FLOATING_DTYPES
)


def _check_device(device: Any) -> None:
    if device is not _SUPPORTED_DEVICE:
        raise ValueError(
            "cuPyNumeric's Array API namespace currently only supports "
            f"device=None, got device={device!r}"
        )


def _iter_kinds(kind: str | tuple[str, ...]) -> Iterable[str]:
    if isinstance(kind, str):
        return (kind,)
    return kind


class ArrayNamespaceInfo:
    """Array API namespace inspection utilities."""

    def capabilities(self) -> dict[str, bool | int]:
        return {
            # Conservative until Array API boolean indexing semantics are
            # audited end-to-end.
            "boolean indexing": False,
            # Conservative until the remaining APIs with data-dependent output
            # shapes are implemented and audited.
            "data-dependent shapes": False,
            "max dimensions": LEGATE_MAX_DIM,
        }

    def default_device(self) -> None:
        return _SUPPORTED_DEVICE

    def default_dtypes(
        self, *, device: Any | None = _SUPPORTED_DEVICE
    ) -> dict[str, type[np.generic]]:
        _check_device(device)
        return {
            "real floating": np.float64,
            "complex floating": np.complex128,
            "integral": np.int64,
            "indexing": np.int64,
        }

    def devices(self) -> tuple[Any, ...]:
        return (_SUPPORTED_DEVICE,)

    def dtypes(
        self,
        *,
        device: Any | None = _SUPPORTED_DEVICE,
        kind: str | tuple[str, ...] | None = None,
    ) -> dict[str, type[np.generic]]:
        _check_device(device)
        if kind is None:
            return dict(_ALL_DTYPES)

        result: dict[str, type[np.generic]] = {}
        for item in _iter_kinds(kind):
            try:
                dtypes = _KIND_TO_DTYPES[item]
            except KeyError:
                raise ValueError(f"unrecognized Array API dtype kind {item!r}")
            result.update(dtypes)
        return result


def __array_namespace_info__() -> ArrayNamespaceInfo:
    """Return Array API namespace inspection utilities."""
    return ArrayNamespaceInfo()
