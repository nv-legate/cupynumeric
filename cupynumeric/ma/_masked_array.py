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
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy.typing as npt

import numpy as _np

from .._array.util import maybe_convert_to_np_ndarray
from .._utils.coverage import clone_class
from ..types import NdShape

NDARRAY_INTERNAL = {
    "__array_finalize__",
    "__array_function__",
    "__array_interface__",
    "__array_prepare__",
    "__array_priority__",
    "__array_struct__",
    "__array_ufunc__",
    "__array_wrap__",
    # Avoid auto-wrapping Array API specifics:
    "__array_namespace__",
    "device",
    "to_device",
}

MaskType = bool
nomask = MaskType(0)


@clone_class(_np.ma.MaskedArray, NDARRAY_INTERNAL, maybe_convert_to_np_ndarray)
class MaskedArray:
    _internal_ma: _np.ma.MaskedArray[Any, Any]

    def __new__(cls, *args: Any, **kw: Any) -> MaskedArray:
        return object.__new__(cls)

    def __init__(
        self,
        data: Any = None,
        mask: bool = nomask,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        subok: bool = True,
        ndmin: int = 0,
        fill_value: Any = None,
        keep_mask: Any = True,
        hard_mask: Any = None,
        shrink: bool = True,
        order: str | None = None,
    ) -> None:
        self._internal_ma = _np.ma.MaskedArray(  # type: ignore
            data=maybe_convert_to_np_ndarray(data),
            mask=maybe_convert_to_np_ndarray(mask),
            dtype=dtype,
            copy=copy,
            subok=subok,
            ndmin=ndmin,
            fill_value=fill_value,
            keep_mask=keep_mask,
            hard_mask=hard_mask,
            shrink=shrink,
            order=order,
        )

    def __array__(self, _dtype: Any = None) -> _np.ma.MaskedArray[Any, Any]:
        return self._internal_ma

    @property
    def size(self) -> int:
        return self._internal_ma.size

    @property
    def shape(self) -> NdShape:
        return cast(NdShape, self._internal_ma.shape)

    @property
    def dtype(self) -> _np.dtype[Any]:
        return cast(_np.dtype[Any], self._internal_ma.dtype)
