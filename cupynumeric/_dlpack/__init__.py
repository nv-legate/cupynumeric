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

from enum import Enum
from typing import TYPE_CHECKING, Protocol

from .. import ndarray
from .._thunk.deferred import DeferredArray

if TYPE_CHECKING:
    from typing_extensions import CapsuleType


class SupportsDLPack(Protocol):
    def __dlpack__(self) -> CapsuleType: ...


def from_dlpack(
    x: SupportsDLPack,
    /,
    *,
    device: tuple[Enum, int] | None = None,
    copy: bool | None = None,
) -> ndarray:
    """Convert an object supporting the DLPack protocol to cuPyNumeric array.

    Parameters
    ----------
    x : SupportsDLPack
        An object to convert
    device : tuple[Enum, int]
    copy : bool
        Whether to copy the data or not

    Returns
    -------
        ndarray

    """
    from legate.core import from_dlpack  # type: ignore [attr-defined]

    store = from_dlpack(x, device=device, copy=copy)

    return ndarray._from_thunk(DeferredArray(store))
