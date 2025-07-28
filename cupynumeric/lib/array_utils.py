# Copyright 2025 NVIDIA Corporation
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

from typing import Iterable

from .._utils import is_np2

if is_np2:
    from numpy.lib.array_utils import normalize_axis_tuple as _nat
    from numpy.lib.array_utils import normalize_axis_index as _nai
else:
    from numpy.core.numeric import normalize_axis_tuple as _nat  # type: ignore
    from numpy.core.multiarray import normalize_axis_index as _nai  # type: ignore


def normalize_axis_tuple(
    axis: int | Iterable[int],
    ndim: int,
    argname: str | None = None,
    allow_duplicate: bool | None = None,
) -> tuple[int, int]:
    return _nat(axis, ndim, argname, allow_duplicate)


def normalize_axis_index(
    axis: int, ndim: int, msg_prefix: str | None = None
) -> int:
    return _nai(axis, ndim, msg_prefix)


del is_np2
