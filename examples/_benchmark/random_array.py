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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any
    from numpy.typing import DTypeLike


def random_array(
    np: ModuleType,
    shape: int | tuple[int, ...],
    dtype: DTypeLike = "float64",
    shift: int | float = 0,
    scale: int | float = 1,
) -> Any:
    """Generate a random array without creating a lot of temporary arrays."""
    rng = np.random.default_rng()
    _dtype = np.dtype(dtype)
    match _dtype.kind:
        case "b" | "i" | "u":
            lo = _dtype.type(-shift * scale)
            hi = _dtype.type((1 - shift) * scale)
            return rng.integers(
                low=lo, high=(hi + 1), size=shape, dtype=_dtype
            )
        case "f":
            r = rng.random(size=shape, dtype=_dtype)
            if shift != 0:
                np.add(r, shift, out=r)
            if scale != 1:
                np.multiply(r, scale, out=r)
            return r
        case "c":
            real_type = f"float{_dtype.itemsize * 4}"
            r = rng.random(size=shape, dtype=real_type).astype(_dtype)
            if shift != 0:
                np.add(r, shift, out=r)
            if scale != 1:
                np.multiply(r, scale, out=r)
            return r
        case _:
            raise RuntimeError(f"Unsupported dtype {dtype}")
