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

import numpy

from types import ModuleType
from typing import Any


def get_numpy(np: ModuleType, a: Any) -> numpy.ndarray:
    """Convert an array from another package to numpy."""
    match np.__name__:
        case "numpy":
            return numpy.array(a)
        case "cupy":
            return numpy.array(a.get())
        case "cupynumeric":
            return numpy.array(a.__array__())
        case _:
            raise RuntimeError(f"Unsupported module {np.__name__}")
