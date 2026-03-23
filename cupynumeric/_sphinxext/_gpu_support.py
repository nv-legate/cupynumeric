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

from enum import Enum
from typing import Any


class GPUSupport(Enum):
    YES = 1
    NO = 2
    PARTIAL = 3


def parse_gpu_support_from_docstring(
    obj: Any,
) -> tuple[GPUSupport, GPUSupport]:
    """
    Extract GPU support from function docstring.

    Looks for these patterns in the docstring:
    - "Single GPU" -> single_gpu = YES
    - "Multiple GPUs" -> multi_gpu = YES
    - "Multiple GPUs (partial)" -> multi_gpu = PARTIAL

    Parameters
    ----------
    obj : Any
            Function or method object

    Returns
    -------
    single_gpu : GPUSupport
            Single GPU support level
    multi_gpu : GPUSupport
            Multi GPU support level
    """
    doc = getattr(obj, "__doc__", None) or ""

    # Parse multi-GPU support
    if "Multiple GPUs (partial)" in doc:
        multi = GPUSupport.PARTIAL
    elif "Multiple GPUs" in doc:
        multi = GPUSupport.YES
    else:
        multi = GPUSupport.NO

    # Parse single-GPU support
    if "Single GPU" in doc:
        single = GPUSupport.YES
    else:
        # If multi-GPU works, single-GPU definitely works
        single = multi if multi != GPUSupport.NO else GPUSupport.NO

    return single, multi
