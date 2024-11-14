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

from typing import Any, Callable


def deep_apply(obj: Any, func: Callable[[Any], Any]) -> Any:
    """
    Apply the provided function to objects contained at any depth within a data
    structure.

    This function will recurse over arbitrary nestings of lists, tuples and
    dicts. This recursion logic is rather limited, but this function is
    primarily meant to be used for arguments of NumPy API calls, which
    shouldn't nest their arrays very deep.
    """
    if isinstance(obj, list):
        return [deep_apply(x, func) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_apply(x, func) for x in obj)
    elif isinstance(obj, dict):
        return {k: deep_apply(v, func) for k, v in obj.items()}
    else:
        return func(obj)
