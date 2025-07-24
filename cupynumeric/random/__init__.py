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

import numpy.random as _nprandom

from .._array.util import maybe_convert_to_np_ndarray
from .._utils.coverage import clone_module
from ..runtime import runtime

from ._random import *  # noqa: F403
from ._bitgenerator import *  # noqa: F403
from ._generator import *  # noqa: F403

clone_module(
    _nprandom,
    globals(),
    maybe_convert_to_np_ndarray,
    include_builtin_function_type=True,
)

del maybe_convert_to_np_ndarray
del clone_module
del runtime
del _nprandom
