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

"""
cuPyNumeric
=====

Provides a distributed task-parallel implementation of the Numpy interface
with GPU acceleration.

:meta private:
"""

from __future__ import annotations

import numpy as _np

from . import linalg, random, fft, ma  # noqa: F401
from ._array.array import ndarray  # noqa: F401
from ._array.util import maybe_convert_to_np_ndarray
from ._dlpack import from_dlpack  # noqa: F401
from ._module import *  # noqa: F403
from ._ufunc import *  # noqa: F403
from ._utils.array import is_supported_dtype, local_task_array  # noqa: F401
from ._utils.coverage import clone_module

clone_module(_np, globals(), maybe_convert_to_np_ndarray)

del maybe_convert_to_np_ndarray
del clone_module
del _np


def _fixup_version() -> str:
    import os

    if (v := os.environ.get("CUPYNUMERIC_USE_VERSION")) is not None:
        return v

    from . import _version

    if hasattr(_version, "get_versions"):
        return str(_version.get_versions()["version"])  # type: ignore [no-untyped-call]
    if hasattr(_version, "__version__"):
        return str(_version.__version__)

    raise RuntimeError("Failed to determine version")


def _warn_deprecated_chunk_settings() -> None:
    import os
    import warnings

    for k in (
        "CUPYNUMERIC_MIN_CPU_CHUNK",
        "CUPYNUMERIC_MIN_GPU_CHUNK",
        "CUPYNUMERIC_MIN_OMP_CHUNK",
    ):
        if k in os.environ:
            warnings.warn(
                f"env setting {k} is deprecated, use CUPYNUMERIC_MAX_EAGER_VOLUME"
            )


__version__ = _fixup_version()

_warn_deprecated_chunk_settings()

del _fixup_version, _warn_deprecated_chunk_settings
