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


from . import linalg, random, fft  # noqa: F401
from ._array.array import ndarray  # noqa: F401
from ._dlpack import from_dlpack  # noqa: F401
from ._module import *  # noqa: F403
from ._ufunc import *  # noqa: F403
from ._utils.array import is_supported_dtype, local_task_array  # noqa: F401

# ============================================================
# NumPy dtypes and constants
# These were previously copied by clone_module(), now explicit
# ============================================================

from numpy import (
    # Commonly used dtypes
    bool_,  # noqa: F401
    int8,  # noqa: F401
    int16,  # noqa: F401
    int32,  # noqa: F401
    int64,  # noqa: F401
    uint8,  # noqa: F401
    uint16,  # noqa: F401
    uint32,  # noqa: F401
    uint64,  # noqa: F401
    float16,  # noqa: F401
    float32,  # noqa: F401
    float64,  # noqa: F401
    complex64,  # noqa: F401
    complex128,  # noqa: F401
    # Type hierarchy (abstract base classes)
    integer,  # noqa: F401
    signedinteger,  # noqa: F401
    unsignedinteger,  # noqa: F401
    inexact,  # noqa: F401
    floating,  # noqa: F401
    complexfloating,  # noqa: F401
    # Commonly used constants
    pi,  # noqa: F401
    e,  # noqa: F401
    inf,  # noqa: F401
    nan,  # noqa: F401
    newaxis,  # noqa: F401
    # Dtype class
    dtype,  # noqa: F401
    # Info functions
    iinfo,  # noqa: F401
    finfo,  # noqa: F401
)


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


__version__ = _fixup_version()

del _fixup_version
