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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._array.array import ndarray
    from os import PathLike
    from typing import BinaryIO

import numpy as np

from .creation_data import array


def load(
    file: str | bytes | PathLike[Any] | BinaryIO,
    *,
    max_header_size: int = 10000,
) -> ndarray:
    """
    Load an array from a ``.npy`` file.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the
        ``seek()`` and ``read()`` methods and must always
        be opened in binary mode.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.

    Returns
    -------
    result : array
        Data stored in the file.

    Raises
    ------
    OSError
        If the input file does not exist or cannot be read.

    See Also
    --------
    numpy.load

    Notes
    -----
    cuPyNumeric does not currently support ``.npz`` and pickled files.

    Availability
    --------
    Single CPU
    """
    return array(np.load(file, max_header_size=max_header_size))
