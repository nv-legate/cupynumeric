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

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .array import ndarray

FlagKeys = Literal["A", "ALIGNED", "W", "WRITEABLE", "X", "WRITEBACKIFCOPY"]


class flagsobj:
    """
    Information about the memory layout of the array.

    These flags don't reflect the properties of the cuPyNumeric array, but
    rather the NumPy array that will be produced if the cuPyNumeric array is
    materialized on a single node.
    """

    _array: ndarray

    def __init__(self, array: ndarray) -> None:
        # prevent infinite __setattr__ recursion
        object.__setattr__(self, "_array", array)

    def __repr__(self) -> str:
        return f"""\
  C_CONTIGUOUS : {self["C"]}
  F_CONTIGUOUS : {self["F"]}
  OWNDATA : {self["O"]}
  WRITEABLE : {self["W"]}
  ALIGNED : {self["A"]}
  WRITEBACKIFCOPY : {self["X"]}
"""

    def __eq__(self, other: Any) -> bool:
        flags = ("C", "F", "O", "W", "A", "X")
        return all(self[f] == other[f] for f in flags)

    def __getattr__(self, name: str) -> Any:
        if name == "writeable":
            return self._array._writeable
        flags = self._array.__array__().flags
        return getattr(flags, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "writeable":
            self._check_writeable(value)
            self._array._writeable = bool(value)
        else:
            flags = self._array.__array__().flags
            setattr(flags, name, value)

    def __getitem__(self, key: Any) -> bool:
        if key == "W":
            return self._array._writeable
        flags = self._array.__array__().flags
        return flags[key]

    def __setitem__(self, key: FlagKeys, value: Any) -> None:
        if key == "W":
            self._check_writeable(value)
            self._array._writeable = bool(value)
        else:
            flags = self._array.__array__().flags
            flags[key] = value

    def _check_writeable(self, value: Any) -> None:
        if value and not self._array._writeable:
            raise ValueError(
                "non-writeable cupynumeric arrays cannot be made writeable"
            )
