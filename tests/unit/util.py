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

from itertools import chain, combinations
from typing import Any, Iterable, Iterator, TypeAlias

import pytest

Capsys: TypeAlias = pytest.CaptureFixture[str]


# ref: https://docs.python.org/3/library/itertools.html
def powerset(iterable: Iterable[Any]) -> Iterator[Any]:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def powerset_nonempty(iterable: Iterable[Any]) -> Iterator[Any]:
    return (x for x in powerset(iterable) if len(x))
