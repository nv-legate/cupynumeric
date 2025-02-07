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

from typing import TYPE_CHECKING

from legate.core import get_legate_runtime

from cupynumeric.config import CuPyNumericOpCode

from ._exception import LinAlgError

if TYPE_CHECKING:
    from legate.core import Library, LogicalStore

    from .._thunk.deferred import DeferredArray


def qr_single(
    library: Library, a: LogicalStore, q: LogicalStore, r: LogicalStore
) -> None:
    task = get_legate_runtime().create_auto_task(library, CuPyNumericOpCode.QR)
    task.throws_exception(LinAlgError)
    task.add_input(a)
    task.add_output(q)
    task.add_output(r)

    task.add_broadcast(a)
    task.add_broadcast(q)
    task.add_broadcast(r)

    task.execute()


def qr_deferred(a: DeferredArray, q: DeferredArray, r: DeferredArray) -> None:
    library = a.library

    qr_single(library, a.base, q.base, r.base)
