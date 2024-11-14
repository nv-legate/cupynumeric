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


def svd_single(
    library: Library,
    a: LogicalStore,
    u: LogicalStore,
    s: LogicalStore,
    vh: LogicalStore,
) -> None:
    task = get_legate_runtime().create_auto_task(
        library, CuPyNumericOpCode.SVD
    )
    task.throws_exception(LinAlgError)
    task.add_input(a)
    task.add_output(u)
    task.add_output(s)
    task.add_output(vh)

    task.add_broadcast(a)
    task.add_broadcast(u)
    task.add_broadcast(s)
    task.add_broadcast(vh)

    task.execute()


def svd_deferred(
    a: DeferredArray, u: DeferredArray, s: DeferredArray, vh: DeferredArray
) -> None:
    library = a.library

    svd_single(library, a.base, u.base, s.base, vh.base)
