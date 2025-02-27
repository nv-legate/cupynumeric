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

import numpy as np
import pytest
from legate.core import StoreTarget, get_legate_runtime, types as ty

import cupynumeric as num

runtime = get_legate_runtime()


def test_local_task_array_with_array() -> None:
    array = runtime.create_array(ty.int64, shape=(10,)).get_physical_array()
    result = num.local_task_array(array)
    assert result.shape == (10,)
    assert result.dtype == np.int64
    on_cpu = array.data().target not in {StoreTarget.FBMEM, StoreTarget.ZCMEM}
    assert isinstance(result, np.ndarray) == on_cpu


def test_local_task_array_with_store() -> None:
    store = runtime.create_store(ty.int64, shape=(20,)).get_physical_store()
    result = num.local_task_array(store)
    assert result.shape == (20,)
    assert result.dtype == np.int64
    on_cpu = store.target not in {StoreTarget.FBMEM, StoreTarget.ZCMEM}
    assert isinstance(result, np.ndarray) == on_cpu


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
