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

import pytest

import cupynumeric as num
from cupynumeric._utils.array import SUPPORTED_DTYPES

DTYPES = SUPPORTED_DTYPES.keys()


# A simple wrapper with a legate data interface implementation for testing
class Wrapper:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    @property
    def __legate_data_interface__(self):
        return self.wrapped


@pytest.mark.parametrize("dtype", DTYPES)
def test_roundtrip(dtype):
    arr1 = num.array([1, 2, 3, 4], dtype=dtype)
    data = Wrapper(arr1.__legate_data_interface__)
    arr2 = num.asarray(data)
    assert num.array_equal(arr1, arr2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
