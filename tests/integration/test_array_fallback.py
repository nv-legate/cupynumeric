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


# ref: https://github.com/nv-legate/cupynumeric/pull/430
def test_unimplemented_method_self_fallback():
    ones = num.ones((10,))
    ones.mean()

    # This test uses std because it is currently unimplemented, and we want
    # to verify a behaviour of unimplemented ndarray method wrappers. If std
    # becomes implemeneted in the future, this assertion will start to fail,
    # and a new (unimplemented) ndarray method should be found to replace it
    assert not ones.std._cupynumeric_metadata.implemented

    ones.std()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
