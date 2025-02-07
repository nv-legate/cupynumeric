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

from cupynumeric import test as nptest

MSG = (
    "cuPyNumeric cannot execute numpy.test() due to reliance "
    "on Numpy internals. For information about running the "
    "cuPyNumeric test suite, see: "
    "https://docs.nvidia.com/cupynumeric/latest/developer/index.html"
)


def test_warning() -> None:
    with pytest.warns(UserWarning) as record:
        nptest(1, 2, 3, foo=10)

    assert len(record) == 1
    assert record[0].message.args[0] == MSG


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
