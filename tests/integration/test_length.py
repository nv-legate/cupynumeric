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

LIST_X = [1, 2, 3]


def test_from_list():
    x = num.array(LIST_X)
    assert len(x) == len(LIST_X)


def test_random():
    N = 100
    x = num.random.random(N)
    assert N == len(x)


def test_binop():
    x = num.array([1, 2, 3, 4])
    y = num.array([1, 2, 3, 4])
    z = x + y
    assert len(x) == len(y) == len(z)


def test_method():
    x = num.array(LIST_X)
    x = num.sqrt(x)
    assert len(x) == len(LIST_X)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
