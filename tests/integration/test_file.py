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

from tempfile import NamedTemporaryFile

import numpy as np
import pytest

import cupynumeric as num


def test_load():
    np_arr = np.arange(360).reshape((3, 4, 5, 6)).astype(np.float32)
    with NamedTemporaryFile(suffix=".npy", delete=False) as f:
        fname = f.name
        np.save(f, np_arr)

    num_arr = num.load(fname)
    assert isinstance(num_arr, num.ndarray)
    assert np.array_equal(np_arr, num_arr)

    with open(fname, mode="rb") as f:
        num_arr = num.load(f)
        assert isinstance(num_arr, num.ndarray)
        assert np.array_equal(np_arr, num_arr)


def test_non_existent_file():
    with pytest.raises(OSError):
        num.load("does-not-exist.npy")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
