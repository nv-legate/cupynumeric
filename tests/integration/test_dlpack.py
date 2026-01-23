# Copyright 2025 NVIDIA Corporation
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

import cupynumeric as num
from cupynumeric.runtime import runtime

SHAPES = ((1,), (10,), (4, 5), (1, 4, 5))


class TestFromDLPackNumpy:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_default(self, shape: tuple[int, ...]) -> None:
        np_array = np.random.random(shape)
        num_array = num.from_dlpack(np_array)
        assert np.array_equal(np_array, num_array)

    def test_copy_true(self) -> None:
        np_array = np.ones((3, 4, 5))
        num_array = num.from_dlpack(np_array, copy=True)
        assert np.array_equal(np_array, num_array)

        np_array[1, 2, 3] = 10
        assert not np.array_equal(np_array, num_array)

    def test_copy_false(self) -> None:
        np_array = np.ones((3, 4, 5))
        num_array = num.from_dlpack(np_array, copy=False)
        assert np.array_equal(np_array, num_array)

        np_array[1, 2, 3] = 10
        assert np.array_equal(np_array, num_array)


@pytest.mark.skipif(runtime.num_gpus == 0, reason="cupy only for GPU tests")
class TestFromDLPackCupy:
    cp = None

    @classmethod
    def setup_class(cls) -> None:
        cls.cp = pytest.importorskip(
            "cupy", reason="CuPy tests require CuPy to be installed"
        )

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_default(self, shape: tuple[int, ...]) -> None:
        cp = self.cp
        cp_array = cp.random.random(shape)
        num_array = num.from_dlpack(cp_array)
        assert num.array_equal(cp_array, num_array)

    @pytest.mark.skip(reason="copy=True only supported for copy to CPU")
    def test_copy_true(self) -> None:
        cp = self.cp
        cp_array = cp.ones((3, 4, 5))
        num_array = num.from_dlpack(cp_array, copy=True)
        assert num.array_equal(cp_array, num_array)

        cp_array[1, 2, 3] = 10
        assert not np.array_equal(cp_array, num_array)

    def test_copy_false(self) -> None:
        cp = self.cp
        cp_array = cp.ones((3, 4, 5))
        num_array = num.from_dlpack(cp_array, copy=False)
        assert num.array_equal(cp_array, num_array)

        cp_array[1, 2, 3] = 10
        assert num.array_equal(cp_array, num_array)


class TestToDLPack:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_default(self, shape: tuple[int, ...]) -> None:
        np_array = np.random.random(shape)
        num_array = num.array(np_array)

        result = np.from_dlpack(num_array)
        assert np.array_equal(np_array, result)

    def test_dlpack_deferred_thunk(self) -> None:
        num_array = num.arange(4, dtype=np.float32)
        # If eager, force conversion; in cuda stage it's already deferred.
        if runtime.is_eager_array(num_array._thunk):
            num_array._thunk.to_deferred_array(read_only=False)

        capsule = num_array.__dlpack__()
        assert capsule is not None

        device = num_array.__dlpack_device__()
        assert device is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
