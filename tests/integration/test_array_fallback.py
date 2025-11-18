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

import numpy as np
import cupynumeric as num
from cupynumeric._array.util import maybe_convert_to_np_ndarray
from cupynumeric._utils.coverage import unimplemented


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


def test_unimplemented_cupynumeric_return():
    # Use the unimplemented decorator directly so this test does not depend on
    # any specific API remaining unimplemented over time.
    wrapped_pad = unimplemented(
        np.pad,
        prefix="cupynumeric",
        name="pad_for_test",
        fallback=maybe_convert_to_np_ndarray,
    )
    assert not wrapped_pad._cupynumeric_metadata.implemented

    arr = num.array([1, 2, 3])
    expected = np.pad(np.array([1, 2, 3]), (2, 2), mode="edge")
    result = wrapped_pad(arr, (2, 2), mode="edge")

    assert isinstance(result, num.ndarray)
    # Guard against regressions where the fallback path stops producing
    # NumPy-equivalent results even though it yields a num.ndarray wrapper.
    assert np.array_equal(np.array(result), expected)


@pytest.mark.parametrize(
    "attr",
    {
        "__array_finalize__",
        "__array_function__",
        "__array_interface__",
        "__array_prepare__",
        "__array_priority__",
        "__array_struct__",
        "__array_ufunc____array_wrap__",
        "__array_namespace__",
        "device",
        "to_device",
    },
)
@pytest.mark.parametrize("cls", [num.ndarray, num.ma.MaskedArray])
def test_skipped_attributes(cls, attr):
    # Check that these special methods are either unimplemented or
    # explicitly implemented and are not using the fallback.
    if not hasattr(cls, attr):
        return

    # Check that if this is has _cupynumeric_metadata it is implemented
    obj = getattr(cls, attr)
    if not hasattr(obj, "_cupynumeric_metadata"):
        return

    meta = getattr(obj, "_cupynumeric_metadata")
    assert meta.implemented


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
