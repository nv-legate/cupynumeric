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

import cupynumeric as num

DIM_CASE = (5, 5)

FLAGS = [
    "c_contiguous",
    "f_contiguous",
    "writeable",
    "aligned",
    "writebackifcopy",
    "fnc",
    "forc",
    "behaved",
    "carray",
    "farray",
]

ODSKIP = [pytest.param("owndata", marks=pytest.mark.skip)]
OSKIP = [pytest.param("O", marks=pytest.mark.skip)]

SHORT_FLAGS = ["C", "F", "W", "A", "X", "FNC", "FORC", "B", "CA", "FA"]

SETFLAGS_PARAMS = [
    ("write", "W", True),
    ("write", "W", False),
    ("align", "A", True),
    ("align", "A", False),
    # NumPy only allows setting "uic" to False
    ("uic", "X", False),
]


class Test_flags:
    @pytest.mark.parametrize("flag", FLAGS + ODSKIP)
    def test_default_attr(self, flag):
        npflags = np.zeros(shape=DIM_CASE).flags
        numflags = num.zeros(shape=DIM_CASE).flags

        assert getattr(npflags, flag) == getattr(numflags, flag)

    @pytest.mark.parametrize("flag", FLAGS + ODSKIP)
    def test_default_attr_view(self, flag):
        npflags = np.zeros(shape=DIM_CASE).view().flags
        numflags = num.zeros(shape=DIM_CASE).view().flags

        assert getattr(npflags, flag) == getattr(numflags, flag)

    @pytest.mark.parametrize("flag", SHORT_FLAGS + OSKIP)
    def test_default_item(self, flag):
        npflags = np.zeros(shape=DIM_CASE).flags
        numflags = num.zeros(shape=DIM_CASE).flags

        assert npflags[flag] == numflags[flag]

    @pytest.mark.parametrize("flag", SHORT_FLAGS + OSKIP)
    def test_default_item_view(self, flag):
        npflags = np.zeros(shape=DIM_CASE).view().flags
        numflags = num.zeros(shape=DIM_CASE).view().flags

        assert npflags[flag] == numflags[flag]

    @pytest.mark.parametrize("params", SETFLAGS_PARAMS)
    @pytest.mark.parametrize("libs", ((np, num), (num, np)))
    def test_setflags_num_to_np(self, params, libs):
        kwarg, flag, value = params
        lib1, lib2 = libs

        arr1 = lib1.zeros(shape=DIM_CASE)
        kwargs = {kwarg: value}
        arr1.setflags(**kwargs)
        # setting "align" has inconsistent behavior in NumPy
        # but at least check that it's accepted
        if kwarg != "align":
            assert arr1.flags[flag] == value

        arr2 = lib2.asarray(arr1)
        if kwarg != "align":
            assert arr2.flags[flag] == value


VIEW_CREATION_PARAMS = [
    ("view", ()),
    ("transpose", ()),
    pytest.param(("__getitem__", (slice(None),)), marks=pytest.mark.xfail),
    pytest.param(("reshape", ((2, 3),)), marks=pytest.mark.xfail),
    pytest.param(("squeeze", ()), marks=pytest.mark.xfail),
    pytest.param(("swapaxes", (0, 1)), marks=pytest.mark.xfail),
]


class Test_writeable:
    def test_non_writeable(self):
        arr = num.zeros(shape=DIM_CASE)
        arr.flags["W"] = False
        with pytest.raises(ValueError, match="not writeable"):
            arr[0, 0] = 12

    def test_cannot_make_nonwriteable_writeable(self):
        arr = num.zeros(shape=DIM_CASE)
        arr.flags["W"] = False
        with pytest.raises(ValueError, match="cannot be made writeable"):
            arr.flags["W"] = True

    def test_broadcast_result_nonwriteable(self):
        x = num.zeros((4,))
        x_bcasted = num.broadcast_to(x, (3, 4))
        assert not x_bcasted.flags["W"]

        y = num.zeros(shape=(3, 4))
        x_bcasted, y_bcasted = num.broadcast_arrays(x, y)
        assert not x_bcasted.flags["W"]

    @pytest.mark.parametrize("params", VIEW_CREATION_PARAMS)
    def test_views_inherit_writeable(self, params):
        method, args = params
        x = num.zeros((2, 1, 3))
        x.flags["W"] = False
        y = getattr(x, method)(*args)
        assert not y.flags["W"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
