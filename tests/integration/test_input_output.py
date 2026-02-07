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

import pickle
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
from numpy.lib import NumpyVersion
from utils.generators import mk_0to1_array, mk_seq_array

import cupynumeric as num


def test_ndarray_dumps():
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    dump_np = arr_np.dumps()
    dump_num = arr_num.dumps()
    assert np.allclose(pickle.loads(dump_np), arr_np)
    assert np.allclose(pickle.loads(dump_num), arr_num)


@pytest.mark.parametrize("shape", [(3, 2, 4), []], ids=str)
def test_ndarray_tolist(shape):
    arr_np = mk_0to1_array(np, shape)
    arr_num = mk_0to1_array(num, shape)
    assert arr_np.tolist() == arr_num.tolist()


def test_ndarray_tostring_returns_bytes() -> None:
    arr = mk_seq_array(num, (3, 2, 4))
    arr_np = arr.__array__()
    out = arr.tostring(order="C")
    assert isinstance(out, (bytes, bytearray))
    assert out == arr.tobytes(order="C")
    assert out == arr_np.tobytes(order="C")


def test_ndarray_setstate_smoke() -> None:
    arr = mk_seq_array(num, (3, 2))
    _reconstruct, _args, state = arr.__array__().__reduce__()
    arr.__setstate__(state)


@pytest.mark.skipif(
    NumpyVersion(np.__version__) >= "2.0.0", reason="tostring is deprecated"
)
@pytest.mark.parametrize("order", ["C", "F", "A"], ids=str)
def test_ndarray_tostring(order):
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    assert arr_np.tostring(order) == arr_num.tostring(order)


@pytest.mark.skipif(
    NumpyVersion(np.__version__) < "1.9.0",
    reason="tobytes is introduced in 1.9.0",
)
@pytest.mark.parametrize("order", ["C", "F", "A"], ids=str)
def test_ndarray_tobytes(order):
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    assert arr_np.tobytes(order) == arr_num.tobytes(order)


def test_ndarray_dump():
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    with NamedTemporaryFile() as fobj:
        arr_np.dump(file=fobj.name)
        assert np.allclose(pickle.load(fobj.file), arr_np)
    with NamedTemporaryFile() as fobj:
        arr_num.dump(file=fobj.name)
        assert np.allclose(pickle.load(fobj.file), arr_num)


def test_ndarray_tofile():
    shape = (3, 2, 4)
    arr_np = mk_seq_array(np, shape)
    arr_num = mk_seq_array(num, shape)
    with NamedTemporaryFile() as fobj_np, NamedTemporaryFile() as fobj_num:
        arr_np.tofile(fobj_np.name)
        arr_num.tofile(fobj_num.name)
        assert fobj_np.file.readlines() == fobj_num.file.readlines()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
