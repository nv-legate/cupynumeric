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

# cupynumeric.exp(*args: Any, out: Union[ndarray, None] = None,
# where: bool = True, casting: CastingKind = 'same_kind',
# order: str = 'K',
# dtype: Union[np.dtype[Any], None] = None, **kwargs: Any) → ndarray

DIM = 5
SIZES = [
    (0,),
    (1,),
    (DIM,),
    (0, 1),
    (1, 0),
    (1, 1),
    (1, DIM),
    (DIM, 1),
    (DIM, DIM),
    (1, 0, 0),
    (1, 1, 0),
    (1, 0, 1),
    (1, 1, 1),
    (DIM, 1, 1),
    (1, DIM, 1),
    (1, 1, DIM),
    (DIM, DIM, DIM),
]


@pytest.mark.parametrize("arr", ([], [[], []], [[[], []], [[], []]]))
def test_arr_empty(arr):
    res_num = num.exp(arr)
    res_np = np.exp(arr)
    assert np.array_equal(res_np, res_num)


def test_out_negative():
    in_shape = (2, 3)
    out_shape = (2, 4)
    arr_num = num.random.randint(1, 10, size=in_shape)
    arr_out = num.ones(shape=out_shape)
    with pytest.raises(ValueError):
        num.exp(arr_num, out=arr_out)


@pytest.mark.xfail
@pytest.mark.parametrize("casting", ("no", "equiv"))
def test_casting_negative(casting):
    in_shape = (2, 3)
    arr_num = num.random.randint(1, 10, size=in_shape)
    arr_np = np.array(arr_num)
    res_num = num.exp(arr_num, casting=casting)
    res_np = np.exp(arr_np, casting=casting)
    # cuPyNumeric run successfully.
    # Numpy raises " numpy.core._exceptions._UFuncInputCastingError:
    # Cannot cast ufunc 'exp' input from dtype('int64') to dtype('float64')
    # with casting rule 'no'
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("shape", SIZES)
def test_out(shape):
    arr_num = num.random.randint(1, 10, size=shape)
    num_out = num.ones(shape=shape)

    arr_np = np.array(arr_num)
    np_out = np.ones(shape=shape)

    num.exp(arr_num, out=num_out)
    np.exp(arr_np, out=np_out)
    assert np.array_equal(num_out, np_out)


@pytest.mark.xfail
def test_where_false():
    shape = (2, 3)
    arr_num = num.random.randint(1, 10, size=shape)
    num_out = num.ones(shape=shape)

    arr_np = np.array(arr_num)
    np_out = np.ones(shape=shape)
    # Numpy get the results.
    # cuPyNumeric raises "NotImplementedError:
    # the 'where' keyword is not yet supported"
    num.exp(arr_num, where=False, out=num_out)
    np.exp(arr_np, where=False, out=np_out)

    assert np.array_equal(num_out, np_out)


@pytest.mark.parametrize("shape", SIZES)
@pytest.mark.parametrize("order", ("K", "C", "F", "A"))
def test_order(shape, order):
    arr_num = num.random.randint(1, 10, size=shape)
    arr_np = np.array(arr_num)
    res_num = num.exp(arr_num, order=order)
    res_np = np.exp(arr_np, order=order)
    assert np.array_equal(res_np, res_num)


@pytest.mark.parametrize("casting", ("safe", "same_kind", "unsafe"))
def test_casting(casting):
    in_shape = (2, 3)
    arr_num = num.random.randint(1, 10, size=in_shape)
    arr_np = np.array(arr_num)
    res_num = num.exp(arr_num, casting=casting)
    res_np = np.exp(arr_np, casting=casting)
    assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
