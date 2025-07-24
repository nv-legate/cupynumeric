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

SQUARE_CASES = [
    (10, 5, 2),
    (-1, 5, 2),
    (5, 2, 10),
    (5, 2, 5, 2),
    (10, 10, 1),
    (10, 1, 10),
    (1, 10, 10),
]


class TestSquare:
    anp = np.arange(100).reshape(10, 10)

    def test_basic(self):
        a = num.arange(100).reshape(10, 10)
        assert np.array_equal(self.anp, a)

    @pytest.mark.parametrize("order", ("C", "F", "A", None), ids=str)
    @pytest.mark.parametrize("shape", SQUARE_CASES, ids=str)
    def test_shape(self, shape, order):
        a = num.arange(100).reshape(10, 10)
        assert np.array_equal(
            num.reshape(a, shape, order=order),
            np.reshape(self.anp, shape, order=order),
        )

    def test_1d(self):
        a = num.arange(100).reshape(10, 10)
        assert np.array_equal(
            num.reshape(a, (100,)), np.reshape(self.anp, (100,))
        )

    def test_ravel(self):
        a = num.arange(100).reshape(10, 10)
        assert np.array_equal(num.ravel(a), np.ravel(self.anp))

        i = num.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        inp = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        )
        b = a[i, :]
        bnp = self.anp[inp, :]
        assert np.array_equal(b.ravel(), bnp.ravel())

        assert np.array_equal(b.reshape((0,)), bnp.reshape((0,)))

        a = num.full((3, 0), 1, dtype=int)
        anp = np.full((3, 0), 1, dtype=int)
        assert np.array_equal(num.ravel(a), np.ravel(anp))

        a = num.full((0, 3), 1, dtype=int)
        anp = np.full((0, 3), 1, dtype=int)
        assert np.array_equal(a.ravel(), anp.ravel())


RECT_CASES = (
    (10, 2, 10),
    (20, 10),
    (20, -5),
    (5, 40),
    (200, 1),
    (1, 200),
    (10, 20),
)


class TestRect:
    anp = np.random.rand(5, 4, 10)

    @pytest.mark.parametrize("order", ("C", "F", "A", None), ids=str)
    @pytest.mark.parametrize("shape", RECT_CASES, ids=str)
    def test_shape(self, shape, order):
        a = num.array(self.anp)
        assert np.array_equal(
            num.reshape(a, shape, order=order),
            np.reshape(self.anp, shape, order=order),
        )

    @pytest.mark.parametrize(
        "shape",
        (200, -1, -2, pytest.param(None, marks=pytest.mark.xfail)),
        ids=str,
    )
    def test_0d(self, shape):
        # for shape=None,
        # In Numpy, pass, returns the flattened 1-D array
        # In cuPyNumeric, raises TypeError: 'NoneType' object is not iterable
        a = num.array(self.anp)
        assert np.array_equal(
            num.reshape(a, shape), np.reshape(self.anp, shape)
        )

    def test_1d(self):
        a = num.array(self.anp)
        assert np.array_equal(
            num.reshape(a, (200,)), np.reshape(self.anp, (200,))
        )

    @pytest.mark.parametrize(
        "order",
        ("C", "F", "A", pytest.param("K", marks=pytest.mark.xfail), None),
        ids=str,
    )
    def test_ravel(self, order):
        # In Numpy, pass with 'K'
        # In cuPyNumeric, when order is 'K', raise ValueError:
        # order 'K' is not permitted for reshaping
        a = num.array(self.anp)
        assert np.array_equal(
            num.ravel(a, order=order), np.ravel(self.anp, order=order)
        )

    @pytest.mark.xfail
    def test_ravel_a_none(self):
        # In Numpy, pass and returns [None]
        # In cuPyNumeric, raises AttributeError:
        # 'NoneType' object has no attribute 'ravel'
        assert np.array_equal(num.ravel(None), np.ravel(None))


@pytest.mark.parametrize("shape", (0, (0,), (1, 0), (0, 1, 1)), ids=str)
def test_reshape_empty_array(shape):
    a = num.arange(0).reshape(0, 1)
    anp = np.arange(0).reshape(0, 1)
    assert np.array_equal(num.reshape(a, shape), np.reshape(anp, shape))


def test_reshape_same_shape():
    shape = (1, 2, 3)
    arr = np.random.rand(*shape)
    assert np.array_equal(np.reshape(arr, shape), num.reshape(arr, shape))


class TestReshapeErrors:
    def setup_method(self):
        self.a = num.arange(24)
        self.shape = (4, 3, 2)

    @pytest.mark.xfail
    def test_a_none(self):
        # In Numpy, it raises ValueError: cannot reshape array
        # In cuPyNumeric, it raises AttributeError:
        # 'NoneType' object has no attribute
        with pytest.raises(ValueError):
            num.reshape(None, self.shape)

    def test_empty_array_shape_invalid_size(self):
        a = num.arange(0).reshape(0, 1, 1)
        shape = (1, 1)
        with pytest.raises(ValueError):
            num.reshape(a, shape)

    @pytest.mark.parametrize(
        "shape",
        ((-1, 0, 2), (4, 3, 4), (4, 3, 0), (4, 3), (4,), (0,), 4),
        ids=str,
    )
    def test_shape_invalid_size(self, shape):
        msg = "cannot reshape array"
        with pytest.raises(ValueError, match=msg):
            num.reshape(self.a, shape)

    def test_shape_unknown_dimensions(self):
        shape = (-5, -1, 2)
        msg = "can only specify one unknown dimension"
        with pytest.raises(ValueError, match=msg):
            num.reshape(self.a, shape)

    @pytest.mark.parametrize("shape", ((4, 3, 2.0), 24.0), ids=str)
    def test_shape_float(self, shape):
        with pytest.raises(TypeError):
            num.reshape(self.a, shape)

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            num.reshape(self.a, self.shape, order="Z")

    def test_reshape_no_args(self):
        expected_exc = TypeError
        with pytest.raises(expected_exc):
            np.array(()).reshape()
        with pytest.raises(expected_exc):
            num.array(()).reshape()


class TestRavelErrors:
    def setup_method(self):
        self.a = num.arange(24).reshape(4, 3, 2)

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            num.ravel(self.a, order="Z")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
