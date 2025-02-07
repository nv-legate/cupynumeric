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
from utils.utils import check_module_function

import cupynumeric as num


def _deepen(depth, x):
    for _ in range(depth):
        x = [x]
    return x


DIM = 10

SIZE_CASES = [
    [(0,), (0,)],  # empty arrays
    [(1,), (1,)],  # singlton arrays
    [(0,), (10,)],  # empty and scalars
    [(DIM, 1), (DIM, DIM)],  # 1D and 2D arrays
    [(DIM, 1), (DIM, 1), (DIM, DIM)],  # 3 arrays in the inner-most list
]


@pytest.mark.parametrize("sizes", SIZE_CASES, ids=str)
@pytest.mark.parametrize("depth", range(3))
def test(depth, sizes):
    a = [np.arange(np.prod(size)).reshape(size) for size in sizes]
    b = [np.arange(np.prod(size)).reshape(size) for size in sizes]

    print_msg = (
        f"depth={depth}, np.block([{_deepen(depth, a)}, "
        f"{_deepen(depth, b)}])"
    )
    arg = [_deepen(depth, a), _deepen(depth, b)]
    check_module_function("block", [arg], {}, print_msg, check_type=False)


class TestBlock:
    def test_block_simple_row_wise(self):
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        arg = [a_2d, b_2d]

        print_msg = (
            f"np & cupynumeric.block([array({a_2d.shape}), "
            f"array({b_2d.shape})])"
        )
        check_module_function("block", [arg], {}, print_msg)

    def test_block_simple_column_wise(self):
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        arg = [[a_2d], [b_2d]]

        print_msg = (
            f"np & cupynumeric.block([[array({a_2d.shape})], "
            f"[array({b_2d.shape})]])"
        )
        check_module_function("block", [arg], {}, print_msg)

    def test_block_with_1d_arrays_multiple_rows(self):
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        arg = [[a, b], [a, b]]

        print_msg = (
            f"np & cupynumeric.block([[array({a.shape}), array({b.shape})], "
            f"[array({a.shape}), array({b.shape})]])"
        )
        check_module_function("block", [arg], {}, print_msg, check_type=False)

    def test_block_mixed_1d_and_2d(self):
        a_2d = np.ones((2, 2))
        b_1d = np.array([2, 2])
        arg = [[a_2d], [b_1d]]

        print_msg = (
            f"np & cupynumeric.block([[array({a_2d.shape})], "
            f"[array({b_1d.shape})]])"
        )
        check_module_function("block", [arg], {}, print_msg)

    def test_block_complicated(self):
        one_2d = np.array([[1, 1, 1]])
        two_2d = np.array([[2, 2, 2]])
        three_2d = np.array([[3, 3, 3, 3, 3, 3]])
        four_1d = np.array([4, 4, 4, 4, 4, 4])
        five_1d = np.array([5])
        six_1d = np.array([6, 6, 6, 6, 6])
        zero_2d = np.zeros((2, 6))
        arg = [
            [one_2d, two_2d],
            [three_2d],
            [four_1d],
            [five_1d, six_1d],
            [zero_2d],
        ]

        print_msg = "np & cupynumeric.block()"
        check_module_function("block", [arg], {}, print_msg)

    def test_nested(self):
        one = np.array([1, 1, 1])
        two = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        three = np.array([3, 3, 3])
        four = np.array([4, 4, 4])
        five = np.array([5])
        six = np.array([6, 6, 6, 6, 6])
        zero = np.zeros((2, 6))

        result = num.block(
            [[num.block([[one], [three], [four]]), two], [five, six], [zero]]
        )
        expected = np.array(
            [
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 2, 2, 2],
                [4, 4, 4, 2, 2, 2],
                [5, 6, 6, 6, 6, 6],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        assert np.array_equal(result, expected)

    def test_3d(self):
        a000 = np.ones((2, 2, 2), int) * 1

        a100 = np.ones((3, 2, 2), int) * 2
        a010 = np.ones((2, 3, 2), int) * 3
        a001 = np.ones((2, 2, 3), int) * 4

        a011 = np.ones((2, 3, 3), int) * 5
        a101 = np.ones((3, 2, 3), int) * 6
        a110 = np.ones((3, 3, 2), int) * 7

        a111 = np.ones((3, 3, 3), int) * 8

        arg = [
            [
                [a000, a001],
                [a010, a011],
            ],
            [
                [a100, a101],
                [a110, a111],
            ],
        ]

        print_msg = "np & cupynumeric.block()"
        check_module_function("block", [arg], {}, print_msg, check_type=False)


class TestBlockErrors:
    def test_mismatched_shape_1(self):
        msg = "All arguments to block must have the same number of dimensions"
        a = np.array([0, 0])
        b = np.eye(2)
        with pytest.raises(ValueError, match=msg):
            num.block([a, b])
        with pytest.raises(ValueError, match=msg):
            num.block([b, a])

    def test_mismatched_shape_2(self):
        a = np.array([[0, 0]])
        b = np.eye(2)
        with pytest.raises(ValueError):
            num.block([a, b])

    @pytest.mark.xfail
    def test_mismatched_shape_3(self):
        a = np.array([[0, 0]])
        b = np.eye(2)

        # numpy: raises ValueError
        # cumunerics: pass, output is [[1, 0, 0, 0]
        #                              [0, 1, 0, 0]]
        with pytest.raises(ValueError):
            num.block([b, a])

    def test_no_lists(self):
        # numpy: pass, output is np.array(1)
        # cupynumeric: raises TypeError, cupynumeric doesn't support 0-D array
        # assert np.array_equal(num.block(1), np.array(1))

        # numpy: pass, output is np.eye(3)
        # cupynumeric: pass, output is 1-D array: [1, 0, 0, 0, 1, 0, 0, 0, 1]
        # assert np.array_equal(num.block(np.eye(3)), np.eye(3))
        np.array_equal(num.block(num.eye(3)), [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_invalid_nesting(self):
        msg = "List depths are mismatched"
        with pytest.raises(ValueError, match=msg):
            num.block([1, [2]])

        with pytest.raises(ValueError, match=msg):
            num.block([[1], 2])

        msg = "cannot be empty"
        with pytest.raises(ValueError, match=msg):
            num.block([1, []])

        with pytest.raises(ValueError, match=msg):
            num.block([[], 2])

        msg = "List depths are mismatched"
        with pytest.raises(ValueError, match=msg):
            num.block([[[1], [2]], [[3, 4]], [5]])

    @pytest.mark.parametrize("input", ([], [[]], [[1], []]))
    def test_empty_lists(self, input):
        msg = r"cannot be empty"
        with pytest.raises(ValueError, match=msg):
            num.block(input)

    def test_tuple(self):
        # numpy: raises TypeError below:
        # TypeError: arrays is a tuple. Only lists can be used
        # to arrange blocks,and np.block does not allow implicit
        # conversion from tuple to ndarray.
        # cupynumeric: pass
        np.array_equal(num.block(([1, 2], [3, 4])), [1, 2, 3, 4])
        np.array_equal(num.block([(1, 2), (3, 4)]), [1, 2, 3, 4])

    def test_different_ndims(self):
        msg = "All arguments to block must have the same number of dimensions"
        a = 1.0
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 1, 3))

        # numpy: pass, output is np.array([[[1., 2., 2., 3., 3., 3.]]])
        # cupynumeric: raises ValueError
        with pytest.raises(ValueError, match=msg):
            num.block([a, b, c])

    def test_different_ndims_depths(self):
        msg = "All arguments to block must have the same number of dimensions"
        a = 1.0
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 2, 3))

        # numpy: pass,output is np.array([[[1., 2., 2.],
        #                       [3., 3., 3.],
        #                       [3., 3., 3.]]])
        # cupynumeric: raises ValueError
        with pytest.raises(ValueError, match=msg):
            num.block([[a, b], [c]])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
