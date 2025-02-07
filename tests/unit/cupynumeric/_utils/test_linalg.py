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

import cupynumeric._utils.linalg as m  # module under test


def _dot_modes_oracle(a_ndim: int, b_ndim: int) -> bool:
    a_modes, b_modes, out_modes = m.dot_modes(a_ndim, b_ndim)
    expr = f"{''.join(a_modes)},{''.join(b_modes)}->{''.join(out_modes)}"
    a = np.random.randint(100, size=3**a_ndim).reshape((3,) * a_ndim)
    b = np.random.randint(100, size=3**b_ndim).reshape((3,) * b_ndim)
    return np.array_equal(np.einsum(expr, a, b), np.dot(a, b))


@pytest.mark.parametrize(
    "a, b",
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
        (5, 1),
        (1, 5),
    ],
)
def test_dot_modes(a: int, b: int) -> None:
    assert _dot_modes_oracle(a, b)


def _inner_modes_oracle(a_ndim: int, b_ndim: int) -> bool:
    a_modes, b_modes, out_modes = m.inner_modes(a_ndim, b_ndim)
    expr = f"{''.join(a_modes)},{''.join(b_modes)}->{''.join(out_modes)}"
    a = np.random.randint(100, size=3**a_ndim).reshape((3,) * a_ndim)
    b = np.random.randint(100, size=3**b_ndim).reshape((3,) * b_ndim)
    return np.array_equal(np.einsum(expr, a, b), np.inner(a, b))


@pytest.mark.parametrize(
    "a, b",
    [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
        (5, 1),
        (1, 5),
    ],
)
def test_inner_modes(a: int, b: int) -> None:
    assert _inner_modes_oracle(a, b)


@pytest.mark.parametrize("a, b", [(0, 0), (0, 1), (1, 0)])
def test_matmul_modes_bad(a: int, b: int) -> None:
    with pytest.raises(ValueError):
        m.matmul_modes(a, b)


def _matmul_modes_oracle(a_ndim: int, b_ndim: int) -> bool:
    a_modes, b_modes, out_modes = m.matmul_modes(a_ndim, b_ndim)
    expr = f"{''.join(a_modes)},{''.join(b_modes)}->{''.join(out_modes)}"
    a = np.random.randint(100, size=3**a_ndim).reshape((3,) * a_ndim)
    b = np.random.randint(100, size=3**b_ndim).reshape((3,) * b_ndim)
    return np.array_equal(np.einsum(expr, a, b), np.matmul(a, b))


@pytest.mark.parametrize(
    "a, b", [(1, 1), (2, 1), (1, 2), (2, 2), (5, 1), (1, 5)]
)
def test_matmul_modes(a: int, b: int) -> None:
    assert _matmul_modes_oracle(a, b)


AxesType = int | tuple[int, int] | tuple[list[int], list[int]]


def _tensordot_modes_oracle(a_ndim: int, b_ndim: int, axes: AxesType) -> bool:
    a_modes, b_modes, out_modes = m.tensordot_modes(a_ndim, b_ndim, axes)
    expr = f"{''.join(a_modes)},{''.join(b_modes)}->{''.join(out_modes)}"
    a = np.random.randint(100, size=3**a_ndim).reshape((3,) * a_ndim)
    b = np.random.randint(100, size=3**b_ndim).reshape((3,) * b_ndim)
    return np.array_equal(np.einsum(expr, a, b), np.tensordot(a, b, axes))


class Test_tensordot_modes:
    @pytest.mark.parametrize(
        "a_ndim, b_ndim, axes", [(1, 3, 2), (3, 1, 2), (1, 1, 2)]
    )
    def test_bad_single_axis(self, a_ndim, b_ndim, axes) -> None:
        with pytest.raises(ValueError):
            m.tensordot_modes(a_ndim, b_ndim, axes)

    def test_bad_axes_length(self) -> None:
        with pytest.raises(ValueError):
            # len(a_axes) > a_ndim
            m.tensordot_modes(1, 3, [(1, 2), (1, 2)])

        with pytest.raises(ValueError):
            # len(b_axes) > b_ndim
            m.tensordot_modes(3, 1, [(1, 2), (1, 2)])

        with pytest.raises(ValueError):
            # len(a_axes) != len(b_axes)
            m.tensordot_modes(2, 3, ([0], [0, 1]))

    def test_bad_negative_axes(self) -> None:
        with pytest.raises(ValueError):
            # any(ax < 0 for ax in a_axes)
            m.tensordot_modes(3, 2, [(1, -1), (1, 2)])

        with pytest.raises(ValueError):
            # any(ax < 0 for ax in b_axes)
            m.tensordot_modes(3, 2, [(1, 2), (1, -1)])

    def test_bad_mismatched_axes(self) -> None:
        with pytest.raises(ValueError):
            # len(a_axes) != len(set(a_axes))
            m.tensordot_modes(4, 4, [(1, 1, 2), (1, 3, 2)])

        with pytest.raises(ValueError):
            # len(b_axes) != len(set(b_axes))
            m.tensordot_modes(4, 4, [(1, 3, 2), (1, 1, 2)])

    def test_bad_axes_oob(self) -> None:
        with pytest.raises(ValueError):
            # any(ax >= a_ndim for ax in a_axes)
            m.tensordot_modes(1, 2, [(1, 3), (1, 2)])

        with pytest.raises(ValueError):
            # any(ax >= b_ndim for ax in b_axes)
            m.tensordot_modes(2, 1, [(1, 2), (1, 3)])

    @pytest.mark.parametrize("a, b, axes", [(0, 0, 0), (2, 2, 1)])
    def test_single_axis(self, a: int, b: int, axes: AxesType):
        assert _tensordot_modes_oracle(a, b, axes)

    @pytest.mark.parametrize(
        "a, b, axes",
        [(2, 2, (1, 0)), (2, 2, (0, 1)), (2, 2, (1, 1)), (2, 2, (-1, 0))],
    )
    def test_tuple_axis(self, a: int, b: int, axes: AxesType):
        assert _tensordot_modes_oracle(a, b, axes)

    @pytest.mark.parametrize(
        "a, b, axes",
        [
            (2, 2, ([1], [0])),
            (2, 2, ([0], [1])),
            (2, 2, ([1], [1])),
            (2, 2, ([1, 0], [0, 1])),
        ],
    )
    def test_explicit_axis(self, a: int, b: int, axes: AxesType):
        assert _tensordot_modes_oracle(a, b, axes)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
