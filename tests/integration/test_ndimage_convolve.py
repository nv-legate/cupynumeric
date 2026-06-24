# Copyright 2026 NVIDIA Corporation
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

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage as scipy_ndimage
from utils.comparisons import allclose

import cupynumeric as num
from cupynumeric.ndimage.ndimage import _MODE_MAP
from cupynumeric.runtime import runtime


pytestmark = pytest.mark.skipif(
    runtime.num_gpus == 0,
    reason="cupynumeric.ndimage.convolve is only supported on GPU",
)


MODES = (
    "reflect",
    "constant",
    "nearest",
    "mirror",
    "wrap",
    "grid-mirror",
    "grid-constant",
    "grid-wrap",
)


def _make_input(shape: tuple[int, ...]) -> np.ndarray:
    return (np.arange(np.prod(shape), dtype=np.float64).reshape(shape) - 5) / 3


def _make_weights(shape: tuple[int, ...]) -> np.ndarray:
    return (np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1) / 7


def _check_convolve(
    input_np: np.ndarray,
    weights_np: np.ndarray,
    *,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
    use_output: bool = False,
) -> None:
    input_num = num.array(input_np)
    weights_num = num.array(weights_np)

    if use_output:
        output_num = num.empty_like(input_num)
        output_np = np.empty_like(input_np)

        result_num = num.ndimage.convolve(
            input_num,
            weights_num,
            output=output_num,
            mode=mode,
            cval=cval,
            origin=origin,
        )
        result_np = scipy_ndimage.convolve(
            input_np,
            weights_np,
            output=output_np,
            mode=mode,
            cval=cval,
            origin=origin,
        )

        assert result_num is output_num
        assert result_np is output_np
    else:
        result_num = num.ndimage.convolve(
            input_num, weights_num, mode=mode, cval=cval, origin=origin
        )
        result_np = scipy_ndimage.convolve(
            input_np, weights_np, mode=mode, cval=cval, origin=origin
        )

    assert allclose(result_num, result_np)


@pytest.mark.parametrize(
    "input_shape, weights_shape",
    [
        ((7,), (3,)),
        ((4, 6), (2, 3)),
        ((3, 4, 5), (2, 3, 2)),
        ((6, 10, 6), (2, 3, 3)),
    ],
    ids=str,
)
@pytest.mark.parametrize("use_output", (False, True), ids=("return", "output"))
def test_data_dimensionality_and_output(
    input_shape: tuple[int, ...],
    weights_shape: tuple[int, ...],
    use_output: bool,
) -> None:
    _check_convolve(
        _make_input(input_shape),
        _make_weights(weights_shape),
        use_output=use_output,
    )


def test_modes_match_mode_map() -> None:
    assert set(MODES) == set(_MODE_MAP)


@pytest.mark.parametrize(
    "input_shape, weights_shape",
    [((3,), (7,)), ((3,), (8,)), ((7, 7), (4, 4)), ((2, 3), (4, 5))],
    ids=str,
)
@pytest.mark.parametrize("mode", MODES)
def test_modes(
    mode: str, input_shape: tuple[int, ...], weights_shape: tuple[int, ...]
) -> None:
    cval = -2.5 if "constant" in mode else 0.0

    _check_convolve(
        _make_input(input_shape),
        _make_weights(weights_shape),
        mode=mode,
        cval=cval,
    )


@pytest.mark.parametrize("cval", (-3.5, 0.0, 2.25))
def test_constant_cval(cval: float) -> None:
    _check_convolve(
        _make_input((10, 8)), _make_weights((3, 2)), mode="constant", cval=cval
    )


@pytest.mark.parametrize(
    "input_shape, weights_shape, origin",
    [
        ((8, 10), (2, 3), 0),
        ((8, 10), (2, 3), -1),
        ((8, 10), (2, 3), (0, 1)),
        ((8, 10), (2, 3), (-1, 0)),
        ((8, 10), (3, 3), 0),
        ((8, 10), (3, 3), (-1, 1)),
        ((8, 10), (4, 2), (1, 0)),
        ((8, 10), (5, 4), (2, -1)),
        ((32, 16), (7, 3), (3, 1)),
    ],
    ids=str,
)
def test_origin(
    input_shape: tuple[int, ...],
    weights_shape: tuple[int, ...],
    origin: int | tuple[int, ...],
) -> None:
    _check_convolve(
        _make_input(input_shape),
        _make_weights(weights_shape),
        mode="nearest",
        origin=origin,
    )


@pytest.mark.parametrize("origin", [(0,), (0, 0, 0), 2, (-2, 0)], ids=str)
def test_invalid_origin(origin: int | tuple[int, ...]) -> None:
    input_num = num.array(_make_input((4, 5)))
    weights_num = num.array(_make_weights((2, 3)))

    with pytest.raises(ValueError):
        num.ndimage.convolve(input_num, weights_num, origin=origin)


def test_invalid_mode() -> None:
    input_num = num.array(_make_input((4, 5)))
    weights_num = num.array(_make_weights((2, 3)))

    with pytest.raises(ValueError, match=r"mode must be one of"):
        num.ndimage.convolve(input_num, weights_num, mode="bad-mode")


@pytest.mark.parametrize(
    "input_np, weights_np",
    [(np.array(1.0), _make_weights((2,))), (_make_input((4,)), np.array(1.0))],
    ids=("zero_dim_input", "zero_dim_weights"),
)
def test_zero_dimensional_rejected(
    input_np: np.ndarray, weights_np: np.ndarray
) -> None:
    with pytest.raises(ValueError, match=r"at least 1-D"):
        num.ndimage.convolve(num.array(input_np), num.array(weights_np))


def test_dimension_mismatch() -> None:
    input_num = num.array(_make_input((4, 5)))
    weights_num = num.array(_make_weights((3,)))

    with pytest.raises(ValueError, match=r"same dimensions"):
        num.ndimage.convolve(input_num, weights_num)


def test_invalid_output_shape() -> None:
    input_num = num.array(_make_input((4, 5)))
    weights_num = num.array(_make_weights((2, 3)))
    output = num.empty((4, 4), dtype=input_num.dtype)

    with pytest.raises(ValueError, match=r"output array shape"):
        num.ndimage.convolve(input_num, weights_num, output=output)


def test_invalid_output_dtype() -> None:
    input_np = _make_input((4, 5)).astype(np.float32)
    input_num = num.array(input_np)
    weights_num = num.array(_make_weights((2, 3)).astype(np.float32))
    output = num.empty(input_num.shape, dtype=np.float64)

    with pytest.raises(ValueError, match=r"output array dtype"):
        num.ndimage.convolve(input_num, weights_num, output=output)


def test_weights_cast_to_input_dtype() -> None:
    input_np = _make_input((4, 5)).astype(np.float32)
    weights_np = _make_weights((2, 3)).astype(np.float64)

    _check_convolve(input_np, weights_np, mode="constant", cval=1.25)


def test_axes_not_implemented() -> None:
    input_num = num.array(_make_input((4, 5)))
    weights_num = num.array(_make_weights((3, 3)))

    with pytest.raises(NotImplementedError):
        num.ndimage.convolve(input_num, weights_num, axes=(0,))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
