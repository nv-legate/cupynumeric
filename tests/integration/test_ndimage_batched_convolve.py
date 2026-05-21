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
from cupynumeric.runtime import runtime


pytestmark = pytest.mark.skipif(
    runtime.num_gpus == 0,
    reason="cupynumeric.ndimage.batched_convolve is only supported on GPU",
)


def _make_input(
    shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.float64)
) -> np.ndarray:
    values = (
        np.arange(np.prod(shape), dtype=np.float64).reshape(shape) - 5
    ) / 3
    return values.astype(dtype)


def _make_weights(
    shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.float64)
) -> np.ndarray:
    values = (
        np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1
    ) / 7
    return values.astype(dtype)


def _expected_batched_convolve(
    input_np: np.ndarray,
    weights_np: np.ndarray,
    *,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int | tuple[int, ...] = 0,
) -> np.ndarray:
    expected = np.empty(
        (input_np.shape[0], weights_np.shape[0], *input_np.shape[1:]),
        dtype=input_np.dtype,
    )
    for input_batch in range(input_np.shape[0]):
        for filter_batch in range(weights_np.shape[0]):
            expected[input_batch, filter_batch] = scipy_ndimage.convolve(
                input_np[input_batch],
                weights_np[filter_batch].astype(input_np.dtype, copy=False),
                mode=mode,
                cval=cval,
                origin=origin,
            )
    return expected


def _check_batched_convolve(
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
    expected = _expected_batched_convolve(
        input_np, weights_np, mode=mode, cval=cval, origin=origin
    )

    if use_output:
        output_num = num.empty(expected.shape, dtype=input_num.dtype)
        result_num = num.ndimage.batched_convolve(
            input_num,
            weights_num,
            output=output_num,
            mode=mode,
            cval=cval,
            origin=origin,
        )
        assert result_num is output_num
    else:
        result_num = num.ndimage.batched_convolve(
            input_num, weights_num, mode=mode, cval=cval, origin=origin
        )

    assert allclose(result_num, expected)


@pytest.mark.parametrize(
    "input_shape, weights_shape",
    [((2, 7), (3, 3)), ((3, 4, 5), (2, 2, 3)), ((4, 5, 5, 10), (3, 2, 3, 2))],
    ids=str,
)
@pytest.mark.parametrize("use_output", (False, True), ids=("return", "output"))
def test_data_dimensionality_and_output(
    input_shape: tuple[int, ...],
    weights_shape: tuple[int, ...],
    use_output: bool,
) -> None:
    _check_batched_convolve(
        _make_input(input_shape),
        _make_weights(weights_shape),
        use_output=use_output,
    )


@pytest.mark.parametrize(
    "mode",
    ("reflect", "constant", "nearest", "mirror", "wrap", "grid-constant"),
)
@pytest.mark.parametrize(
    "input_shape,weights_shape",
    [
        ((2, 6), (4, 3)),
        ((2, 8), (3, 15)),
        ((3, 4, 5), (2, 2, 3)),
        ((4, 5, 5, 10), (3, 2, 2, 2)),
        ((2, 8, 10), (3, 2, 3)),
    ],
    ids=lambda x: str(x),
)
def test_modes(mode: str, input_shape, weights_shape) -> None:
    cval = -2.5 if "constant" in mode else 0.0
    _check_batched_convolve(
        _make_input(input_shape),
        _make_weights(weights_shape),
        mode=mode,
        cval=cval,
    )


@pytest.mark.parametrize(
    "input_shape, weights_shape, origin",
    [
        ((2, 9, 12), (3, 2, 3), 0),
        ((2, 7, 11), (3, 2, 3), -1),
        ((2, 9, 12), (3, 2, 3), (0, 1)),
        ((2, 7, 11), (3, 2, 3), (-1, 0)),
        ((2, 9, 11), (3, 3, 3), (-1, 1)),
        ((2, 8, 13), (3, 4, 2), (1, 0)),
        ((2, 10, 9), (3, 5, 4), (2, -1)),
        ((2, 32, 16), (3, 7, 3), (3, 1)),
        ((4, 12, 12, 6), (2, 5, 5, 2), (0, 2, -1)),
        ((2, 8, 10), (3, 8, 8), (1, 2)),
        ((2, 16, 16), (3, 2, 11), (0, 5)),
        ((5, 24, 17, 8), (2, 3, 3, 4), (1, 1, 0)),
        ((2, 17, 9, 5), (2, 7, 5, 4), (0, 2, -2)),
    ],
    ids=str,
)
def test_origin(
    input_shape: tuple[int, ...],
    weights_shape: tuple[int, ...],
    origin: int | tuple[int, ...],
) -> None:
    _check_batched_convolve(
        _make_input(input_shape),
        _make_weights(weights_shape),
        mode="nearest",
        origin=origin,
    )


def test_weights_cast_to_input_dtype() -> None:
    _check_batched_convolve(
        _make_input((2, 7), np.dtype(np.float32)),
        _make_weights((3, 3), np.dtype(np.float64)),
        mode="constant",
        cval=1.25,
    )


@pytest.mark.parametrize(
    "input_shape, weights_shape",
    [((4,), (2, 3)), ((2, 4), (3,)), ((2, 4), (3, 2, 2))],
    ids=str,
)
def test_invalid_shapes(
    input_shape: tuple[int, ...], weights_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        num.ndimage.batched_convolve(
            num.array(_make_input(input_shape)),
            num.array(_make_weights(weights_shape)),
        )


def test_invalid_output() -> None:
    input_num = num.array(_make_input((2, 4)))
    weights_num = num.array(_make_weights((3, 2)))

    with pytest.raises(ValueError):
        num.ndimage.batched_convolve(
            input_num, weights_num, output=num.empty((2, 4))
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
