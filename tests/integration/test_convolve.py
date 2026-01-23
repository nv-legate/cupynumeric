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
import os

import numpy as np
import pytest
import scipy.signal as sig
from utils.comparisons import allclose

import cupynumeric as num
from cupynumeric.runtime import runtime

CUDA_TEST = runtime.num_gpus > 0
MULTI_GPU = runtime.num_gpus > 1
EAGER_TEST = os.environ.get("CUPYNUMERIC_FORCE_THUNK", None) == "eager"

SHAPES = [(100,), (10, 10), (10, 10, 10), (32, 2, 32)]
FILTER_SHAPES = [(5,), (3, 5), (3, 5, 3), (32, 1, 32)]

LARGE_SHAPES = [
    pytest.param(
        (128, 2, 1024),
        (64, 2, 512),
        marks=pytest.mark.xfail(
            not CUDA_TEST, run=False, reason="test hang on CPU variants"
        ),
    ),
    pytest.param(
        (1024, 2, 4096),
        (128, 16, 64),
        marks=pytest.mark.xfail(
            not CUDA_TEST, run=False, reason="test hang on CPU variants"
        ),
    ),
    pytest.param(
        (1024, 2, 1024),
        (5, 1, 5),
        marks=pytest.mark.xfail(
            CUDA_TEST, run=False, reason="GPU variant hits SIGABRT"
        ),
    ),
    pytest.param(
        (1024, 2, 1024),
        (128, 1, 128),
        marks=pytest.mark.xfail(
            run=False, reason="GPU variant hits SIGFPE, CPU hangs"
        ),
    ),
]

SPECIAL_SHAPES = [
    pytest.param((262144,), (16384,)),
    pytest.param((8, 8, 8), (65, 65, 65)),
    pytest.param((100, 100, 100), (7, 9, 11)),
    pytest.param(
        (200, 200, 200),
        (7, 15, 15),
        marks=pytest.mark.xfail(
            MULTI_GPU, run=False, reason="test fails on multi-GPU in pipeline"
        ),
    ),
]


DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
]


@pytest.mark.xfail
def test_none():
    expected_exc = TypeError
    with pytest.raises(expected_exc):
        num.convolve(None, None, mode="same")
        # cuPyNumeric raises AttributeError
    with pytest.raises(expected_exc):
        np.convolve(None, None, mode="same")


def test_empty():
    expected_exc = ValueError
    with pytest.raises(expected_exc):
        num.convolve([], [], mode="same")
    with pytest.raises(expected_exc):
        np.convolve([], [], mode="same")


@pytest.mark.skipif(not EAGER_TEST, reason="warning is only emitted in eager")
def test_convolve_warns_direct_in_1d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUPYNUMERIC_WARN", "1")
    msg = r"the method direct is ignored for the 1D convolution"
    a = num.array([1, 2, 3])
    v = num.array([0, 1, 0])
    with pytest.warns(UserWarning, match=msg):
        num.convolve(a, v, mode="same", method="direct")


@pytest.mark.skipif(EAGER_TEST, reason="deferred/cuda behavior")
def test_convolve_direct_ignored_matches_default() -> None:
    a = num.array([1, 2, 3])
    v = num.array([0, 1, 0])
    baseline = num.convolve(a, v, mode="same")
    out = num.convolve(a, v, mode="same", method="direct")
    assert allclose(out, baseline)


def test_diff_dims():
    shape1 = (5,) * 3
    shape2 = (5,) * 2
    arr1 = num.random.random(shape1)
    arr2 = num.random.random(shape2)
    expected_exc = RuntimeError
    with pytest.raises(expected_exc):
        num.convolve(arr1, arr2, mode="same")
    with pytest.raises(expected_exc):
        np.convolve(arr1, arr2, mode="same")


def check_convolve(a, v):
    anp = a.__array__()
    vnp = v.__array__()

    out = num.convolve(a, v, mode="same")
    if a.ndim > 1:
        out_np = sig.convolve(anp, vnp, mode="same")
    else:
        out_np = np.convolve(anp, vnp, mode="same")

    assert allclose(out, out_np)


@pytest.mark.parametrize(
    "shape, filter_shape",
    list(zip(SHAPES, FILTER_SHAPES)) + LARGE_SHAPES + SPECIAL_SHAPES,
    ids=str,
)
def test_int(shape: tuple[int, ...], filter_shape: tuple[int, ...]) -> None:
    a = num.random.randint(0, 5, shape)
    v = num.random.randint(0, 5, filter_shape)

    check_convolve(a, v)


@pytest.mark.parametrize(
    "shape, filter_shape",
    list(zip(SHAPES, FILTER_SHAPES)) + LARGE_SHAPES,
    ids=str,
)
def test_double(shape: tuple[int, ...], filter_shape: tuple[int, ...]) -> None:
    a = num.random.rand(*shape)
    v = num.random.rand(*filter_shape)

    check_convolve(a, v)
    check_convolve(v, a)


@pytest.mark.parametrize(
    "shape, filter_shape",
    list(zip(SHAPES, FILTER_SHAPES)) + LARGE_SHAPES,
    ids=str,
)
def test_float(shape: tuple[int, ...], filter_shape: tuple[int, ...]) -> None:
    a = num.random.rand(*shape).astype(num.float32)
    v = num.random.rand(*filter_shape).astype(num.float32)

    check_convolve(a, v)
    check_convolve(v, a)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_dtype(dtype):
    shape = (5,) * 2
    arr1 = num.random.randint(0, 5, shape, dtype=np.int64)
    arr2 = num.random.random(shape).astype(dtype)
    out_num = num.convolve(arr1, arr2, mode="same")
    out_np = np.convolve(arr1, arr2, mode="same")
    assert allclose(out_num, out_np)


@pytest.mark.parametrize(
    "mode",
    [
        ("same"),
        pytest.param("valid", marks=pytest.mark.xfail),
        pytest.param("full", marks=pytest.mark.xfail),
    ],
)
def test_modes(mode):
    shape = (5,) * 2
    arr1 = num.random.random(shape)
    arr2 = num.random.random(shape)
    out_num = num.convolve(arr1, arr2, mode=mode)
    # when mode!="same", cupynumeric raises
    # NotImplementedError: Need to implement other convolution modes
    out_np = np.convolve(arr1, arr2, mode=mode)
    assert allclose(out_num, out_np)


@pytest.mark.parametrize(
    "ndim",
    [
        (0),
        (1),
        (2),
        (3),
        pytest.param(4, marks=pytest.mark.xfail),
        pytest.param(5, marks=pytest.mark.xfail),
    ],
)
def test_ndim(ndim):
    shape = (5,) * ndim
    arr1 = num.random.random(shape)
    arr2 = num.random.random(shape)
    out_num = num.convolve(arr1, arr2, mode="same")
    # cupynumeric raises NotImplementedError: 4-D arrays are not yet supported
    out_np = np.convolve(arr1, arr2, mode="same")
    assert allclose(out_num, out_np)


@pytest.mark.parametrize("method", ("auto", "direct", "fft"))
def test_methods(method):
    shape = (5,) * 2
    arr1 = num.random.random(shape)
    arr2 = num.random.random(shape)
    out_num = num.convolve(arr1, arr2, mode="same", method=method)
    out_np = sig.convolve(arr1, arr2, mode="same", method=method)
    assert allclose(out_num, out_np)


def test_invalid_method():
    with pytest.raises(ValueError):
        num.convolve([], [], mode="same", method="test")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
