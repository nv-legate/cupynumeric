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

import sys

import legate.core  # noqa: F401 — ensure Legate runtime is loaded first

import numpy as np
import pytest
from legate.core import TaskTarget, get_legate_runtime

import cupynumeric as num
from cupynumeric.config import CuPyNumericOpCode
from cupynumeric.runtime import runtime as cpn_runtime
from cupynumeric.settings import settings


def _get_store(arr):
    thunk = arr._thunk
    if not hasattr(thunk, "base"):
        thunk = thunk.to_deferred_array(read_only=False)
    if thunk.base.has_scalar_storage:
        thunk = thunk._convert_future_to_regionfield()
    return thunk.base


def _to_deferred(arr):
    thunk = arr._thunk
    if not hasattr(thunk, "base"):
        thunk = thunk.to_deferred_array(read_only=False)
    return thunk


def _run_all2all_stores(source_store, index_store, output_store):
    legate_runtime = get_legate_runtime()
    library = cpn_runtime.library

    task = legate_runtime.create_auto_task(library, CuPyNumericOpCode.ALL2ALL)
    task.add_input(source_store)
    task.add_input(index_store)
    task.add_output(output_store)
    task.add_alignment(output_store, index_store)
    task.add_nccl_communicator()
    task.execute()

    legate_runtime.issue_execution_fence(block=True)


def _run_all2all_1d(source, indices, output):
    out_thunk = _to_deferred(output)
    converted = out_thunk.base.has_scalar_storage
    if converted:
        out_thunk = out_thunk._convert_future_to_regionfield()
    _run_all2all_stores(
        _get_store(source), _get_store(indices), out_thunk.base
    )
    if converted:
        _to_deferred(output).copy(out_thunk, deep=True)


def _build_point_index(source, *coord_arrays):
    src_thunk = _to_deferred(source)
    coord_thunks = tuple(_to_deferred(c) for c in coord_arrays)
    return src_thunk._zip_indices(
        0, coord_thunks, check_bounds=settings.bounds_check_enabled("indexing")
    )


def _run_all2all_nd(source, point_index_thunk, output):
    _run_all2all_stores(
        _get_store(source), point_index_thunk.base, _get_store(output)
    )


_gpu_only = pytest.mark.skipif(
    get_legate_runtime().machine.count(TaskTarget.GPU) <= 1,
    reason="All2All is GPU-only",
)


@pytest.fixture(autouse=True)
def enable_nccl_gather():
    settings.use_nccl_gather = True
    yield
    settings.use_nccl_gather.unset_value()


@_gpu_only
class TestAll2All1DSource:
    @pytest.mark.parametrize("dtype", [num.float32, num.float64])
    def test_basic_floating(self, dtype) -> None:
        source = num.arange(10, dtype=dtype)
        indices = num.array([0, 2, 5, 9], dtype=num.int64)
        output = num.empty(4, dtype=dtype)
        _run_all2all_1d(source, indices, output)
        expected = np.array([0.0, 2.0, 5.0, 9.0], dtype=dtype)
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    @pytest.mark.parametrize("dtype", [num.int32, num.int64])
    def test_basic_integer(self, dtype) -> None:
        source = num.arange(8, dtype=dtype)
        indices = num.array([1, 3, 7], dtype=num.int64)
        output = num.empty(3, dtype=dtype)
        _run_all2all_1d(source, indices, output)
        np.testing.assert_array_equal(np.asarray(output), [1, 3, 7])

    def test_duplicate_indices(self) -> None:
        source = num.array([10, 20, 30, 40, 50], dtype=num.float64)
        indices = num.array([2, 2, 0, 0, 4], dtype=num.int64)
        output = num.empty(5, dtype=num.float64)
        _run_all2all_1d(source, indices, output)
        expected = np.array([30.0, 30.0, 10.0, 10.0, 50.0])
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_reversed(self) -> None:
        source = num.arange(5, dtype=num.int64)
        indices = num.array([4, 3, 2, 1, 0], dtype=num.int64)
        output = num.empty(5, dtype=num.int64)
        _run_all2all_1d(source, indices, output)
        np.testing.assert_array_equal(np.asarray(output), [4, 3, 2, 1, 0])

    def test_single_element(self) -> None:
        source = num.arange(100, dtype=num.float32)
        indices = num.array([42], dtype=num.int64)
        output = num.empty(1, dtype=num.float32)
        _run_all2all_1d(source, indices, output)
        np.testing.assert_array_equal(np.asarray(output), [42.0])

    def test_2d_index_grid(self) -> None:
        source = num.arange(12, dtype=num.float32)
        indices = num.array([[0, 3, 6], [9, 11, 1]], dtype=num.int64)
        output = num.empty((2, 3), dtype=num.float32)
        _run_all2all_1d(source, indices, output)
        expected = np.array([[0, 3, 6], [9, 11, 1]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    def test_3d_index_grid(self) -> None:
        source = num.arange(24, dtype=num.float32)
        indices = num.array(
            [[[0, 1, 2], [3, 4, 5]], [[23, 22, 21], [20, 19, 18]]],
            dtype=num.int64,
        )
        output = num.empty((2, 2, 3), dtype=num.float32)
        _run_all2all_1d(source, indices, output)
        expected = np.array(
            [[[0, 1, 2], [3, 4, 5]], [[23, 22, 21], [20, 19, 18]]],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    def test_random(self) -> None:
        n = 10_000
        source = num.arange(n, dtype=num.float32)
        rng = np.random.default_rng(42)
        idx_np = rng.integers(0, n, size=2048)
        indices = num.array(idx_np, dtype=num.int64)
        output = num.empty(2048, dtype=num.float32)
        _run_all2all_1d(source, indices, output)
        expected = np.arange(n, dtype=np.float32)[idx_np]
        np.testing.assert_array_equal(np.asarray(output), expected)


@_gpu_only
class TestAll2All2DSource:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_floating(self, dtype) -> None:
        src_np = np.arange(20, dtype=dtype).reshape(4, 5)
        rows_np = np.array([0, 1, 3, 2])
        cols_np = np.array([4, 0, 2, 1])

        source = num.array(src_np)
        rows = num.array(rows_np, dtype=num.int64)
        cols = num.array(cols_np, dtype=num.int64)

        point_idx = _build_point_index(source, rows, cols)
        output = num.empty(point_idx.shape, dtype=dtype)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    def test_basic_int32(self) -> None:
        src_np = np.arange(12, dtype=np.int32).reshape(3, 4)
        rows_np = np.array([0, 2, 1])
        cols_np = np.array([3, 1, 0])

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.int32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_duplicate_indices(self) -> None:
        src_np = np.arange(9, dtype=np.float64).reshape(3, 3)
        rows_np = np.array([0, 0, 1, 1, 2, 2])
        cols_np = np.array([0, 0, 1, 1, 2, 2])

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float64)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_2d_output(self) -> None:
        src_np = np.arange(30, dtype=np.float32).reshape(5, 6)
        rows_np = np.array([[0], [2], [4]])
        cols_np = np.array([1, 3, 5])

        rows_bc, cols_bc = np.broadcast_arrays(rows_np, cols_np)
        rows_bc = np.ascontiguousarray(rows_bc)
        cols_bc = np.ascontiguousarray(cols_bc)

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(rows_bc, dtype=num.int64),
            num.array(cols_bc, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_random(self) -> None:
        rng = np.random.default_rng(123)
        src_np = rng.random((500, 500)).astype(np.float32)
        n_idx = 10_000
        rows_np = rng.integers(0, 500, size=n_idx)
        cols_np = rng.integers(0, 500, size=n_idx)

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_almost_equal(np.asarray(output), expected)


@_gpu_only
class TestAll2All3DSource:
    def test_float32(self) -> None:
        src_np = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        d0 = np.array([0, 1, 2, 0])
        d1 = np.array([3, 0, 2, 1])
        d2 = np.array([4, 1, 3, 0])

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(d0, dtype=num.int64),
            num.array(d1, dtype=num.int64),
            num.array(d2, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[d0, d1, d2]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_3d_output(self) -> None:
        src_np = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
        d0 = np.array([[[0], [1]], [[2], [3]]])
        d1 = np.array([[[0, 2, 4]]])
        d2 = np.array([[[1, 3, 5]]])

        d0_bc, d1_bc, d2_bc = np.broadcast_arrays(d0, d1, d2)
        d0_bc = np.ascontiguousarray(d0_bc)
        d1_bc = np.ascontiguousarray(d1_bc)
        d2_bc = np.ascontiguousarray(d2_bc)

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(d0_bc, dtype=num.int64),
            num.array(d1_bc, dtype=num.int64),
            num.array(d2_bc, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[d0, d1, d2]
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    def test_random(self) -> None:
        rng = np.random.default_rng(88)
        src_np = rng.random((20, 30, 40)).astype(np.float32)
        n_idx = 500
        d0 = rng.integers(0, 20, size=n_idx)
        d1 = rng.integers(0, 30, size=n_idx)
        d2 = rng.integers(0, 40, size=n_idx)

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(d0, dtype=num.int64),
            num.array(d1, dtype=num.int64),
            num.array(d2, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[d0, d1, d2]
        np.testing.assert_array_almost_equal(np.asarray(output), expected)


@_gpu_only
class TestAll2AllFortranOrder:
    """ALL2ALL with Fortran-order (column-major) source arrays.

    Validates that the kernel handles non-C-contiguous memory layouts
    correctly via accessor-based indexing rather than assuming C-order.
    """

    def test_2d_source(self) -> None:
        src_np = np.asfortranarray(
            np.arange(20, dtype=np.float32).reshape(4, 5)
        )
        rows_np = np.array([0, 1, 3, 2])
        cols_np = np.array([4, 0, 2, 1])

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_3d_source(self) -> None:
        src_np = np.asfortranarray(
            np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        )
        d0 = np.array([0, 1, 2, 0])
        d1 = np.array([3, 0, 2, 1])
        d2 = np.array([4, 1, 3, 0])

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(d0, dtype=num.int64),
            num.array(d1, dtype=num.int64),
            num.array(d2, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[d0, d1, d2]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_2d_random(self) -> None:
        rng = np.random.default_rng(77)
        src_np = np.asfortranarray(rng.random((200, 300)).astype(np.float32))
        n_idx = 5_000
        rows_np = rng.integers(0, 200, size=n_idx)
        cols_np = rng.integers(0, 300, size=n_idx)

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    def test_output(self) -> None:
        src_np = np.arange(20, dtype=np.float64).reshape(4, 5)
        rows_np = np.array([[0, 1], [2, 3]])
        cols_np = np.array([[4, 0], [2, 1]])

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.array(
            np.empty(point_idx.shape, dtype=np.float64, order="F")
        )
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_input_output(self) -> None:
        src_np = np.asfortranarray(
            np.arange(30, dtype=np.float32).reshape(5, 6)
        )
        rows_np = np.array([[0], [2], [4]])
        cols_np = np.array([1, 3, 5])

        rows_bc, cols_bc = np.broadcast_arrays(rows_np, cols_np)
        rows_bc = np.ascontiguousarray(rows_bc)
        cols_bc = np.ascontiguousarray(cols_bc)

        source = num.array(src_np)
        point_idx = _build_point_index(
            source,
            num.array(rows_bc, dtype=num.int64),
            num.array(cols_bc, dtype=num.int64),
        )
        output = num.array(
            np.empty(point_idx.shape, dtype=np.float32, order="F")
        )
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)


@_gpu_only
class TestAll2AllSlicedSource:
    def test_1d_slice(self) -> None:
        full = num.arange(100, dtype=num.float32)
        source = full[20:50]
        indices = num.array([0, 5, 10, 29], dtype=num.int64)
        output = num.empty(4, dtype=num.float32)
        _run_all2all_1d(source, indices, output)
        expected = np.arange(100, dtype=np.float32)[20:50][[0, 5, 10, 29]]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_2d_row_slice(self) -> None:
        full_np = np.arange(60, dtype=np.float32).reshape(10, 6)
        full = num.array(full_np)
        source = full[3:7, :]
        src_np = full_np[3:7, :]

        rows_np = np.array([0, 1, 2, 3])
        cols_np = np.array([5, 0, 3, 2])

        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_2d_col_slice(self) -> None:
        full_np = np.arange(40, dtype=np.float64).reshape(5, 8)
        full = num.array(full_np)
        source = full[:, 2:6]
        src_np = full_np[:, 2:6]

        rows_np = np.array([0, 1, 4, 3])
        cols_np = np.array([0, 3, 1, 2])

        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float64)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_2d_both_axes_sliced(self) -> None:
        full_np = np.arange(120, dtype=np.float32).reshape(10, 12)
        full = num.array(full_np)
        source = full[2:8, 3:9]
        src_np = full_np[2:8, 3:9]

        rows_np = np.array([0, 5, 3, 1, 4])
        cols_np = np.array([0, 5, 2, 4, 1])

        point_idx = _build_point_index(
            source,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        output = num.empty(point_idx.shape, dtype=num.float32)
        _run_all2all_nd(source, point_idx, output)

        expected = src_np[rows_np, cols_np]
        np.testing.assert_array_equal(np.asarray(output), expected)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
