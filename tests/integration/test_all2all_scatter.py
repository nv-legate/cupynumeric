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


def _run_nccl_scatter_stores(source_store, index_store, output_store):
    legate_runtime = get_legate_runtime()
    library = cpn_runtime.library

    task = legate_runtime.create_auto_task(
        library, CuPyNumericOpCode.ALL2ALL_SCATTER
    )
    task.add_input(source_store)
    task.add_input(index_store)
    task.add_output(output_store)
    task.add_input(output_store)
    task.add_alignment(source_store, index_store)
    task.add_nccl_communicator()
    task.execute()

    legate_runtime.issue_execution_fence(block=True)


def _run_nccl_scatter_1d(source, indices, output):
    out_thunk = _to_deferred(output)
    converted = out_thunk.base.has_scalar_storage
    if converted:
        out_thunk = out_thunk._convert_future_to_regionfield()
    _run_nccl_scatter_stores(
        _get_store(source), _get_store(indices), out_thunk.base
    )
    if converted:
        _to_deferred(output).copy(out_thunk, deep=True)


def _build_point_index(target, *coord_arrays):
    tgt_thunk = _to_deferred(target)
    coord_thunks = tuple(_to_deferred(c) for c in coord_arrays)
    return tgt_thunk._zip_indices(
        0, coord_thunks, check_bounds=settings.bounds_check_enabled("indexing")
    )


def _run_nccl_scatter_nd(point_index_thunk, source, output):
    _run_nccl_scatter_stores(
        _get_store(source), point_index_thunk.base, _get_store(output)
    )


_gpu_only = pytest.mark.skipif(
    get_legate_runtime().machine.count(TaskTarget.GPU) <= 1,
    reason="NCCL scatter is GPU-only",
)


@pytest.fixture(autouse=True)
def enable_nccl_scatter():
    settings.use_nccl_scatter = True
    yield
    settings.use_nccl_scatter.unset_value()


@_gpu_only
class TestNcclScatter1DOutput:
    @pytest.mark.parametrize("dtype", [num.float32, num.float64])
    def test_basic_floating(self, dtype) -> None:
        # output[indices[p]] = source[p]
        sentinel = -7.0
        output = num.full(10, sentinel, dtype=dtype)
        source = num.array([100.0, 200.0, 300.0, 400.0], dtype=dtype)
        indices = num.array([0, 2, 5, 9], dtype=num.int64)

        _run_nccl_scatter_1d(source, indices, output)

        expected = np.full(10, sentinel, dtype=dtype)
        expected[[0, 2, 5, 9]] = [100.0, 200.0, 300.0, 400.0]
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    @pytest.mark.parametrize("dtype", [num.int32, num.int64])
    def test_basic_integer(self, dtype) -> None:
        sentinel = -1
        output = num.full(8, sentinel, dtype=dtype)
        source = num.array([10, 20, 30], dtype=dtype)
        indices = num.array([1, 3, 7], dtype=num.int64)

        _run_nccl_scatter_1d(source, indices, output)

        expected = np.full(8, sentinel, dtype=dtype)
        expected[[1, 3, 7]] = [10, 20, 30]
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_reversed(self) -> None:
        output = num.full(5, -1, dtype=num.int64)
        source = num.array([100, 200, 300, 400, 500], dtype=num.int64)
        indices = num.array([4, 3, 2, 1, 0], dtype=num.int64)

        _run_nccl_scatter_1d(source, indices, output)

        np.testing.assert_array_equal(
            np.asarray(output), [500, 400, 300, 200, 100]
        )

    def test_single_element(self) -> None:
        output = num.full(100, 0.0, dtype=num.float32)
        source = num.array([42.0], dtype=num.float32)
        indices = num.array([42], dtype=num.int64)

        _run_nccl_scatter_1d(source, indices, output)

        expected = np.zeros(100, dtype=np.float32)
        expected[42] = 42.0
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_2d_source(self) -> None:
        output = num.full(12, -1.0, dtype=num.float32)
        src_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        idx_np = np.array([[0, 3, 6], [9, 11, 1]], dtype=np.int64)
        source = num.array(src_np)
        indices = num.array(idx_np)

        _run_nccl_scatter_1d(source, indices, output)

        expected = np.full(12, -1.0, dtype=np.float32)
        expected[idx_np.ravel()] = src_np.ravel()
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_3d_source(self) -> None:
        output = num.full(24, -1.0, dtype=num.float32)
        src_np = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        idx_np = np.array(
            [[[0, 1, 2], [3, 4, 5]], [[23, 22, 21], [20, 19, 18]]],
            dtype=np.int64,
        )
        source = num.array(src_np)
        indices = num.array(idx_np)

        _run_nccl_scatter_1d(source, indices, output)

        expected = np.full(24, -1.0, dtype=np.float32)
        expected[idx_np.ravel()] = src_np.ravel()
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_random_unique(self) -> None:
        n = 10_000
        rng = np.random.default_rng(42)
        idx_np = rng.permutation(n)[:2048].astype(np.int64)
        src_np = rng.random(2048).astype(np.float32)

        output = num.full(n, -1.0, dtype=num.float32)
        source = num.array(src_np)
        indices = num.array(idx_np)

        _run_nccl_scatter_1d(source, indices, output)

        expected = np.full(n, -1.0, dtype=np.float32)
        expected[idx_np] = src_np
        np.testing.assert_array_almost_equal(np.asarray(output), expected)


@_gpu_only
class TestNcclScatter2DOutput:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_basic_floating(self, dtype) -> None:
        out_np = np.full((4, 5), -1.0, dtype=dtype)
        rows_np = np.array([0, 1, 3, 2])
        cols_np = np.array([4, 0, 2, 1])
        src_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=dtype)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[rows_np, cols_np] = src_np
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    def test_basic_int32(self) -> None:
        out_np = np.full((3, 4), -1, dtype=np.int32)
        rows_np = np.array([0, 2, 1])
        cols_np = np.array([3, 1, 0])
        src_np = np.array([7, 8, 9], dtype=np.int32)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[rows_np, cols_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_2d_source_array(self) -> None:
        out_np = np.full((5, 6), -1.0, dtype=np.float32)
        rows_np = np.array([[0], [2], [4]])
        cols_np = np.array([1, 3, 5])
        rows_bc, cols_bc = np.broadcast_arrays(rows_np, cols_np)
        rows_bc = np.ascontiguousarray(rows_bc)
        cols_bc = np.ascontiguousarray(cols_bc)
        src_np = np.arange(rows_bc.size, dtype=np.float32).reshape(
            rows_bc.shape
        )

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(rows_bc, dtype=num.int64),
            num.array(cols_bc, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[rows_np, cols_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_random_unique(self) -> None:
        rng = np.random.default_rng(123)
        out_np = np.full((500, 500), -1.0, dtype=np.float32)
        n_idx = 5_000

        flat_unique = rng.choice(500 * 500, size=n_idx, replace=False)
        rows_np = (flat_unique // 500).astype(np.int64)
        cols_np = (flat_unique % 500).astype(np.int64)
        src_np = rng.random(n_idx).astype(np.float32)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[rows_np, cols_np] = src_np
        np.testing.assert_array_almost_equal(np.asarray(output), expected)


@_gpu_only
class TestNcclScatter3DOutput:
    def test_float32(self) -> None:
        out_np = np.full((3, 4, 5), -1.0, dtype=np.float32)
        d0 = np.array([0, 1, 2, 0])
        d1 = np.array([3, 0, 2, 1])
        d2 = np.array([4, 1, 3, 0])
        src_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(d0, dtype=num.int64),
            num.array(d1, dtype=num.int64),
            num.array(d2, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[d0, d1, d2] = src_np
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_3d_output(self) -> None:
        out_np = np.full((4, 5, 6), -1.0, dtype=np.float32)
        d0 = np.array([[[0], [1]], [[2], [3]]])
        d1 = np.array([[[0, 2, 4]]])
        d2 = np.array([[[1, 3, 5]]])

        d0_bc, d1_bc, d2_bc = np.broadcast_arrays(d0, d1, d2)
        d0_bc = np.ascontiguousarray(d0_bc)
        d1_bc = np.ascontiguousarray(d1_bc)
        d2_bc = np.ascontiguousarray(d2_bc)
        src_np = np.arange(d0_bc.size, dtype=np.float32).reshape(d0_bc.shape)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(d0_bc, dtype=num.int64),
            num.array(d1_bc, dtype=num.int64),
            num.array(d2_bc, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[d0_bc, d1_bc, d2_bc] = src_np
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    def test_random_unique(self) -> None:
        rng = np.random.default_rng(88)
        out_np = np.full((20, 30, 40), -1.0, dtype=np.float32)
        n_idx = 500
        flat_unique = rng.choice(20 * 30 * 40, size=n_idx, replace=False)
        d0 = (flat_unique // (30 * 40)).astype(np.int64)
        d1 = ((flat_unique // 40) % 30).astype(np.int64)
        d2 = (flat_unique % 40).astype(np.int64)
        src_np = rng.random(n_idx).astype(np.float32)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(d0, dtype=num.int64),
            num.array(d1, dtype=num.int64),
            num.array(d2, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[d0, d1, d2] = src_np
        np.testing.assert_array_almost_equal(np.asarray(output), expected)


@_gpu_only
class TestNcclScatterFortranOrder:
    """NCCL scatter with Fortran-order (column-major) output arrays.

    Validates that the kernel handles non-C-contiguous memory layouts
    correctly via accessor-based indexing.
    """

    def test_2d_output(self) -> None:
        out_np = np.asfortranarray(np.full((4, 5), -1.0, dtype=np.float32))
        rows_np = np.array([0, 1, 3, 2])
        cols_np = np.array([4, 0, 2, 1])
        src_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[rows_np, cols_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_3d_output(self) -> None:
        out_np = np.asfortranarray(np.full((3, 4, 5), -1.0, dtype=np.float32))
        d0 = np.array([0, 1, 2, 0])
        d1 = np.array([3, 0, 2, 1])
        d2 = np.array([4, 1, 3, 0])
        src_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(d0, dtype=num.int64),
            num.array(d1, dtype=num.int64),
            num.array(d2, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[d0, d1, d2] = src_np
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_2d_random_unique(self) -> None:
        rng = np.random.default_rng(77)
        out_np = np.asfortranarray(np.full((200, 300), -1.0, dtype=np.float32))
        n_idx = 2_500
        flat_unique = rng.choice(200 * 300, size=n_idx, replace=False)
        rows_np = (flat_unique // 300).astype(np.int64)
        cols_np = (flat_unique % 300).astype(np.int64)
        src_np = rng.random(n_idx).astype(np.float32)

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[rows_np, cols_np] = src_np
        np.testing.assert_array_almost_equal(np.asarray(output), expected)

    def test_source(self) -> None:
        out_np = np.full((4, 5), -1.0, dtype=np.float64)
        rows_np = np.array([[0, 1], [2, 3]])
        cols_np = np.array([[4, 0], [2, 1]])
        src_np = np.asfortranarray(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        )

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[rows_np, cols_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), expected)

    def test_input_output(self) -> None:
        out_np = np.asfortranarray(np.full((5, 6), -1.0, dtype=np.float32))
        rows_np = np.array([[0], [2], [4]])
        cols_np = np.array([1, 3, 5])
        rows_bc, cols_bc = np.broadcast_arrays(rows_np, cols_np)
        rows_bc = np.ascontiguousarray(rows_bc)
        cols_bc = np.ascontiguousarray(cols_bc)
        src_np = np.asfortranarray(
            np.arange(rows_bc.size, dtype=np.float32).reshape(rows_bc.shape)
        )

        output = num.array(out_np)
        source = num.array(src_np)
        point_idx = _build_point_index(
            output,
            num.array(rows_bc, dtype=num.int64),
            num.array(cols_bc, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        expected = out_np.copy()
        expected[rows_np, cols_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), expected)


@_gpu_only
class TestNcclScatterSlicedOutput:
    def test_1d_slice(self) -> None:
        full = num.full(100, -1.0, dtype=num.float32)
        output = full[20:50]
        full_ref = np.full(100, -1.0, dtype=np.float32)
        out_ref = full_ref[20:50].copy()

        src_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        idx_np = np.array([0, 5, 10, 29], dtype=np.int64)
        source = num.array(src_np)
        indices = num.array(idx_np)

        _run_nccl_scatter_1d(source, indices, output)

        out_ref[idx_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), out_ref)

    def test_2d_row_slice(self) -> None:
        full_np = np.full((10, 6), -1.0, dtype=np.float32)
        full = num.array(full_np)
        output = full[3:7, :]
        out_ref = full_np[3:7, :].copy()

        rows_np = np.array([0, 1, 2, 3])
        cols_np = np.array([5, 0, 3, 2])
        src_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        source = num.array(src_np)

        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        out_ref[rows_np, cols_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), out_ref)

    def test_2d_col_slice(self) -> None:
        full_np = np.full((5, 8), -1.0, dtype=np.float64)
        full = num.array(full_np)
        output = full[:, 2:6]
        out_ref = full_np[:, 2:6].copy()

        rows_np = np.array([0, 1, 4, 3])
        cols_np = np.array([0, 3, 1, 2])
        src_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        source = num.array(src_np)

        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        out_ref[rows_np, cols_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), out_ref)

    def test_2d_both_axes_sliced(self) -> None:
        full_np = np.full((10, 12), -1.0, dtype=np.float32)
        full = num.array(full_np)
        output = full[2:8, 3:9]
        out_ref = full_np[2:8, 3:9].copy()

        rows_np = np.array([0, 5, 3, 1, 4])
        cols_np = np.array([0, 5, 2, 4, 1])
        src_np = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        source = num.array(src_np)

        point_idx = _build_point_index(
            output,
            num.array(rows_np, dtype=num.int64),
            num.array(cols_np, dtype=num.int64),
        )
        _run_nccl_scatter_nd(point_idx, source, output)

        out_ref[rows_np, cols_np] = src_np
        np.testing.assert_array_equal(np.asarray(output), out_ref)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
