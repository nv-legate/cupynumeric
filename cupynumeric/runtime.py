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
from __future__ import annotations

import math
import warnings
from functools import lru_cache, reduce
from typing import TYPE_CHECKING, Any, Sequence, TypeGuard

import legate.core.types as ty
import numpy as np
from legate.core import (
    LEGATE_MAX_DIM,
    Scalar,
    TaskTarget,
    get_legate_runtime,
    DimOrdering,
)

from ._utils.array import calculate_volume, is_supported_dtype, to_core_type
from ._utils.stack import find_last_user_stacklevel
from .config import (
    BitGeneratorOperation,
    CuPyNumericOpCode,
    TransferType,
    cupynumeric_lib,
)

# We need to be careful about importing from other cupynumeric modules. The
# runtime is global and used in many places, but also depends on many of the
# other modules. Things like config and utils are OK, but imports for thunks,
# array types, etc. need to be deferred in order to avoid circular imports.


if TYPE_CHECKING:
    import numpy.typing as npt
    from legate.core import AutoTask, ManualTask

    from ._thunk.deferred import DeferredArray
    from ._thunk.eager import EagerArray
    from ._thunk.thunk import NumPyThunk
    from .types import NdShape

DIMENSION = int

legate_runtime = get_legate_runtime()


def thunk_from_scalar(
    bytes: bytes, shape: NdShape, dtype: np.dtype[Any]
) -> DeferredArray:
    from ._thunk.deferred import DeferredArray

    store = legate_runtime.create_store_from_scalar(
        Scalar(bytes, to_core_type(dtype)), shape=shape
    )
    return DeferredArray(store)


@lru_cache
def cached_thunk_from_scalar(
    bytes: bytes, shape: NdShape, dtype: np.dtype[Any]
) -> DeferredArray:
    return thunk_from_scalar(bytes, shape, dtype)


class Runtime(object):
    def __init__(self) -> None:
        self.library = legate_runtime.find_library(cupynumeric_lib.name)
        self.current_random_epoch = 0
        self.current_random_bitgenid = 0
        self.current_random_bitgen_zombies: tuple[Any, ...] = ()
        self.destroyed = False
        self.api_calls: list[tuple[str, str, bool]] = []

        max_eager_volume = (
            cupynumeric_lib.shared_object.cupynumeric_max_eager_volume()
        )
        self.max_eager_volume = int(np.asarray(max_eager_volume))

        assert cupynumeric_lib.shared_object is not None
        self.cupynumeric_lib = cupynumeric_lib.shared_object
        self.has_cusolvermp = (
            cupynumeric_lib.shared_object.cupynumeric_has_cusolvermp()
        )

        from .settings import settings

        settings.warn = settings.warn() or settings.test()

        if self.num_gpus > 0 and settings.preload_cudalibs():
            self._load_cudalibs()

        # Maps value types to struct types used in argmin/argmax
        self._cached_argred_types: dict[ty.Type, ty.Type] = dict()

    def cusolver_has_geev(self) -> bool:
        if not hasattr(self, "cusolver_has_geev_"):
            self.cusolver_has_geev_ = (
                cupynumeric_lib.shared_object.cupynumeric_cusolver_has_geev()
            )
        return self.cusolver_has_geev_

    @property
    def num_procs(self) -> int:
        return len(legate_runtime.machine)

    @property
    def num_gpus(self) -> int:
        return legate_runtime.machine.count(TaskTarget.GPU)

    def record_api_call(
        self, name: str, location: str, implemented: bool
    ) -> None:
        from .settings import settings

        assert settings.report_coverage()
        self.api_calls.append((name, location, implemented))

    def _load_cudalibs(self) -> None:
        task = legate_runtime.create_manual_task(
            self.library, CuPyNumericOpCode.LOAD_CUDALIBS, [self.num_gpus]
        )
        task.execute()
        legate_runtime.issue_execution_fence(block=True)

    def get_argred_type(self, value_dtype: ty.Type) -> ty.Type:
        cached = self._cached_argred_types.get(value_dtype)
        if cached is not None:
            return cached
        argred_dtype = ty.struct_type([ty.int64, value_dtype], True)
        self._cached_argred_types[value_dtype] = argred_dtype
        ids = self.cupynumeric_lib.cupynumeric_register_reduction_ops(
            value_dtype.code
        )
        argred_dtype.record_reduction_op(
            ty.ReductionOpKind.MAX, ids.argmax_redop_id
        )
        argred_dtype.record_reduction_op(
            ty.ReductionOpKind.MIN, ids.argmin_redop_id
        )
        return argred_dtype

    def _report_coverage(self) -> None:
        total = len(self.api_calls)
        implemented = sum(int(impl) for (_, _, impl) in self.api_calls)

        if total == 0:
            print("cuPyNumeric API coverage: 0/0")
        else:
            print(
                f"cuPyNumeric API coverage: {implemented}/{total} "
                f"({implemented / total * 100}%)"
            )

        from .settings import settings

        if (dump_csv := settings.report_dump_csv()) is not None:
            with open(dump_csv, "w") as f:
                print("function_name,location,implemented", file=f)
                for func_name, loc, impl in self.api_calls:
                    print(f"{func_name},{loc},{impl}", file=f)

    def destroy(self) -> None:
        from .settings import settings

        assert not self.destroyed
        if settings.report_coverage():
            self._report_coverage()
        self.destroyed = True

    def bitgenerator_populate_task(
        self,
        task: AutoTask | ManualTask,
        taskop: int,
        generatorID: int,
        generatorType: int = 0,
        seed: int | None = 0,
        flags: int = 0,
    ) -> None:
        task.add_scalar_arg(taskop, ty.int32)
        task.add_scalar_arg(generatorID, ty.int32)
        task.add_scalar_arg(generatorType, ty.uint32)
        task.add_scalar_arg(seed, ty.uint64)
        task.add_scalar_arg(flags, ty.uint32)

    def bitgenerator_create(
        self,
        generatorType: int,
        seed: int | None,
        flags: int,
        forceCreate: bool = False,
    ) -> int:
        self.current_random_bitgenid = self.current_random_bitgenid + 1
        if forceCreate:
            task = legate_runtime.create_manual_task(
                self.library, CuPyNumericOpCode.BITGENERATOR, (self.num_procs,)
            )
            self.bitgenerator_populate_task(
                task,
                BitGeneratorOperation.CREATE,
                self.current_random_bitgenid,
                generatorType,
                seed,
                flags,
            )
            task.add_scalar_arg(
                self.current_random_bitgen_zombies, (ty.uint32,)
            )
            self.current_random_bitgen_zombies = ()
            task.execute()
            legate_runtime.issue_execution_fence()
        return self.current_random_bitgenid

    def bitgenerator_destroy(
        self, handle: Any, disposing: bool = True
    ) -> None:
        if disposing:
            # when called from within destructor, do not schedule a task
            self.current_random_bitgen_zombies += (handle,)
        else:
            # with explicit destruction, do schedule a task
            legate_runtime.issue_execution_fence()
            task = legate_runtime.create_manual_task(
                self.library, CuPyNumericOpCode.BITGENERATOR, (self.num_procs,)
            )
            self.bitgenerator_populate_task(
                task, BitGeneratorOperation.DESTROY, handle
            )
            task.add_scalar_arg(
                self.current_random_bitgen_zombies, (ty.uint32,)
            )
            self.current_random_bitgen_zombies = ()
            task.execute()

    def set_next_random_epoch(self, epoch: int) -> None:
        self.current_random_epoch = epoch

    def get_next_random_epoch(self) -> int:
        result = self.current_random_epoch
        self.current_random_epoch += 1
        return result

    def get_numpy_thunk(
        self, obj: Any, share: bool = False, dtype: np.dtype[Any] | None = None
    ) -> NumPyThunk:
        # Check to see if this object implements the Legate data interface
        if hasattr(obj, "__legate_data_interface__"):
            legate_data = obj.__legate_data_interface__
            if legate_data["version"] != 1:
                raise NotImplementedError(
                    "Need support for other Legate data interface versions"
                )
            data = legate_data["data"]
            if len(data) != 1:
                raise ValueError("Legate data must be array-like")
            field = next(iter(data))
            array = data[field]
            if array.nested or array.nullable:
                raise NotImplementedError(
                    "Array must be non-nullable and not nested"
                )

            from ._thunk.deferred import DeferredArray

            return DeferredArray(array.data)

        # See if this is a normal numpy array
        # Make sure to convert numpy matrices to numpy arrays here
        # as the former doesn't behave quite like the latter
        if type(obj) is not np.ndarray:
            # Check to see if this object implements the dlpack interface
            if hasattr(obj, "__dlpack__"):
                from . import from_dlpack

                return from_dlpack(obj)._thunk

            # If it's not, make it into a numpy array
            if share:
                obj = np.asarray(obj, dtype=dtype)
            else:
                from ._array.array import ndarray
                from ._module.array_joining import stack

                if (
                    isinstance(obj, (tuple, list))
                    and len(obj) > 1
                    and all(isinstance(o, (ndarray, np.ndarray)) for o in obj)
                    and math.prod(obj[0].shape) != 0
                ):
                    return stack(obj)._thunk
                obj = np.array(obj, dtype=dtype)
        elif dtype is not None and dtype != obj.dtype:
            obj = obj.astype(dtype)
        # We can't attach NumPy ndarrays in shared mode unless they are
        # writeable
        share = share and obj.flags["W"]
        transfer = TransferType.SHARE if share else TransferType.MAKE_COPY
        return self.find_or_create_array_thunk(obj, transfer)

    @staticmethod
    def compute_parent_child_mapping(
        array: npt.NDArray[Any],
    ) -> tuple[slice | None, ...] | None:
        # We need an algorithm for figuring out how to compute the
        # slice object that was used to generate a child array from
        # a parent array so we can build the same mapping from a
        # logical region to a subregion
        assert array.base is not None
        parent_ptr = int(array.base.ctypes.data)
        child_ptr = int(array.ctypes.data)
        assert child_ptr >= parent_ptr
        ptr_diff = child_ptr - parent_ptr
        parent_shape = array.base.shape
        div = (
            reduce(lambda x, y: x * y, parent_shape)
            if len(parent_shape) > 1
            else parent_shape[0]
        )
        div *= array.dtype.itemsize
        offsets = list()
        # Compute the offsets in the parent index
        for n in parent_shape:
            mod = div
            div //= n
            offsets.append((ptr_diff % mod) // div)
        assert div == array.dtype.itemsize
        # Now build the view and dimmap for the parent to create the view
        key: tuple[slice | None, ...] = ()
        child_idx = 0
        child_strides = tuple(array.strides)
        parent_strides = tuple(array.base.strides)
        for idx in range(array.base.ndim):
            # Handle the adding and removing dimension cases
            if parent_strides[idx] == 0:
                # This was an added dimension in the parent
                if child_strides[child_idx] == 0:
                    # Kept an added dimension
                    key += (slice(None, None, None),)
                else:
                    # Removed an added dimension
                    key += (slice(None, None, None),)
                child_idx += 1
                continue
            elif child_idx == array.ndim:
                key += (slice(offsets[idx], offsets[idx] + 1, 1),)
                continue
            elif child_strides[child_idx] == 0:
                # Added dimension in the child not in the parent
                while child_strides[child_idx] == 0:
                    key += (np.newaxis,)
                    child_idx += 1
                # Fall through to the base case
            # Stides in the child should always be greater than or equal
            # to the strides in the parent, if they're not, then that
            # must be an added dimension
            start = offsets[idx]
            if child_strides[child_idx] < parent_strides[idx]:
                key += (slice(start, start + 1, 1),)
                # Doesn't count against the child_idx
            else:
                stride = child_strides[child_idx] // parent_strides[idx]
                stop = start + stride * array.shape[child_idx]
                key += (slice(start, stop, stride),)
                child_idx += 1
        assert child_idx <= array.ndim
        if child_idx < array.ndim:
            return None
        else:
            return key

    def find_or_create_array_thunk(
        self,
        array: npt.NDArray[Any],
        transfer: TransferType,
        read_only: bool = False,
        defer: bool = False,
    ) -> NumPyThunk:
        from ._thunk.deferred import DeferredArray

        assert isinstance(array, np.ndarray)
        if not is_supported_dtype(array.dtype):
            raise TypeError(
                f"cuPyNumeric does not support dtype={array.dtype}"
            )

        # We have to be really careful here to handle the case of
        # aliased numpy arrays that are passed in from the application
        # In case of aliasing we need to make sure that they are
        # mapped to the same logical region. The way we handle this
        # is to always create the thunk for the root array and
        # then create sub-thunks that mirror the array views
        if (
            transfer == TransferType.SHARE
            and array.base is not None
            and isinstance(array.base, np.ndarray)
        ):
            key = self.compute_parent_child_mapping(array)
            if key is None:
                # This base array wasn't made with a view
                raise NotImplementedError(
                    "cuPyNumeric does not currently know "
                    + "how to attach to array views that are not affine "
                    + "transforms of their parent array."
                )

            parent_thunk = self.find_or_create_array_thunk(
                array.base, transfer, read_only, defer
            )
            return parent_thunk.get_item(key)

        # Once it's a normal numpy array we can make it into one of our arrays
        # Check to see if it is a type that we support for doing deferred
        # execution and big enough to be worth off-loading onto Legion
        if defer or not self.is_eager_shape(array.shape):
            if array.size == 1 and transfer != TransferType.SHARE:
                # This is a single value array that we're not attaching to.
                # We cache these, but only if the user has promised not to
                # write-through them.
                # TODO(mpapadakis): Also mark the Store as read-only, whenever
                # Legate supports that.
                if read_only:
                    return cached_thunk_from_scalar(
                        array.tobytes(), array.shape, array.dtype
                    )
                else:
                    return thunk_from_scalar(
                        array.tobytes(), array.shape, array.dtype
                    )

            if transfer == TransferType.MAKE_COPY:
                ordering = DimOrdering.c_order()
            else:
                if array.flags["F_CONTIGUOUS"]:
                    ordering = DimOrdering.fortran_order()
                elif array.flags["C_CONTIGUOUS"]:
                    ordering = DimOrdering.c_order()
                else:
                    raise ValueError(
                        "Only F_CONTIGUOUS and C_CONTIGUOUS arrays are supported."
                    )

            # This is not a scalar so make a field.
            # We won't try to cache these bigger arrays.
            store = legate_runtime.create_store_from_buffer(
                to_core_type(array.dtype),
                array.shape,
                array.copy() if transfer == TransferType.MAKE_COPY else array,
                # This argument should really be called "donate"
                read_only=read_only,
                ordering=ordering,
            )
            return DeferredArray(store)

        from ._thunk.eager import EagerArray

        # Make this into an eagerly evaluated thunk
        return EagerArray(
            array.copy() if transfer == TransferType.MAKE_COPY else array
        )

    def create_eager_thunk(
        self, shape: NdShape, dtype: np.dtype[Any]
    ) -> NumPyThunk:
        from ._thunk.eager import EagerArray

        return EagerArray(np.empty(shape, dtype=dtype))

    def create_deferred_thunk(
        self, shape: NdShape, dtype: ty.Type
    ) -> DeferredArray:
        from ._thunk.deferred import DeferredArray

        store = legate_runtime.create_store(
            dtype, shape=shape, optimize_scalar=True
        )

        return DeferredArray(store)

    def create_unbound_thunk(
        self, dtype: ty.Type, ndim: int = 1
    ) -> DeferredArray:
        from ._thunk.deferred import DeferredArray

        store = legate_runtime.create_store(dtype, ndim=ndim)
        return DeferredArray(store)

    def is_eager_shape(self, shape: NdShape) -> bool:
        volume = calculate_volume(shape)

        # Special cases that must always be eager:

        # Newly created empty arrays
        if volume == 0:
            return True

        # Arrays with more dimensions than what Legion was compiled for
        if len(shape) > LEGATE_MAX_DIM:
            return True

        from .settings import settings

        # CUPYNUMERIC_FORCE_THUNK == "eager"
        if settings.force_thunk() == "eager":
            return True

        if settings.force_thunk() == "deferred":
            return False

        # no forcing; auto mode
        if len(shape) == 0:
            return self.max_eager_volume > 0

        # Otherwise, see if the volume is large enough
        return volume <= self.max_eager_volume

    @staticmethod
    def are_all_eager_inputs(inputs: Sequence[NumPyThunk] | None) -> bool:
        from ._thunk.eager import EagerArray
        from ._thunk.thunk import NumPyThunk

        if inputs is None:
            return True
        for inp in inputs:
            assert isinstance(inp, NumPyThunk)
            if not isinstance(inp, EagerArray):
                return False
        return True

    @staticmethod
    def is_eager_array(array: NumPyThunk) -> TypeGuard[EagerArray]:
        from ._thunk.eager import EagerArray

        return isinstance(array, EagerArray)

    @staticmethod
    def is_deferred_array(
        array: NumPyThunk | None,
    ) -> TypeGuard[DeferredArray]:
        from ._thunk.deferred import DeferredArray

        return isinstance(array, DeferredArray)

    def to_eager_array(self, array: NumPyThunk) -> EagerArray:
        from ._thunk.eager import EagerArray

        if self.is_eager_array(array):
            return array
        elif self.is_deferred_array(array):
            return EagerArray(array.__numpy_array__())
        else:
            raise RuntimeError("invalid array type")

    def to_deferred_array(
        self, array: NumPyThunk, read_only: bool
    ) -> DeferredArray:
        if self.is_deferred_array(array):
            return array
        elif self.is_eager_array(array):
            return array.to_deferred_array(read_only)
        else:
            raise RuntimeError("invalid array type")

    def warn(self, msg: str, category: type = UserWarning) -> None:
        from .settings import settings

        if not settings.warn():
            return
        stacklevel = find_last_user_stacklevel()
        warnings.warn(msg, stacklevel=stacklevel, category=category)


runtime = Runtime()


def _shutdown_callback() -> None:
    runtime.destroy()


legate_runtime.add_shutdown_callback(_shutdown_callback)
