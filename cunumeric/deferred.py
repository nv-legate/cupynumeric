# Copyright 2021-2023 NVIDIA Corporation
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

import weakref
from collections import Counter
from collections.abc import Iterable
from enum import IntEnum, unique
from functools import reduce, wraps
from inspect import signature
from itertools import chain, product
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import legate.core.types as ty
import numpy as np
from legate.core import Annotation, Future, ReductionOp, Store
from legate.core.store import RegionField
from legate.core.utils import OrderedSet
from numpy.core.numeric import (  # type: ignore [attr-defined]
    normalize_axis_tuple,
)
from typing_extensions import ParamSpec

from .config import (
    BinaryOpCode,
    BitGeneratorDistribution,
    BitGeneratorOperation,
    Bitorder,
    ConvertCode,
    CuNumericOpCode,
    RandGenCode,
    UnaryOpCode,
    UnaryRedCode,
)
from .linalg.cholesky import cholesky
from .linalg.solve import solve
from .sort import sort
from .thunk import NumPyThunk
from .utils import is_advanced_indexing, to_core_dtype

if TYPE_CHECKING:
    import numpy.typing as npt
    from legate.core import FieldID, Region
    from legate.core.operation import AutoTask, ManualTask

    from .config import BitGeneratorType, FFTDirection, FFTType, WindowOpCode
    from .runtime import Runtime
    from .types import (
        BitOrder,
        ConvolveMode,
        NdShape,
        OrderType,
        SelectKind,
        SortSide,
        SortType,
    )


_COMPLEX_FIELD_DTYPES = {
    ty.complex64: ty.float32,
    ty.complex128: ty.float64,
}


def _prod(tpl: Sequence[int]) -> int:
    return reduce(lambda a, b: a * b, tpl, 1)


R = TypeVar("R")
P = ParamSpec("P")


def auto_convert(
    *thunk_params: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Converts all named parameters to DeferredArrays.
    """
    keys = OrderedSet(thunk_params)
    assert len(keys) == len(thunk_params)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        assert not hasattr(
            func, "__wrapped__"
        ), "this decorator must be the innermost"

        # For each parameter specified by name, also consider the case where
        # it's passed as a positional parameter.
        params = signature(func).parameters
        extra = keys - OrderedSet(params)
        assert len(extra) == 0, f"unknown parameter(s): {extra}"
        indices = {idx for (idx, param) in enumerate(params) if param in keys}

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # Convert relevant arguments to DeferredArrays
            self = args[0]
            args = tuple(
                self.runtime.to_deferred_array(arg)
                if idx in indices and arg is not None
                else arg
                for (idx, arg) in enumerate(args)
            )
            for k, v in kwargs.items():
                if k in keys and v is not None:
                    kwargs[k] = self.runtime.to_deferred_array(v)

            return func(*args, **kwargs)

        return wrapper

    return decorator


# This is a dummy object that is only used as an initializer for the
# RegionField object above. It is thrown away as soon as the
# RegionField is constructed.
class _CuNumericNDarray(object):
    __slots__ = ["__array_interface__"]

    def __init__(
        self,
        shape: NdShape,
        field_type: Any,
        base_ptr: Any,
        strides: tuple[int, ...],
        read_only: bool,
    ) -> None:
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        self.__array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": field_type.str,
            "data": (base_ptr, read_only),
            "strides": strides,
        }


_UNARY_RED_TO_REDUCTION_OPS: Dict[int, int] = {
    UnaryRedCode.SUM: ReductionOp.ADD,
    UnaryRedCode.SUM_SQUARES: ReductionOp.ADD,
    UnaryRedCode.VARIANCE: ReductionOp.ADD,
    UnaryRedCode.PROD: ReductionOp.MUL,
    UnaryRedCode.MAX: ReductionOp.MAX,
    UnaryRedCode.MIN: ReductionOp.MIN,
    UnaryRedCode.ARGMAX: ReductionOp.MAX,
    UnaryRedCode.ARGMIN: ReductionOp.MIN,
    UnaryRedCode.NANARGMAX: ReductionOp.MAX,
    UnaryRedCode.NANARGMIN: ReductionOp.MIN,
    UnaryRedCode.NANMAX: ReductionOp.MAX,
    UnaryRedCode.NANMIN: ReductionOp.MIN,
    UnaryRedCode.NANPROD: ReductionOp.MUL,
    UnaryRedCode.NANSUM: ReductionOp.ADD,
    UnaryRedCode.CONTAINS: ReductionOp.ADD,
    UnaryRedCode.COUNT_NONZERO: ReductionOp.ADD,
    UnaryRedCode.ALL: ReductionOp.MUL,
    UnaryRedCode.ANY: ReductionOp.ADD,
}


def max_identity(
    ty: np.dtype[Any],
) -> Union[int, np.floating[Any], bool, np.complexfloating[Any, Any]]:
    if ty.kind == "i" or ty.kind == "u":
        return np.iinfo(ty).min
    elif ty.kind == "f":
        return np.finfo(ty).min
    elif ty.kind == "c":
        return np.finfo(np.float64).min + np.finfo(np.float64).min * 1j
    elif ty.kind == "b":
        return False
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


def min_identity(
    ty: np.dtype[Any],
) -> Union[int, np.floating[Any], bool, np.complexfloating[Any, Any]]:
    if ty.kind == "i" or ty.kind == "u":
        return np.iinfo(ty).max
    elif ty.kind == "f":
        return np.finfo(ty).max
    elif ty.kind == "c":
        return np.finfo(np.float64).max + np.finfo(np.float64).max * 1j
    elif ty.kind == "b":
        return True
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


_UNARY_RED_IDENTITIES: Dict[UnaryRedCode, Callable[[Any], Any]] = {
    UnaryRedCode.SUM: lambda _: 0,
    UnaryRedCode.SUM_SQUARES: lambda _: 0,
    UnaryRedCode.VARIANCE: lambda _: 0,
    UnaryRedCode.PROD: lambda _: 1,
    UnaryRedCode.MIN: min_identity,
    UnaryRedCode.MAX: max_identity,
    UnaryRedCode.ARGMAX: lambda ty: (np.iinfo(np.int64).min, max_identity(ty)),
    UnaryRedCode.ARGMIN: lambda ty: (np.iinfo(np.int64).min, min_identity(ty)),
    UnaryRedCode.CONTAINS: lambda _: False,
    UnaryRedCode.COUNT_NONZERO: lambda _: 0,
    UnaryRedCode.ALL: lambda _: True,
    UnaryRedCode.ANY: lambda _: False,
    UnaryRedCode.NANARGMAX: lambda ty: (
        np.iinfo(np.int64).min,
        max_identity(ty),
    ),
    UnaryRedCode.NANARGMIN: lambda ty: (
        np.iinfo(np.int64).min,
        min_identity(ty),
    ),
    UnaryRedCode.NANMAX: max_identity,
    UnaryRedCode.NANMIN: min_identity,
    UnaryRedCode.NANPROD: lambda _: 1,
    UnaryRedCode.NANSUM: lambda _: 0,
}


@unique
class BlasOperation(IntEnum):
    VV = 1
    MV = 2
    MM = 3


class DeferredArray(NumPyThunk):
    """This is a deferred thunk for describing NumPy computations.
    It is backed by either a Legion logical region or a Legion future
    for describing the result of a computation.

    :meta private:
    """

    def __init__(
        self,
        runtime: Runtime,
        base: Store,
        numpy_array: Optional[npt.NDArray[Any]] = None,
    ) -> None:
        super().__init__(runtime, base.type.to_numpy_dtype())
        assert base is not None
        assert isinstance(base, Store)
        self.base = base  # a Legate Store
        self.numpy_array = (
            None if numpy_array is None else weakref.ref(numpy_array)
        )

    def __str__(self) -> str:
        return f"DeferredArray(base: {self.base})"

    @property
    def storage(self) -> Union[Future, tuple[Region, Union[int, FieldID]]]:
        storage = self.base.storage
        if self.base.kind == Future:
            assert isinstance(storage, Future)
            return storage
        else:
            assert isinstance(storage, RegionField)
            return (storage.region, storage.field.field_id)

    @property
    def shape(self) -> NdShape:
        return tuple(self.base.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _copy_if_overlapping(self, other: DeferredArray) -> DeferredArray:
        if not self.base.overlaps(other.base):
            return self
        copy = cast(
            DeferredArray,
            self.runtime.create_empty_thunk(
                self.shape,
                self.base.type,
                inputs=[self],
            ),
        )
        copy.copy(self, deep=True)
        return copy

    def __numpy_array__(self) -> npt.NDArray[Any]:
        if self.numpy_array is not None:
            result = self.numpy_array()
            if result is not None:
                return result
        elif self.size == 0:
            # Return an empty array with the right number of dimensions
            # and type
            return np.empty(shape=self.shape, dtype=self.dtype)

        if self.base.kind == Future:
            # TODO: Updates to this "inline mapped" array won't actually
            # write-through to the Future
            result = np.full(
                self.shape,
                self.get_scalar_array(),
                dtype=self.dtype,
            )
        else:
            alloc = self.base.get_inline_allocation()

            def construct_ndarray(
                shape: NdShape, address: Any, strides: tuple[int, ...]
            ) -> npt.NDArray[Any]:
                initializer = _CuNumericNDarray(
                    shape, self.dtype, address, strides, False
                )
                result = np.asarray(initializer)
                if self.shape == ():
                    result = result.reshape(())
                return result

            result = cast("npt.NDArray[Any]", alloc.consume(construct_ndarray))

        self.numpy_array = weakref.ref(result)
        return result

    @property
    def can_write_through_numpy_array(self) -> bool:
        return self.base.kind != Future

    # TODO: We should return a view of the field instead of a copy
    def imag(self) -> NumPyThunk:
        result = self.runtime.create_empty_thunk(
            self.shape,
            dtype=_COMPLEX_FIELD_DTYPES[self.base.type],
            inputs=[self],
        )

        result.unary_op(
            UnaryOpCode.IMAG,
            self,
            True,
            [],
        )

        return result

    # TODO: We should return a view of the field instead of a copy
    def real(self) -> NumPyThunk:
        result = self.runtime.create_empty_thunk(
            self.shape,
            dtype=_COMPLEX_FIELD_DTYPES[self.base.type],
            inputs=[self],
        )

        result.unary_op(
            UnaryOpCode.REAL,
            self,
            True,
            [],
        )

        return result

    def conj(self) -> NumPyThunk:
        result = self.runtime.create_empty_thunk(
            self.shape,
            dtype=self.base.type,
            inputs=[self],
        )

        result.unary_op(
            UnaryOpCode.CONJ,
            self,
            True,
            [],
        )

        return result

    # Copy source array to the destination array
    @auto_convert("rhs")
    def copy(self, rhs: Any, deep: bool = False) -> None:
        if self.scalar and rhs.scalar:
            self.base.set_storage(rhs.base.storage)
            return
        self.unary_op(
            UnaryOpCode.COPY,
            rhs,
            True,
            [],
        )

    @property
    def scalar(self) -> bool:
        return self.base.scalar

    def get_scalar_array(self) -> npt.NDArray[Any]:
        assert self.scalar
        assert isinstance(self.base.storage, Future)
        buf = self.base.storage.get_buffer(self.dtype.itemsize)
        result = np.frombuffer(buf, dtype=self.dtype, count=1)
        return result.reshape(())

    def _zip_indices(
        self, start_index: int, arrays: tuple[Any, ...]
    ) -> DeferredArray:
        if not isinstance(arrays, tuple):
            raise TypeError("zip_indices expects tuple of arrays")
        # start_index is the index from witch indices arrays are passed
        # for example of arr[:, indx, :], start_index =1
        if start_index == -1:
            start_index = 0

        new_arrays: tuple[Any, ...] = tuple()
        # check array's type and convert them to deferred arrays
        for a in arrays:
            a = self.runtime.to_deferred_array(a)
            data_type = a.dtype
            if data_type != np.int64:
                raise TypeError("index arrays should be int64 type")
            new_arrays += (a,)
        arrays = new_arrays

        # find a broadcasted shape for all arrays passed as indices
        shapes = tuple(a.shape for a in arrays)
        if len(arrays) > 1:
            from .module import broadcast_shapes

            b_shape = broadcast_shapes(*shapes)
        else:
            b_shape = arrays[0].shape

        # key dim - dimension of indices arrays
        key_dim = len(b_shape)
        out_shape = b_shape

        # broadcast shapes
        new_arrays = tuple()
        for a in arrays:
            if a.shape != b_shape:
                new_arrays += (a._broadcast(b_shape),)
            else:
                new_arrays += (a.base,)
        arrays = new_arrays

        if len(arrays) < self.ndim:
            # the case when # of arrays passed is smaller than dimension of
            # the input array
            N = len(arrays)
            # output shape
            out_shape = (
                tuple(self.shape[i] for i in range(0, start_index))
                + b_shape
                + tuple(
                    self.shape[i] for i in range(start_index + N, self.ndim)
                )
            )
            new_arrays = tuple()
            # promote all index arrays to have the same shape as output
            for a in arrays:
                for i in range(0, start_index):
                    a = a.promote(i, self.shape[i])
                for i in range(start_index + N, self.ndim):
                    a = a.promote(key_dim + i - N, self.shape[i])
                new_arrays += (a,)
            arrays = new_arrays
        elif len(arrays) > self.ndim:
            raise ValueError("wrong number of index arrays passed")

        # create output array which will store Point<N> field where
        # N is number of index arrays
        # shape of the output array should be the same as the shape of each
        # index array
        # NOTE: We need to instantiate a RegionField of non-primitive
        # dtype, to store N-dimensional index points, to be used as the
        # indirection field in a copy.
        N = self.ndim
        pointN_dtype = self.runtime.get_point_type(N)
        output_arr = cast(
            DeferredArray,
            self.runtime.create_empty_thunk(
                shape=out_shape,
                dtype=pointN_dtype,
                inputs=[self],
            ),
        )

        # call ZIP function to combine index arrays into a singe array
        task = self.context.create_auto_task(CuNumericOpCode.ZIP)
        task.throws_exception(IndexError)
        task.add_output(output_arr.base)
        task.add_scalar_arg(self.ndim, ty.int64)  # N of points in Point<N>
        task.add_scalar_arg(key_dim, ty.int64)  # key_dim
        task.add_scalar_arg(start_index, ty.int64)  # start_index
        task.add_scalar_arg(self.shape, (ty.int64,))
        for a in arrays:
            task.add_input(a)
            task.add_alignment(output_arr.base, a)
        task.execute()

        return output_arr

    def _copy_store(self, store: Any) -> DeferredArray:
        store_to_copy = DeferredArray(
            self.runtime,
            base=store,
        )
        store_copy = self.runtime.create_empty_thunk(
            store_to_copy.shape,
            self.base.type,
            inputs=[store_to_copy],
        )
        store_copy.copy(store_to_copy, deep=True)
        return cast(DeferredArray, store_copy)

    @staticmethod
    def _slice_store(k: slice, store: Store, dim: int) -> tuple[slice, Store]:
        start = k.start
        end = k.stop
        step = k.step
        size = store.shape[dim]

        if start is not None:
            if start < 0:
                start += size
            start = max(0, min(size, start))
        if end is not None:
            if end < 0:
                end += size
            end = max(0, min(size, end))
        if (
            (start is not None and start == size)
            or (end is not None and end == 0)
            or (start is not None and end is not None and start >= end)
        ):
            start = 0
            end = 0
            step = 1
        k = slice(start, end, step)

        if start == end and start == 0:  # empty slice
            store = store.project(dim, 0)
            store = store.promote(dim, 0)
        else:
            store = store.slice(dim, k)

        return k, store

    def _has_single_boolean_array(
        self, key: Any, is_set: bool
    ) -> tuple[bool, DeferredArray, Any]:
        if isinstance(key, NumPyThunk) and key.dtype == bool:
            return True, self, key
        else:
            # key is a single array of indices
            if isinstance(key, NumPyThunk):
                return False, self, key

            assert isinstance(key, tuple)

            key = self._unpack_ellipsis(key, self.ndim)

            # loop through all the keys to check if there
            # is a single NumPyThunk entry
            num_arrays = 0
            transpose_index = 0
            for dim, k in enumerate(key):
                if isinstance(k, NumPyThunk):
                    num_arrays += 1
                    transpose_index = dim

            # this is the case when there is a single boolean array passed
            # in this case we transpose original array so that the indx
            # to which boolean array is passed to goes first
            # doing this we can avoid going through Realm Copy which should
            # improve performance
            if (
                num_arrays == 1
                and key[transpose_index].dtype == bool
                and is_set
            ):
                lhs = self
                key_dim = key[transpose_index].ndim
                transpose_indices = tuple(
                    (transpose_index + i) for i in range(0, key_dim)
                )
                transpose_indices += tuple(
                    i for i in range(0, transpose_index)
                )
                transpose_indices += tuple(
                    i for i in range(transpose_index + key_dim, lhs.ndim)
                )

                new_key = tuple(key[i] for i in range(0, transpose_index))
                new_key += tuple(
                    key[i] for i in range(transpose_index + 1, len(key))
                )
                lhs = lhs.transpose(transpose_indices)

                # transform original array for all other keys in the tuple
                if len(new_key) > 0:
                    shift = 0
                    store = lhs.base
                    for dim, k in enumerate(new_key):
                        if np.isscalar(k):
                            if k < 0:  # type: ignore [operator]
                                k += store.shape[dim + key_dim + shift]
                            store = store.project(dim + key_dim + shift, k)
                            shift -= 1
                        elif k is np.newaxis:
                            store = store.promote(dim + key_dim + shift, 1)
                        elif isinstance(k, slice):
                            k, store = self._slice_store(
                                k, store, dim + key_dim + shift
                            )
                        else:
                            raise TypeError(
                                "Unsupported entry type passed to advanced ",
                                "indexing operation",
                            )
                    lhs = DeferredArray(self.runtime, store)

                return True, lhs, key[transpose_index]

            # this is a general advanced indexing case
            else:
                return False, self, key

    def _advanced_indexing_with_boolean_array(
        self,
        key: Any,
        is_set: bool = False,
        set_value: Optional[Any] = None,
    ) -> tuple[bool, Any, Any, Any]:
        rhs = self
        if not isinstance(key, DeferredArray):
            key = self.runtime.to_deferred_array(key)

        # in case when boolean array is passed as an index, shape for all
        # its dimensions should be the same as the shape of
        # corresponding dimensions of the input array
        for i in range(key.ndim):
            if key.shape[i] != rhs.shape[i]:
                raise ValueError(
                    "shape of the index array for "
                    f"dimension {i} doesn't match to the shape of the"
                    f"index array which is {rhs.shape[i]}"
                )

        # if key or rhs are empty, return an empty array with correct shape
        if key.size == 0 or rhs.size == 0:
            if rhs.size == 0 and key.size != 0:
                # we need to calculate shape of the 0 dim of output region
                # even though the size of it is 0
                # this can potentially be replaced with COUNT_NONZERO
                s = key.nonzero()[0].size
            else:
                s = 0

            out_shape = (s,) + tuple(
                rhs.shape[i] for i in range(key.ndim, rhs.ndim)
            )

            out = cast(
                DeferredArray,
                self.runtime.create_empty_thunk(
                    out_shape,
                    rhs.base.type,
                    inputs=[rhs],
                ),
            )
            out.fill(np.zeros((), dtype=out.dtype))
            return False, rhs, out, self

        key_store = key.base
        # bring key to the same shape as rhs
        for i in range(key_store.ndim, rhs.ndim):
            key_store = key_store.promote(i, rhs.shape[i])

        # has_set_value && set_value.size==1 corresponds to the case
        # when a[bool_indices]=scalar
        # then we can call "putmask" to modify input array
        # and avoid calling Copy
        has_set_value = set_value is not None and set_value.size == 1
        if has_set_value:
            mask = DeferredArray(
                self.runtime,
                base=key_store,
            )
            rhs.putmask(mask, set_value)
            return False, rhs, rhs, self
        else:
            out_dtype = rhs.base.type
            # in the case this operation is called for the set_item, we
            # return Point<N> type field that is later used for
            # indirect copy operation
            if is_set:
                N = rhs.ndim
                out_dtype = rhs.runtime.get_point_type(N)

            # TODO : current implementation of the ND output regions
            # requires out.ndim == rhs.ndim. This will be fixed in the
            # future
            out = rhs.runtime.create_unbound_thunk(out_dtype, ndim=rhs.ndim)
            key_dims = key.ndim  # dimension of the original key

            task = rhs.context.create_auto_task(
                CuNumericOpCode.ADVANCED_INDEXING
            )
            task.add_output(out.base)
            task.add_input(rhs.base)
            task.add_input(key_store)
            task.add_scalar_arg(is_set, ty.bool_)
            task.add_scalar_arg(key_dims, ty.int64)
            task.add_alignment(rhs.base, key_store)
            task.add_broadcast(
                rhs.base, axes=tuple(range(1, len(rhs.base.shape)))
            )
            task.execute()

            # TODO : current implementation of the ND output regions
            # requires out.ndim == rhs.ndim.
            # The logic below will be removed in the future
            out_dim = rhs.ndim - key_dims + 1

            if out_dim != rhs.ndim:
                out_tmp = out.base

                if out.size == 0:
                    out_shape = tuple(out.shape[i] for i in range(0, out_dim))
                    out = cast(
                        DeferredArray,
                        self.runtime.create_empty_thunk(
                            out_shape,
                            out_dtype,
                            inputs=[out],
                        ),
                    )
                    if not is_set:
                        out.fill(np.array(0, dtype=out_dtype.to_numpy_dtype()))
                else:
                    for dim in range(rhs.ndim - out_dim):
                        out_tmp = out_tmp.project(rhs.ndim - dim - 1, 0)

                    out = out._copy_store(out_tmp)
            return is_set, rhs, out, self

    def _create_indexing_array(
        self,
        key: Any,
        is_set: bool = False,
        set_value: Optional[Any] = None,
    ) -> tuple[bool, Any, Any, Any]:
        is_bool_array, lhs, bool_key = self._has_single_boolean_array(
            key, is_set
        )

        # the case when single boolean array is passed to the advanced
        # indexing operation
        if is_bool_array:
            return lhs._advanced_indexing_with_boolean_array(
                bool_key, is_set, set_value
            )
        # general advanced indexing case

        store = self.base
        rhs = self
        computed_key: tuple[Any, ...]
        if isinstance(key, NumPyThunk):
            computed_key = (key,)
        else:
            computed_key = key
        assert isinstance(computed_key, tuple)
        computed_key = self._unpack_ellipsis(computed_key, self.ndim)

        # the index where the first index_array is passed to the [] operator
        start_index = -1
        shift = 0
        last_index = self.ndim
        # in case when index arrays are passed in the scattered way,
        # we need to transpose original array so all index arrays
        # are close to each other
        transpose_needed = False
        transpose_indices: NdShape = tuple()
        key_transpose_indices: tuple[int, ...] = tuple()
        tuple_of_arrays: tuple[Any, ...] = ()

        # First, we need to check if transpose is needed
        for dim, k in enumerate(computed_key):
            if np.isscalar(k) or isinstance(k, NumPyThunk):
                if start_index == -1:
                    start_index = dim
                key_transpose_indices += (dim,)
                transpose_needed = transpose_needed or ((dim - last_index) > 1)
                if (
                    isinstance(k, NumPyThunk)
                    and k.dtype == bool
                    and k.ndim >= 2
                ):
                    for i in range(dim, dim + k.ndim):
                        transpose_indices += (shift + i,)
                    shift += k.ndim - 1
                else:
                    transpose_indices += ((dim + shift),)
                last_index = dim

        if transpose_needed:
            start_index = 0
            post_indices = tuple(
                i for i in range(store.ndim) if i not in transpose_indices
            )
            transpose_indices += post_indices
            post_indices = tuple(
                i
                for i in range(len(computed_key))
                if i not in key_transpose_indices
            )
            key_transpose_indices += post_indices
            store = store.transpose(transpose_indices)
            computed_key = tuple(
                computed_key[i] for i in key_transpose_indices
            )

        shift = 0
        for dim, k in enumerate(computed_key):
            if np.isscalar(k):
                if k < 0:  # type: ignore [operator]
                    k += store.shape[dim + shift]  # type: ignore [operator]
                store = store.project(dim + shift, k)
                shift -= 1
            elif k is np.newaxis:
                store = store.promote(dim + shift, 1)
            elif isinstance(k, slice):
                k, store = self._slice_store(k, store, dim + shift)
            elif isinstance(k, NumPyThunk):
                if not isinstance(computed_key, DeferredArray):
                    k = self.runtime.to_deferred_array(k)
                if k.dtype == bool:
                    for i in range(k.ndim):
                        if k.shape[i] != store.shape[dim + i + shift]:
                            raise ValueError(
                                "shape of boolean index did not match "
                                "indexed array "
                            )
                    # in case of the mixed indices we all nonzero
                    # for the boolean array
                    k = k.nonzero()
                    shift += len(k) - 1
                    tuple_of_arrays += k
                else:
                    tuple_of_arrays += (k,)
            else:
                raise TypeError(
                    "Unsupported entry type passed to advanced ",
                    "indexing operation",
                )
        if store.transformed:
            # in the case this operation is called for the set_item, we need
            # to apply all the transformations done to `store` to `self`
            # as well before creating a copy
            if is_set:
                self = DeferredArray(self.runtime, store)
            # after store is transformed we need to to return a copy of
            # the store since Copy operation can't be done on
            # the store with transformation
            rhs = self._copy_store(store)

        if len(tuple_of_arrays) <= rhs.ndim:
            output_arr = rhs._zip_indices(start_index, tuple_of_arrays)
            return True, rhs, output_arr, self
        else:
            raise ValueError("Advanced indexing dimension mismatch")

    @staticmethod
    def _unpack_ellipsis(key: Any, ndim: int) -> tuple[Any, ...]:
        num_ellipsis = sum(k is Ellipsis for k in key)
        num_newaxes = sum(k is np.newaxis for k in key)

        if num_ellipsis == 0:
            return key
        elif num_ellipsis > 1:
            raise ValueError("Only a single ellipsis must be present")

        free_dims = ndim - (len(key) - num_newaxes - num_ellipsis)
        to_replace = (slice(None),) * free_dims
        unpacked: tuple[Any, ...] = ()
        for k in key:
            if k is Ellipsis:
                unpacked += to_replace
            else:
                unpacked += (k,)
        return unpacked

    def _get_view(self, key: Any) -> DeferredArray:
        key = self._unpack_ellipsis(key, self.ndim)
        store = self.base
        shift = 0
        for dim, k in enumerate(key):
            if k is np.newaxis:
                store = store.promote(dim + shift, 1)
            elif isinstance(k, slice):
                k, store = self._slice_store(k, store, dim + shift)
            elif np.isscalar(k):
                if k < 0:  # type: ignore [operator]
                    k += store.shape[dim + shift]  # type: ignore [operator]
                store = store.project(dim + shift, k)
                shift -= 1
            else:
                assert False

        return DeferredArray(
            self.runtime,
            base=store,
        )

    def _broadcast(self, shape: NdShape) -> Any:
        result = self.base
        diff = len(shape) - result.ndim
        for dim in range(diff):
            result = result.promote(dim, shape[dim])

        for dim in range(len(shape)):
            if result.shape[dim] != shape[dim]:
                if result.shape[dim] != 1:
                    raise ValueError(
                        f"Shape did not match along dimension {dim} "
                        "and the value is not equal to 1"
                    )
                result = result.project(dim, 0).promote(dim, shape[dim])

        return result

    def _convert_future_to_regionfield(
        self, change_shape: bool = False
    ) -> DeferredArray:
        if change_shape and self.shape == ():
            shape: NdShape = (1,)
        else:
            shape = self.shape
        store = self.context.create_store(
            self.base.type,
            shape=shape,
            optimize_scalar=False,
        )
        thunk_copy = DeferredArray(
            self.runtime,
            base=store,
        )
        thunk_copy.copy(self, deep=True)
        return thunk_copy

    def get_item(self, key: Any) -> NumPyThunk:
        # Check to see if this is advanced indexing or not
        if is_advanced_indexing(key):
            # Create the indexing array
            (
                copy_needed,
                rhs,
                index_array,
                self,
            ) = self._create_indexing_array(key)

            if copy_needed:
                if rhs.base.kind == Future:
                    rhs = rhs._convert_future_to_regionfield()
                result: NumPyThunk
                if index_array.base.kind == Future:
                    index_array = index_array._convert_future_to_regionfield()
                    result_store = self.context.create_store(
                        self.base.type,
                        shape=index_array.shape,
                        optimize_scalar=False,
                    )
                    result = DeferredArray(
                        self.runtime,
                        base=result_store,
                    )

                else:
                    result = self.runtime.create_empty_thunk(
                        index_array.base.shape,
                        self.base.type,
                        inputs=[self],
                    )

                copy = self.context.create_copy()
                copy.set_source_indirect_out_of_range(False)
                copy.add_input(rhs.base)
                copy.add_source_indirect(index_array.base)
                copy.add_output(result.base)  # type: ignore
                copy.execute()

            else:
                return index_array

        else:
            result = self._get_view(key)

            if result.shape == ():
                input = result
                result = self.runtime.create_empty_thunk(
                    (), self.base.type, inputs=[self]
                )

                task = self.context.create_auto_task(CuNumericOpCode.READ)
                task.add_input(input.base)
                task.add_output(result.base)  # type: ignore

                task.execute()

        return result

    @auto_convert("rhs")
    def set_item(self, key: Any, rhs: Any) -> None:
        assert self.dtype == rhs.dtype
        # Check to see if this is advanced indexing or not
        if is_advanced_indexing(key):
            # Create the indexing array
            (
                copy_needed,
                lhs,
                index_array,
                self,
            ) = self._create_indexing_array(key, True, rhs)

            if not copy_needed:
                return

            if rhs.shape != index_array.shape:
                rhs_tmp = rhs._broadcast(index_array.base.shape)
                rhs_tmp = rhs._copy_store(rhs_tmp)
                rhs_store = rhs_tmp.base
            else:
                if rhs.base.transformed:
                    rhs = rhs._copy_store(rhs.base)
                rhs_store = rhs.base

            # the case when rhs is a scalar and indices array contains
            # a single value
            # TODO this logic should be removed when copy accepts Futures
            if rhs_store.kind == Future:
                rhs_tmp = DeferredArray(
                    self.runtime,
                    base=rhs_store,
                )
                rhs_tmp2 = rhs_tmp._convert_future_to_regionfield()
                rhs_store = rhs_tmp2.base

            if index_array.base.kind == Future:
                index_array = index_array._convert_future_to_regionfield()
            if lhs.base.kind == Future:
                lhs = lhs._convert_future_to_regionfield()
            if lhs.base.transformed:
                lhs = lhs._copy_store(lhs.base)

            if index_array.size != 0:
                copy = self.context.create_copy()
                copy.set_target_indirect_out_of_range(False)

                copy.add_input(rhs_store)
                copy.add_target_indirect(index_array.base)
                copy.add_output(lhs.base)
                copy.execute()

            # TODO this copy will be removed when affine copies are
            # supported in Legion/Realm
            if lhs is not self:
                self.copy(lhs, deep=True)

        else:
            view = self._get_view(key)

            if view.shape == ():
                # We're just writing a single value
                assert rhs.size == 1

                task = self.context.create_auto_task(CuNumericOpCode.WRITE)
                # Since we pass the view with write discard privilege,
                # we should make sure that the mapper either creates a fresh
                # instance just for this one-element view or picks one of the
                # existing valid instances for the parent.
                task.add_output(view.base)
                task.add_input(rhs.base)
                task.execute()
            else:
                # In Python, any inplace update of form arr[key] op= value
                # goes through three steps: 1) __getitem__ fetching the object
                # for the key, 2) __iop__ for the update, and 3) __setitem__
                # to set the result back. In cuNumeric, the object we
                # return in step (1) is actually a subview to the array arr
                # through which we make updates in place, so after step (2) is
                # done, the effect of inplace update is already reflected
                # to the arr. Therefore, we skip the copy to avoid redundant
                # copies if we know that we hit such a scenario.
                # TODO: We should make this work for the advanced indexing case
                # NOTE: Neither Store nor Storage have an __eq__, so we can
                # only check that the underlying RegionField/Future corresponds
                # to the same Legion handle.
                if (
                    view.base.has_storage
                    and rhs.base.has_storage
                    and view.base.storage.same_handle(rhs.base.storage)
                ):
                    return

                view.copy(rhs, deep=False)

    def broadcast_to(self, shape: NdShape) -> NumPyThunk:
        return DeferredArray(self.runtime, base=self._broadcast(shape))

    def reshape(self, newshape: NdShape, order: OrderType) -> NumPyThunk:
        assert isinstance(newshape, Iterable)
        if order == "A":
            order = "C"

        if order != "C":
            # If we don't have a transform then we need to make a copy
            self.runtime.warn(
                "cuNumeric has not implemented reshape using Fortran-like "
                "index order and is falling back to canonical numpy. You may "
                "notice significantly decreased performance for this "
                "function call.",
                category=RuntimeWarning,
            )
            numpy_array = self.__numpy_array__()
            # Force a copy here because we know we can't build a view
            result_array = numpy_array.reshape(newshape, order=order).copy()
            result = self.runtime.get_numpy_thunk(result_array)

            return self.runtime.to_deferred_array(result)

        if self.shape == newshape:
            return self

        # Find a combination of domain transformations to convert this store
        # to the new shape. First we identify a pair of subsets of the source
        # and target extents whose products are the same, and infer necessary
        # domain transformations to align the two. In case where the target
        # isn't a transformed view of the source, the data is copied. This
        # table summarizes five possible cases:
        #
        # +-------+---------+------+-----------------------------------+
        # |Source | Target  | Copy | Plan                              |
        # +-------+---------+------+-----------------------------------+
        # |(a,b,c)| (abc,)  | Yes  | Delinearize(tgt, (a,b,c)) <- src  |
        # +-------+---------+------+-----------------------------------+
        # |(abc,) | (a,b,c,)| No   | tgt = Delinearize(src, (a,b,c))   |
        # +-------+---------+------+-----------------------------------+
        # |(a,b)  | (c,d)   | Yes  | tmp = new store((ab,))            |
        # |       |         |      | Delinearize(tmp, (a,b)) <- src    |
        # |       |         |      | tgt = Delinearize(tmp, (c,d))     |
        # +-------+---------+------+-----------------------------------+
        # |(a,1)  | (a,)    | No   | tgt = Project(src, 0, 0)          |
        # +-------+---------+------+-----------------------------------+
        # |(a,)   | (a,1)   | No   | tgt = Promote(src, 0, 1)          |
        # +-------+---------+------+-----------------------------------+
        #
        # Update 9/22/2021: the non-affineness with delinearization leads
        # to non-contiguous subregions in several places, and thus we
        # decided to avoid using it and make copies instead. This means
        # the third case in the table above now leads to two copies, one from
        # the source to a 1-D temporary array and one from that temporary
        # to the target array. We expect that such reshaping requests are
        # infrequent enough that the extra copies are causing any noticeable
        # performance issues, but we will revisit this decision later once
        # we have enough evidence that that's not the case.

        in_dim = 0
        out_dim = 0

        in_shape = self.shape
        out_shape = newshape

        in_ndim = len(in_shape)
        out_ndim = len(out_shape)

        groups = []

        while in_dim < in_ndim and out_dim < out_ndim:
            prev_in_dim = in_dim
            prev_out_dim = out_dim

            in_prod = 1
            out_prod = 1

            while True:
                if in_prod < out_prod:
                    in_prod *= in_shape[in_dim]
                    in_dim += 1
                else:
                    out_prod *= out_shape[out_dim]
                    out_dim += 1
                if in_prod == out_prod:
                    if in_dim < in_ndim and in_shape[in_dim] == 1:
                        in_dim += 1
                    break

            in_group = in_shape[prev_in_dim:in_dim]
            out_group = out_shape[prev_out_dim:out_dim]
            groups.append((in_group, out_group))

        while in_dim < in_ndim:
            assert in_shape[in_dim] == 1
            groups.append(((1,), ()))
            in_dim += 1

        while out_dim < out_ndim:
            assert out_shape[out_dim] == 1
            groups.append(((), (1,)))
            out_dim += 1

        needs_linearization = any(len(src_g) > 1 for src_g, _ in groups)
        needs_delinearization = any(len(tgt_g) > 1 for _, tgt_g in groups)
        needs_copy = needs_linearization or needs_delinearization

        if needs_copy:
            tmp_shape: NdShape = ()
            for src_g, tgt_g in groups:
                if len(src_g) > 1 and len(tgt_g) > 1:
                    tmp_shape += (_prod(tgt_g),)
                else:
                    tmp_shape += tgt_g

            result = self.runtime.create_empty_thunk(
                tmp_shape, dtype=self.base.type, inputs=[self]
            )

            src = self.base
            tgt = result.base  # type: ignore

            src_dim = 0
            tgt_dim = 0
            for src_g, tgt_g in groups:
                diff = 1
                if src_g == tgt_g:
                    assert len(src_g) == 1
                elif len(src_g) == 0:
                    assert tgt_g == (1,)
                    src = src.promote(src_dim, 1)
                elif len(tgt_g) == 0:
                    assert src_g == (1,)
                    tgt = tgt.promote(tgt_dim, 1)
                elif len(src_g) == 1:
                    src = src.delinearize(src_dim, tgt_g)
                    diff = len(tgt_g)
                else:
                    tgt = tgt.delinearize(tgt_dim, src_g)
                    diff = len(src_g)

                src_dim += diff
                tgt_dim += diff

            assert src.shape == tgt.shape

            src_array = DeferredArray(self.runtime, src)
            tgt_array = DeferredArray(self.runtime, tgt)
            tgt_array.copy(src_array, deep=True)

            if needs_delinearization and needs_linearization:
                src = result.base  # type: ignore
                src_dim = 0
                for src_g, tgt_g in groups:
                    if len(src_g) > 1 and len(tgt_g) > 1:
                        src = src.delinearize(src_dim, tgt_g)
                        src_dim += len(tgt_g)

                assert src.shape == newshape
                src_array = DeferredArray(self.runtime, src)
                result = self.runtime.create_empty_thunk(
                    newshape, dtype=self.base.type, inputs=[self]
                )
                result.copy(src_array, deep=True)

        else:
            src = self.base
            src_dim = 0
            for src_g, tgt_g in groups:
                diff = 1
                if src_g == tgt_g:
                    assert len(src_g) == 1
                elif len(src_g) == 0:
                    assert tgt_g == (1,)
                    src = src.promote(src_dim, 1)
                elif len(tgt_g) == 0:
                    assert src_g == (1,)
                    src = src.project(src_dim, 0)
                    diff = 0
                else:
                    # unreachable
                    assert False

                src_dim += diff

            result = DeferredArray(self.runtime, src)

        return result

    def squeeze(
        self, axis: Optional[Union[int, tuple[int, ...]]]
    ) -> DeferredArray:
        result = self.base
        if axis is None:
            shift = 0
            for dim in range(self.ndim):
                if result.shape[dim + shift] == 1:
                    result = result.project(dim + shift, 0)
                    shift -= 1
        elif isinstance(axis, int):
            result = result.project(axis, 0)
        elif isinstance(axis, tuple):
            shift = 0
            for dim in axis:
                result = result.project(dim + shift, 0)
                shift -= 1
        else:
            raise TypeError(
                '"axis" argument for squeeze must be int-like or tuple-like'
            )
        if result is self.base:
            return self
        return DeferredArray(self.runtime, result)

    def swapaxes(self, axis1: int, axis2: int) -> DeferredArray:
        if self.size == 1 or axis1 == axis2:
            return self
        # Make a new deferred array object and swap the results
        assert axis1 < self.ndim and axis2 < self.ndim

        dims = list(range(self.ndim))
        dims[axis1], dims[axis2] = dims[axis2], dims[axis1]

        result = self.base.transpose(tuple(dims))

        return DeferredArray(self.runtime, result)

    # Convert the source array to the destination array
    @auto_convert("rhs")
    def convert(
        self,
        rhs: Any,
        warn: bool = True,
        nan_op: ConvertCode = ConvertCode.NOOP,
        temporary: bool = False,
    ) -> None:
        lhs_array = self
        rhs_array = rhs
        assert lhs_array.dtype != rhs_array.dtype

        if warn:
            self.runtime.warn(
                "cuNumeric performing implicit type conversion from "
                + str(rhs_array.dtype)
                + " to "
                + str(lhs_array.dtype),
                category=UserWarning,
            )

        lhs = lhs_array.base
        rhs = rhs_array.base

        task = self.context.create_auto_task(CuNumericOpCode.CONVERT)
        task.add_output(lhs)
        task.add_input(rhs)
        task.add_scalar_arg(nan_op, ty.int32)

        task.add_alignment(lhs, rhs)

        task.execute()

        if temporary:
            lhs.set_linear()

    @auto_convert("v", "lhs")
    def convolve(self, v: Any, lhs: Any, mode: ConvolveMode) -> None:
        input = self.base
        filter = v.base
        out = lhs.base

        task = self.context.create_auto_task(CuNumericOpCode.CONVOLVE)

        offsets = (filter.shape + 1) // 2
        stencils: list[tuple[int, ...]] = []
        for offset in offsets:
            stencils.append((-offset, 0, offset))
        stencils = list(product(*stencils))
        stencils.remove((0,) * self.ndim)

        p_out = task.declare_partition(out)
        p_input = task.declare_partition(input)
        p_stencils = []
        for _ in stencils:
            p_stencils.append(task.declare_partition(input, complete=False))

        task.add_output(out, partition=p_out)
        task.add_input(filter)
        task.add_input(input, partition=p_input)
        for p_stencil in p_stencils:
            task.add_input(input, partition=p_stencil)
        task.add_scalar_arg(self.shape, (ty.int64,))

        task.add_constraint(p_out == p_input)
        for stencil, p_stencil in zip(stencils, p_stencils):
            task.add_constraint(p_input + stencil <= p_stencil)  # type: ignore
        task.add_broadcast(filter)

        task.execute()

    @auto_convert("rhs")
    def fft(
        self,
        rhs: Any,
        axes: Sequence[int],
        kind: FFTType,
        direction: FFTDirection,
    ) -> None:
        lhs = self
        # For now, deferred only supported with GPU, use eager / numpy for CPU
        if self.runtime.num_gpus == 0:
            lhs_eager = lhs.runtime.to_eager_array(lhs)
            rhs_eager = rhs.runtime.to_eager_array(rhs)
            lhs_eager.fft(rhs_eager, axes, kind, direction)
            lhs.base = lhs.runtime.to_deferred_array(lhs_eager).base
        else:
            input = rhs.base
            output = lhs.base

            task = self.context.create_auto_task(CuNumericOpCode.FFT)
            p_output = task.declare_partition(output)
            p_input = task.declare_partition(input)

            task.add_output(output, partition=p_output)
            task.add_input(input, partition=p_input)
            task.add_scalar_arg(kind.type_id, ty.int32)
            task.add_scalar_arg(direction.value, ty.int32)
            task.add_scalar_arg(
                len(OrderedSet(axes)) != len(axes)
                or len(axes) != input.ndim
                or tuple(axes) != tuple(sorted(axes)),
                ty.bool_,
            )
            for ax in axes:
                task.add_scalar_arg(ax, ty.int64)

            if input.ndim > len(OrderedSet(axes)):
                task.add_broadcast(input, axes=OrderedSet(axes))
            else:
                task.add_broadcast(input)
            task.add_constraint(p_output == p_input)

            task.execute()

    # Fill the cuNumeric array with the value in the numpy array
    def _fill(self, value: Any) -> None:
        assert value.scalar
        assert self.base is not None

        if self.scalar:
            # Handle the 0D case special
            self.base.set_storage(value.storage)
        elif self.dtype.kind != "V" and self.base.kind is not Future:
            # Emit a Legion fill
            self.context.issue_fill(self.base, value)
        else:
            # Perform the fill using a task
            # If this is a fill for an arg value, make sure to pass
            # the value dtype so that we get it packed correctly
            argval = self.dtype.kind == "V"
            task = self.context.create_auto_task(CuNumericOpCode.FILL)
            task.add_output(self.base)
            task.add_input(value)
            task.add_scalar_arg(argval, ty.bool_)
            task.execute()

    def fill(self, numpy_array: Any) -> None:
        assert isinstance(numpy_array, np.ndarray)
        if numpy_array.size != 1:
            raise ValueError("Filled value array size is not equal to 1")
        assert self.dtype == numpy_array.dtype
        # Have to copy the numpy array because this launch is asynchronous
        # and we need to make sure the application doesn't mutate the value
        # so make a future result, this is immediate so no dependence
        value = self.runtime.create_scalar(numpy_array.data)
        store = self.context.create_store(
            self.base.type, shape=(1,), storage=value, optimize_scalar=True
        )
        self._fill(store)

    @auto_convert("rhs1_thunk", "rhs2_thunk")
    def contract(
        self,
        lhs_modes: list[str],
        rhs1_thunk: Any,
        rhs1_modes: list[str],
        rhs2_thunk: Any,
        rhs2_modes: list[str],
        mode2extent: dict[str, int],
    ) -> None:
        supported_dtypes: list[np.dtype[Any]] = [
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
            np.dtype(np.complex64),
            np.dtype(np.complex128),
        ]
        lhs_thunk: NumPyThunk = self

        # Sanity checks
        # no duplicate modes within an array
        assert len(lhs_modes) == len(OrderedSet(lhs_modes))
        assert len(rhs1_modes) == len(OrderedSet(rhs1_modes))
        assert len(rhs2_modes) == len(OrderedSet(rhs2_modes))
        # no singleton modes
        mode_counts: Counter[str] = Counter()
        mode_counts.update(lhs_modes)
        mode_counts.update(rhs1_modes)
        mode_counts.update(rhs2_modes)
        for count in mode_counts.values():
            assert count == 2 or count == 3
        # arrays and mode lists agree on dimensionality
        assert lhs_thunk.ndim == len(lhs_modes)
        assert rhs1_thunk.ndim == len(rhs1_modes)
        assert rhs2_thunk.ndim == len(rhs2_modes)
        # array shapes agree with mode extents (broadcasting should have been
        # handled by the frontend)
        assert all(
            mode2extent[mode] == dim_sz
            for (mode, dim_sz) in zip(
                lhs_modes + rhs1_modes + rhs2_modes,
                lhs_thunk.shape + rhs1_thunk.shape + rhs2_thunk.shape,
            )
        )
        # casting has been handled by the frontend
        assert lhs_thunk.dtype == rhs1_thunk.dtype
        assert lhs_thunk.dtype == rhs2_thunk.dtype

        # Handle store overlap
        rhs1_thunk = rhs1_thunk._copy_if_overlapping(lhs_thunk)
        rhs2_thunk = rhs2_thunk._copy_if_overlapping(lhs_thunk)

        # Test for special cases where we can use BLAS
        blas_op = None
        if any(c != 2 for c in mode_counts.values()):
            pass
        elif (
            len(lhs_modes) == 0
            and len(rhs1_modes) == 1
            and len(rhs2_modes) == 1
        ):
            # this case works for any arithmetic type, not just floats
            blas_op = BlasOperation.VV
        elif (
            lhs_thunk.dtype in supported_dtypes
            and len(lhs_modes) == 1
            and (
                len(rhs1_modes) == 2
                and len(rhs2_modes) == 1
                or len(rhs1_modes) == 1
                and len(rhs2_modes) == 2
            )
        ):
            blas_op = BlasOperation.MV
        elif (
            lhs_thunk.dtype in supported_dtypes
            and len(lhs_modes) == 2
            and len(rhs1_modes) == 2
            and len(rhs2_modes) == 2
        ):
            blas_op = BlasOperation.MM

        # Our half-precision BLAS tasks expect a single-precision accumulator.
        # This is done to avoid the precision loss that results from repeated
        # reductions into a half-precision accumulator, and to enable the use
        # of tensor cores. In the general-purpose tensor contraction case
        # below the tasks do this adjustment internally.
        if blas_op is not None and lhs_thunk.dtype == np.float16:
            lhs_thunk = self.runtime.create_empty_thunk(
                lhs_thunk.shape, ty.float32, inputs=[lhs_thunk]
            )

        # Clear output array
        lhs_thunk.fill(np.array(0, dtype=lhs_thunk.dtype))

        # Pull out the stores
        lhs = lhs_thunk.base  # type: ignore
        rhs1 = rhs1_thunk.base
        rhs2 = rhs2_thunk.base

        # The underlying libraries are not guaranteed to work with stride
        # values of 0. The frontend should therefore handle broadcasting
        # directly, instead of promoting stores.
        assert not lhs.has_fake_dims()
        assert not rhs1.has_fake_dims()
        assert not rhs2.has_fake_dims()

        # Special cases where we can use BLAS
        if blas_op is not None:
            if blas_op == BlasOperation.VV:
                # Vector dot product
                task = self.context.create_auto_task(CuNumericOpCode.DOT)
                task.add_reduction(lhs, ReductionOp.ADD)
                task.add_input(rhs1)
                task.add_input(rhs2)
                task.add_alignment(rhs1, rhs2)
                task.execute()

            elif blas_op == BlasOperation.MV:
                # Matrix-vector or vector-matrix multiply

                # b,(ab/ba)->a --> (ab/ba),b->a
                if len(rhs1_modes) == 1:
                    rhs1, rhs2 = rhs2, rhs1
                    rhs1_modes, rhs2_modes = rhs2_modes, rhs1_modes
                # ba,b->a --> ab,b->a
                if rhs1_modes[0] == rhs2_modes[0]:
                    rhs1 = rhs1.transpose([1, 0])
                    rhs1_modes = [rhs1_modes[1], rhs1_modes[0]]

                (m, n) = rhs1.shape
                rhs2 = rhs2.promote(0, m)
                lhs = lhs.promote(1, n)

                task = self.context.create_auto_task(CuNumericOpCode.MATVECMUL)
                task.add_reduction(lhs, ReductionOp.ADD)
                task.add_input(rhs1)
                task.add_input(rhs2)
                task.add_alignment(lhs, rhs1)
                task.add_alignment(lhs, rhs2)
                task.execute()

            elif blas_op == BlasOperation.MM:
                # Matrix-matrix multiply

                # (cb/bc),(ab/ba)->ac --> (ab/ba),(cb/bc)->ac
                if lhs_modes[0] not in rhs1_modes:
                    rhs1, rhs2 = rhs2, rhs1
                    rhs1_modes, rhs2_modes = rhs2_modes, rhs1_modes
                assert (
                    lhs_modes[0] in rhs1_modes and lhs_modes[1] in rhs2_modes
                )
                # ba,?->ac --> ab,?->ac
                if lhs_modes[0] != rhs1_modes[0]:
                    rhs1 = rhs1.transpose([1, 0])
                    rhs1_modes = [rhs1_modes[1], rhs1_modes[0]]
                # ?,cb->ac --> ?,bc->ac
                if lhs_modes[1] != rhs2_modes[1]:
                    rhs2 = rhs2.transpose([1, 0])
                    rhs2_modes = [rhs2_modes[1], rhs2_modes[0]]

                m = lhs.shape[0]
                n = lhs.shape[1]
                k = rhs1.shape[1]
                assert m == rhs1.shape[0]
                assert n == rhs2.shape[1]
                assert k == rhs2.shape[0]
                lhs = lhs.promote(1, k)
                rhs1 = rhs1.promote(2, n)
                rhs2 = rhs2.promote(0, m)

                task = self.context.create_auto_task(CuNumericOpCode.MATMUL)
                task.add_reduction(lhs, ReductionOp.ADD)
                task.add_input(rhs1)
                task.add_input(rhs2)
                task.add_alignment(lhs, rhs1)
                task.add_alignment(lhs, rhs2)
                task.execute()

            else:
                assert False

            # If we used a single-precision intermediate accumulator, cast the
            # result back to half-precision.
            if rhs1_thunk.dtype == np.float16:
                self.convert(
                    lhs_thunk,
                    warn=False,
                )

            return

        # General-purpose contraction
        if lhs_thunk.dtype not in supported_dtypes:
            raise TypeError(f"Unsupported type: {lhs_thunk.dtype}")

        # Transpose arrays according to alphabetical order of mode labels
        def alphabetical_transpose(
            store: Store, modes: Sequence[str]
        ) -> Store:
            perm = tuple(
                dim for (_, dim) in sorted(zip(modes, range(len(modes))))
            )
            return store.transpose(perm)

        lhs = alphabetical_transpose(lhs, lhs_modes)
        rhs1 = alphabetical_transpose(rhs1, rhs1_modes)
        rhs2 = alphabetical_transpose(rhs2, rhs2_modes)

        # Promote dimensions as required to align the stores
        lhs_dim_mask: list[bool] = []
        rhs1_dim_mask: list[bool] = []
        rhs2_dim_mask: list[bool] = []
        for dim, mode in enumerate(sorted(mode2extent.keys())):
            extent = mode2extent[mode]

            def add_mode(
                store: Store, modes: Sequence[str], dim_mask: list[bool]
            ) -> Any:
                if mode not in modes:
                    dim_mask.append(False)
                    return store.promote(dim, extent)
                else:
                    dim_mask.append(True)
                    return store

            lhs = add_mode(lhs, lhs_modes, lhs_dim_mask)
            rhs1 = add_mode(rhs1, rhs1_modes, rhs1_dim_mask)
            rhs2 = add_mode(rhs2, rhs2_modes, rhs2_dim_mask)
        assert lhs.shape == rhs1.shape
        assert lhs.shape == rhs2.shape

        # Prepare the launch
        task = self.context.create_auto_task(CuNumericOpCode.CONTRACT)
        task.add_reduction(lhs, ReductionOp.ADD)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_scalar_arg(tuple(lhs_dim_mask), (ty.bool_,))
        task.add_scalar_arg(tuple(rhs1_dim_mask), (ty.bool_,))
        task.add_scalar_arg(tuple(rhs2_dim_mask), (ty.bool_,))
        task.add_alignment(lhs, rhs1)
        task.add_alignment(lhs, rhs2)
        task.execute()

    # Create array from input array and indices
    def choose(self, rhs: Any, *args: Any) -> None:
        # convert all arrays to deferred
        index_arr = self.runtime.to_deferred_array(rhs)
        ch_def = tuple(self.runtime.to_deferred_array(c) for c in args)

        out_arr = self.base
        # broadcast input array and all choices arrays to the same shape
        index = index_arr._broadcast(out_arr.shape.extents)
        ch_tuple = tuple(c._broadcast(out_arr.shape.extents) for c in ch_def)

        task = self.context.create_auto_task(CuNumericOpCode.CHOOSE)
        task.add_output(out_arr)
        task.add_input(index)
        for c in ch_tuple:
            task.add_input(c)

        task.add_alignment(index, out_arr)
        for c in ch_tuple:
            task.add_alignment(index, c)
        task.execute()

    def select(
        self,
        condlist: Iterable[Any],
        choicelist: Iterable[Any],
        default: npt.NDArray[Any],
    ) -> None:
        condlist_ = tuple(self.runtime.to_deferred_array(c) for c in condlist)
        choicelist_ = tuple(
            self.runtime.to_deferred_array(c) for c in choicelist
        )

        task = self.context.create_auto_task(CuNumericOpCode.SELECT)
        out_arr = self.base
        task.add_output(out_arr)
        for c in chain(condlist_, choicelist_):
            c_arr = c._broadcast(self.shape)
            task.add_input(c_arr)
            task.add_alignment(c_arr, out_arr)
        task.add_scalar_arg(default, to_core_dtype(default.dtype))
        task.execute()

    # Create or extract a diagonal from a matrix
    @auto_convert("rhs")
    def _diag_helper(
        self,
        rhs: Any,
        offset: int,
        naxes: int,
        extract: bool,
        trace: bool,
    ) -> None:
        # fill output array with 0
        self.fill(np.array(0, dtype=self.dtype))
        if extract:
            diag = self.base
            matrix = rhs.base
            ndim = rhs.ndim
            start = matrix.ndim - naxes
            n = ndim - 1
            if naxes == 2:
                # get slice of the original array by the offset
                if offset > 0:
                    matrix = matrix.slice(start + 1, slice(offset, None))
                if trace:
                    if matrix.ndim == 2:
                        diag = diag.promote(0, matrix.shape[0])
                        diag = diag.project(1, 0).promote(1, matrix.shape[1])
                    else:
                        for i in range(0, naxes):
                            diag = diag.promote(start, matrix.shape[-i - 1])
                else:
                    if matrix.shape[n - 1] < matrix.shape[n]:
                        diag = diag.promote(start + 1, matrix.shape[ndim - 1])
                    else:
                        diag = diag.promote(start, matrix.shape[ndim - 2])
            else:
                # promote output to the shape of the input  array
                for i in range(1, naxes):
                    diag = diag.promote(start, matrix.shape[-i - 1])
        else:
            matrix = self.base
            diag = rhs.base
            ndim = self.ndim
            # get slice of the original array by the offset
            if offset > 0:
                matrix = matrix.slice(1, slice(offset, None))
            elif offset < 0:
                matrix = matrix.slice(0, slice(-offset, None))

            if matrix.shape[0] < matrix.shape[1]:
                diag = diag.promote(1, matrix.shape[1])
            else:
                diag = diag.promote(0, matrix.shape[0])

        task = self.context.create_auto_task(CuNumericOpCode.DIAG)

        if extract:
            task.add_reduction(diag, ReductionOp.ADD)
            task.add_input(matrix)
            task.add_alignment(matrix, diag)
        else:
            task.add_output(matrix)
            task.add_input(diag)
            task.add_input(matrix)
            task.add_alignment(diag, matrix)

        task.add_scalar_arg(naxes, ty.int32)
        task.add_scalar_arg(extract, ty.bool_)

        task.execute()

    @auto_convert("indices", "values")
    def put(self, indices: Any, values: Any, check_bounds: bool) -> None:
        if indices.base.kind == Future or indices.base.transformed:
            change_shape = indices.base.kind == Future
            indices = indices._convert_future_to_regionfield(change_shape)
        if values.base.kind == Future or values.base.transformed:
            change_shape = values.base.kind == Future
            values = values._convert_future_to_regionfield(change_shape)

        if self.base.kind == Future or self.base.transformed:
            change_shape = self.base.kind == Future
            self_tmp = self._convert_future_to_regionfield(change_shape)
        else:
            self_tmp = self

        assert indices.size == values.size

        # Handle store overlap
        values = values._copy_if_overlapping(self_tmp)

        # first, we create indirect array with PointN type that
        # (indices.size,) shape and is used to copy data from values
        # to the target ND array (self)
        N = self_tmp.ndim
        pointN_dtype = self.runtime.get_point_type(N)
        indirect = cast(
            DeferredArray,
            self.runtime.create_empty_thunk(
                shape=indices.shape,
                dtype=pointN_dtype,
                inputs=[indices],
            ),
        )

        shape = self_tmp.shape
        task = self.context.create_auto_task(CuNumericOpCode.WRAP)
        task.add_output(indirect.base)
        task.add_scalar_arg(shape, (ty.int64,))
        task.add_scalar_arg(True, ty.bool_)  # has_input
        task.add_scalar_arg(check_bounds, ty.bool_)
        task.add_input(indices.base)
        task.add_alignment(indices.base, indirect.base)
        task.throws_exception(IndexError)
        task.execute()
        if indirect.base.kind == Future:
            indirect = indirect._convert_future_to_regionfield()

        copy = self.context.create_copy()
        copy.set_target_indirect_out_of_range(False)
        copy.add_input(values.base)
        copy.add_target_indirect(indirect.base)
        copy.add_output(self_tmp.base)
        copy.execute()

        if self_tmp is not self:
            self.copy(self_tmp, deep=True)

    @auto_convert("mask", "values")
    def putmask(self, mask: Any, values: Any) -> None:
        assert self.shape == mask.shape
        values = values._copy_if_overlapping(self)
        if values.shape != self.shape:
            values_new = values._broadcast(self.shape)
        else:
            values_new = values.base

        task = self.context.create_auto_task(CuNumericOpCode.PUTMASK)
        task.add_input(self.base)
        task.add_input(mask.base)
        task.add_input(values_new)
        task.add_output(self.base)
        task.add_alignment(self.base, mask.base)
        task.add_alignment(self.base, values_new)
        task.execute()

    # Create an identity array with the ones offset from the diagonal by k
    def eye(self, k: int) -> None:
        assert self.ndim == 2  # Only 2-D arrays should be here
        # First issue a fill to zero everything out
        self.fill(np.array(0, dtype=self.dtype))

        # We need to add the store we're filling as an input as well, so we get
        # read-write privileges rather than write-discard. That's because we
        # cannot create tight region requirements that include just the
        # diagonal, so necessarily there will be elements in the region whose
        # values must be carried over from the previous contents. Write-discard
        # privilege, then, is not appropriate for this call, as it essentially
        # tells the runtime that it can throw away the previous contents of the
        # entire region.
        task = self.context.create_auto_task(CuNumericOpCode.EYE)
        task.add_input(self.base)
        task.add_output(self.base)
        task.add_scalar_arg(k, ty.int32)

        task.execute()

    def arange(self, start: float, stop: float, step: float) -> None:
        assert self.ndim == 1  # Only 1-D arrays should be here
        if self.scalar:
            # Handle the special case of a single value here
            assert self.shape[0] == 1
            array = np.array(start, dtype=self.dtype)
            future = self.runtime.create_scalar(array.data)
            self.base.set_storage(future)
            return

        def create_scalar(value: Any, dtype: np.dtype[Any]) -> Any:
            array = np.array(value, dtype)
            return self.runtime.create_wrapped_scalar(
                array.data,
                array.dtype,
                shape=(1,),
            ).base

        task = self.context.create_auto_task(CuNumericOpCode.ARANGE)
        task.add_output(self.base)
        task.add_input(create_scalar(start, self.dtype))
        task.add_input(create_scalar(stop, self.dtype))
        task.add_input(create_scalar(step, self.dtype))

        task.execute()

    # Tile the src array onto the destination array
    @auto_convert("rhs")
    def tile(self, rhs: Any, reps: Union[Any, Sequence[int]]) -> None:
        src_array = rhs
        dst_array = self
        assert src_array.ndim <= dst_array.ndim
        assert src_array.dtype == dst_array.dtype
        if src_array.scalar:
            self._fill(src_array.base)
            return

        task = self.context.create_auto_task(CuNumericOpCode.TILE)

        task.add_output(self.base)
        task.add_input(rhs.base)

        task.add_broadcast(rhs.base)

        task.execute()

    # Transpose the matrix dimensions
    def transpose(
        self, axes: Union[None, tuple[int, ...], list[int]]
    ) -> DeferredArray:
        computed_axes = tuple(axes) if axes is not None else ()
        result = self.base.transpose(computed_axes)
        return DeferredArray(self.runtime, result)

    @auto_convert("rhs")
    def trilu(self, rhs: Any, k: int, lower: bool) -> None:
        lhs = self.base
        rhs = rhs._broadcast(lhs.shape)

        task = self.context.create_auto_task(CuNumericOpCode.TRILU)

        task.add_output(lhs)
        task.add_input(rhs)
        task.add_scalar_arg(lower, ty.bool_)
        task.add_scalar_arg(k, ty.int32)

        task.add_alignment(lhs, rhs)

        task.execute()

    # Repeat elements of an array.
    def repeat(
        self, repeats: Any, axis: int, scalar_repeats: bool
    ) -> DeferredArray:
        out = self.runtime.create_unbound_thunk(self.base.type, ndim=self.ndim)
        task = self.context.create_auto_task(CuNumericOpCode.REPEAT)
        task.add_input(self.base)
        task.add_output(out.base)
        # We pass axis now but don't use for 1D case (will use for ND case
        task.add_scalar_arg(axis, ty.int32)
        task.add_scalar_arg(scalar_repeats, ty.bool_)
        if scalar_repeats:
            task.add_scalar_arg(repeats, ty.int64)
        else:
            shape = self.shape
            repeats = self.runtime.to_deferred_array(repeats).base
            for dim, extent in enumerate(shape):
                if dim == axis:
                    continue
                repeats = repeats.promote(dim, extent)
            task.add_input(repeats)
            task.add_alignment(self.base, repeats)
        task.execute()
        return out

    @auto_convert("rhs")
    def flip(self, rhs: Any, axes: Union[None, int, tuple[int, ...]]) -> None:
        input = rhs.base
        output = self.base

        if axes is None:
            axes = tuple(range(self.ndim))
        else:
            axes = normalize_axis_tuple(axes, self.ndim)

        task = self.context.create_auto_task(CuNumericOpCode.FLIP)
        task.add_output(output)
        task.add_input(input)
        task.add_scalar_arg(axes, (ty.int32,))

        task.add_broadcast(input)
        task.add_alignment(input, output)

        task.execute()

    # Perform a bin count operation on the array
    @auto_convert("rhs", "weights")
    def bincount(self, rhs: Any, weights: Optional[NumPyThunk] = None) -> None:
        weight_array = weights
        src_array = rhs
        dst_array = self
        assert src_array.size > 1
        assert dst_array.ndim == 1
        if weight_array is not None:
            assert src_array.shape == weight_array.shape or (
                src_array.size == 1 and weight_array.size == 1
            )
        else:
            weight_array = self.runtime.create_wrapped_scalar(
                np.array(1, dtype=np.int64),
                np.dtype(np.int64),
                shape=(),
            )

        dst_array.fill(np.array(0, dst_array.dtype))

        task = self.context.create_auto_task(CuNumericOpCode.BINCOUNT)
        task.add_reduction(dst_array.base, ReductionOp.ADD)
        task.add_input(src_array.base)
        task.add_input(weight_array.base)  # type: ignore

        task.add_broadcast(dst_array.base)
        if not weight_array.scalar:
            task.add_alignment(src_array.base, weight_array.base)  # type: ignore  # noqa

        task.execute()

    def nonzero(self) -> tuple[NumPyThunk, ...]:
        results = tuple(
            self.runtime.create_unbound_thunk(ty.int64)
            for _ in range(self.ndim)
        )

        task = self.context.create_auto_task(CuNumericOpCode.NONZERO)

        task.add_input(self.base)
        for result in results:
            task.add_output(result.base)

        task.add_broadcast(self.base, axes=range(1, self.ndim))

        task.execute()
        return results

    def bitgenerator_random_raw(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
    ) -> None:
        task = self.context.create_auto_task(CuNumericOpCode.BITGENERATOR)

        task.add_output(self.base)

        task.add_scalar_arg(BitGeneratorOperation.RAND_RAW, ty.int32)
        task.add_scalar_arg(handle, ty.int32)
        task.add_scalar_arg(generatorType, ty.uint32)
        task.add_scalar_arg(seed, ty.uint64)
        task.add_scalar_arg(flags, ty.uint32)

        # strides
        task.add_scalar_arg(self.compute_strides(self.shape), (ty.int64,))

        task.execute()

    def bitgenerator_distribution(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        distribution: BitGeneratorDistribution,
        intparams: tuple[int, ...],
        floatparams: tuple[float, ...],
        doubleparams: tuple[float, ...],
    ) -> None:
        task = self.context.create_auto_task(CuNumericOpCode.BITGENERATOR)

        task.add_output(self.base)

        task.add_scalar_arg(BitGeneratorOperation.DISTRIBUTION, ty.int32)
        task.add_scalar_arg(handle, ty.int32)
        task.add_scalar_arg(generatorType, ty.uint32)
        task.add_scalar_arg(seed, ty.uint64)
        task.add_scalar_arg(flags, ty.uint32)
        task.add_scalar_arg(distribution, ty.uint32)

        # strides
        task.add_scalar_arg(self.compute_strides(self.shape), (ty.int64,))
        task.add_scalar_arg(intparams, (ty.int64,))
        task.add_scalar_arg(floatparams, (ty.float32,))
        task.add_scalar_arg(doubleparams, (ty.float64,))

        task.execute()

    def bitgenerator_integers(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        low: int,
        high: int,
    ) -> None:
        intparams = (low, high)
        if self.dtype == np.int32:
            distribution = BitGeneratorDistribution.INTEGERS_32
        elif self.dtype == np.int64:
            distribution = BitGeneratorDistribution.INTEGERS_64
        elif self.dtype == np.int16:
            distribution = BitGeneratorDistribution.INTEGERS_16
        else:
            raise NotImplementedError(
                "type for random.integers has to be int64 or int32 or int16"
            )
        self.bitgenerator_distribution(
            handle, generatorType, seed, flags, distribution, intparams, (), ()
        )

    def bitgenerator_uniform(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        low: float,
        high: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.UNIFORM_32
            floatparams = (float(low), float(high))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.UNIFORM_64
            floatparams = ()
            doubleparams = (float(low), float(high))
        else:
            raise NotImplementedError(
                "type for random.uniform has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_lognormal(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mean: float,
        sigma: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.LOGNORMAL_32
            floatparams = (float(mean), float(sigma))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.LOGNORMAL_64
            floatparams = ()
            doubleparams = (float(mean), float(sigma))
        else:
            raise NotImplementedError(
                "type for random.lognormal has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_normal(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mean: float,
        sigma: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.NORMAL_32
            floatparams = (float(mean), float(sigma))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.NORMAL_64
            floatparams = ()
            doubleparams = (float(mean), float(sigma))
        else:
            raise NotImplementedError(
                "type for random.normal has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_poisson(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        lam: float,
    ) -> None:
        if self.dtype == np.uint32:
            distribution = BitGeneratorDistribution.POISSON
            doubleparams = (float(lam),)
        else:
            raise NotImplementedError(
                "type for random.random has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            (),
            doubleparams,
        )

    def bitgenerator_exponential(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        scale: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.EXPONENTIAL_32
            floatparams = (float(scale),)
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.EXPONENTIAL_64
            floatparams = ()
            doubleparams = (float(scale),)
        else:
            raise NotImplementedError(
                "type for random.exponential has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_gumbel(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.GUMBEL_32
            floatparams = (float(mu), float(beta))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.GUMBEL_64
            floatparams = ()
            doubleparams = (float(mu), float(beta))
        else:
            raise NotImplementedError(
                "type for random.gumbel has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_laplace(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.LAPLACE_32
            floatparams = (float(mu), float(beta))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.LAPLACE_64
            floatparams = ()
            doubleparams = (float(mu), float(beta))
        else:
            raise NotImplementedError(
                "type for random.laplace has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_logistic(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.LOGISTIC_32
            floatparams = (float(mu), float(beta))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.LOGISTIC_64
            floatparams = ()
            doubleparams = (float(mu), float(beta))
        else:
            raise NotImplementedError(
                "type for random.logistic has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_pareto(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        alpha: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.PARETO_32
            floatparams = (float(alpha),)
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.PARETO_64
            floatparams = ()
            doubleparams = (float(alpha),)
        else:
            raise NotImplementedError(
                "type for random.pareto has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_power(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        alpha: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.POWER_32
            floatparams = (float(alpha),)
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.POWER_64
            floatparams = ()
            doubleparams = (float(alpha),)
        else:
            raise NotImplementedError(
                "type for random.power has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_rayleigh(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        sigma: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.RAYLEIGH_32
            floatparams = (float(sigma),)
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.RAYLEIGH_64
            floatparams = ()
            doubleparams = (float(sigma),)
        else:
            raise NotImplementedError(
                "type for random.rayleigh has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_cauchy(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        x0: float,
        gamma: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.CAUCHY_32
            floatparams = (float(x0), float(gamma))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.CAUCHY_64
            floatparams = ()
            doubleparams = (float(x0), float(gamma))
        else:
            raise NotImplementedError(
                "type for random.cauchy has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_triangular(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        a: float,
        b: float,
        c: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.TRIANGULAR_32
            floatparams = (float(a), float(b), float(c))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.TRIANGULAR_64
            floatparams = ()
            doubleparams = (float(a), float(b), float(c))
        else:
            raise NotImplementedError(
                "type for random.triangular has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_weibull(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        lam: float,
        k: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.WEIBULL_32
            floatparams = (float(lam), float(k))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.WEIBULL_64
            floatparams = ()
            doubleparams = (float(lam), float(k))
        else:
            raise NotImplementedError(
                "type for random.weibull has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_bytes(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
    ) -> None:
        if self.dtype == np.uint8:
            distribution = BitGeneratorDistribution.BYTES
        else:
            raise NotImplementedError("type for random.bytes has to be uint8")
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            (),
            (),
        )

    def bitgenerator_beta(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        a: float,
        b: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.BETA_32
            floatparams = (float(a), float(b))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.BETA_64
            floatparams = ()
            doubleparams = (float(a), float(b))
        else:
            raise NotImplementedError(
                "type for random.beta has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_f(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        dfnum: float,
        dfden: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.F_32
            floatparams = (float(dfnum), float(dfden))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.F_64
            floatparams = ()
            doubleparams = (float(dfnum), float(dfden))
        else:
            raise NotImplementedError(
                "type for random.beta has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_logseries(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        p: float,
    ) -> None:
        if self.dtype == np.uint32:
            distribution = BitGeneratorDistribution.LOGSERIES
        else:
            raise NotImplementedError("type for random.beta has to be uint32")
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            (),
            (float(p),),
        )

    def bitgenerator_noncentral_f(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        dfnum: float,
        dfden: float,
        nonc: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.NONCENTRAL_F_32
            floatparams = (float(dfnum), float(dfden), float(nonc))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.NONCENTRAL_F_64
            floatparams = ()
            doubleparams = (float(dfnum), float(dfden), float(nonc))
        else:
            raise NotImplementedError(
                "type for random.noncentral_f has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_chisquare(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        df: float,
        nonc: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.CHISQUARE_32
            floatparams = (float(df), float(nonc))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.CHISQUARE_64
            floatparams = ()
            doubleparams = (float(df), float(nonc))
        else:
            raise NotImplementedError(
                "type for random.chisquare has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_gamma(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        k: float,
        theta: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.GAMMA_32
            floatparams = (float(k), float(theta))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.GAMMA_64
            floatparams = ()
            doubleparams = (float(k), float(theta))
        else:
            raise NotImplementedError(
                "type for random.gamma has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_standard_t(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        df: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.STANDARD_T_32
            floatparams = (float(df),)
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.STANDARD_T_64
            floatparams = ()
            doubleparams = (float(df),)
        else:
            raise NotImplementedError(
                "type for random.standard_t has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_hypergeometric(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        ngood: int,
        nbad: int,
        nsample: int,
    ) -> None:
        if self.dtype == np.uint32:
            distribution = BitGeneratorDistribution.HYPERGEOMETRIC
        else:
            raise NotImplementedError(
                "type for random.hypergeometric has to be uint32"
            )
        intparams = (int(ngood), int(nbad), int(nsample))
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            intparams,
            (),
            (),
        )

    def bitgenerator_vonmises(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mu: float,
        kappa: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.VONMISES_32
            floatparams = (float(mu), float(kappa))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.VONMISES_64
            floatparams = ()
            doubleparams = (float(mu), float(kappa))
        else:
            raise NotImplementedError(
                "type for random.vonmises has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_zipf(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        alpha: float,
    ) -> None:
        if self.dtype == np.uint32:
            distribution = BitGeneratorDistribution.ZIPF
            doubleparams = (float(alpha),)
        else:
            raise NotImplementedError("type for random.zipf has to be uint32")
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            (),
            doubleparams,
        )

    def bitgenerator_geometric(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        p: float,
    ) -> None:
        if self.dtype == np.uint32:
            distribution = BitGeneratorDistribution.GEOMETRIC
            doubleparams = (float(p),)
        else:
            raise NotImplementedError(
                "type for random.geometric has to be uint32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            (),
            doubleparams,
        )

    def bitgenerator_wald(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        mean: float,
        scale: float,
    ) -> None:
        floatparams: tuple[float, ...]
        doubleparams: tuple[float, ...]
        if self.dtype == np.float32:
            distribution = BitGeneratorDistribution.WALD_32
            floatparams = (float(mean), float(scale))
            doubleparams = ()
        elif self.dtype == np.float64:
            distribution = BitGeneratorDistribution.WALD_64
            floatparams = ()
            doubleparams = (float(mean), float(scale))
        else:
            raise NotImplementedError(
                "type for random.wald has to be float64 or float32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            (),
            floatparams,
            doubleparams,
        )

    def bitgenerator_binomial(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        ntrials: int,
        p: float,
    ) -> None:
        if self.dtype == np.uint32:
            distribution = BitGeneratorDistribution.BINOMIAL
            intparams = (int(ntrials),)
            doubleparams = (float(p),)
        else:
            raise NotImplementedError(
                "type for random.binomial has to be uint32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            intparams,
            (),
            doubleparams,
        )

    def bitgenerator_negative_binomial(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: Union[int, None],
        flags: int,
        ntrials: int,
        p: float,
    ) -> None:
        if self.dtype == np.uint32:
            distribution = BitGeneratorDistribution.NEGATIVE_BINOMIAL
            intparams = (int(ntrials),)
            doubleparams = (float(p),)
        else:
            raise NotImplementedError(
                "type for random.negative_binomial has to be uint32"
            )
        self.bitgenerator_distribution(
            handle,
            generatorType,
            seed,
            flags,
            distribution,
            intparams,
            (),
            doubleparams,
        )

    def random(self, gen_code: Any, args: Any = ()) -> None:
        task = self.context.create_auto_task(CuNumericOpCode.RAND)

        task.add_output(self.base)
        task.add_scalar_arg(gen_code.value, ty.int32)
        epoch = self.runtime.get_next_random_epoch()
        task.add_scalar_arg(epoch, ty.uint32)
        task.add_scalar_arg(self.compute_strides(self.shape), (ty.int64,))
        self.add_arguments(task, args)

        task.execute()

    def random_uniform(self) -> None:
        assert self.dtype == np.float64
        self.random(RandGenCode.UNIFORM)

    def random_normal(self) -> None:
        assert self.dtype == np.float64
        self.random(RandGenCode.NORMAL)

    def random_integer(
        self,
        low: Union[int, npt.NDArray[Any]],
        high: Union[int, npt.NDArray[Any]],
    ) -> None:
        assert self.dtype.kind == "i"
        low = np.array(low, self.dtype)
        high = np.array(high, self.dtype)
        self.random(RandGenCode.INTEGER, [low, high])

    # Perform the unary operation and put the result in the array
    @auto_convert("src")
    def unary_op(
        self,
        op: UnaryOpCode,
        src: Any,
        where: Any,
        args: Any,
        multiout: Optional[Any] = None,
    ) -> None:
        lhs = self.base
        src = src._copy_if_overlapping(self)
        rhs = src._broadcast(lhs.shape)

        with Annotation({"OpCode": op.name}):
            task = self.context.create_auto_task(CuNumericOpCode.UNARY_OP)
            task.add_output(lhs)
            task.add_input(rhs)
            task.add_scalar_arg(op.value, ty.int32)
            self.add_arguments(task, args)

            task.add_alignment(lhs, rhs)

            if multiout is not None:
                for out in multiout:
                    task.add_output(out.base)
                    task.add_alignment(out.base, rhs)

            task.execute()

    # Perform a unary reduction operation from one set of dimensions down to
    # fewer
    @auto_convert("src", "where")
    def unary_reduction(
        self,
        op: UnaryRedCode,
        src: Any,
        where: Any,
        orig_axis: Union[int, None],
        axes: tuple[int, ...],
        keepdims: bool,
        args: Any,
        initial: Any,
    ) -> None:
        lhs_array: Union[NumPyThunk, DeferredArray] = self
        rhs_array = src
        assert lhs_array.ndim <= rhs_array.ndim

        argred = op in (
            UnaryRedCode.ARGMAX,
            UnaryRedCode.ARGMIN,
            UnaryRedCode.NANARGMAX,
            UnaryRedCode.NANARGMIN,
        )

        if argred:
            argred_dtype = self.runtime.get_argred_type(rhs_array.base.type)
            lhs_array = self.runtime.create_empty_thunk(
                lhs_array.shape,
                dtype=argred_dtype,
                inputs=[self],
            )

        is_where = bool(where is not None)
        # See if we are doing reduction to a point or another region
        if lhs_array.size == 1:
            assert axes is None or lhs_array.ndim == rhs_array.ndim - (
                0 if keepdims else len(axes)
            )

            if initial is not None:
                assert not argred
                fill_value = initial
            else:
                fill_value = _UNARY_RED_IDENTITIES[op](rhs_array.dtype)

            lhs_array.fill(np.array(fill_value, dtype=lhs_array.dtype))

            lhs = lhs_array.base  # type: ignore
            while lhs.ndim > 1:
                lhs = lhs.project(0, 0)

            with Annotation({"OpCode": op.name, "ArgRed?": str(argred)}):
                task = self.context.create_auto_task(
                    CuNumericOpCode.SCALAR_UNARY_RED
                )

                task.add_reduction(lhs, _UNARY_RED_TO_REDUCTION_OPS[op])
                task.add_input(rhs_array.base)
                task.add_scalar_arg(op, ty.int32)
                task.add_scalar_arg(rhs_array.shape, (ty.int64,))
                task.add_scalar_arg(is_where, ty.bool_)
                if is_where:
                    task.add_input(where.base)
                    task.add_alignment(rhs_array.base, where.base)

                self.add_arguments(task, args)

                task.execute()

        else:
            # Before we perform region reduction, make sure to have the lhs
            # initialized. If an initial value is given, we use it, otherwise
            # we use the identity of the reduction operator
            if initial is not None:
                assert not argred
                fill_value = initial
            else:
                fill_value = _UNARY_RED_IDENTITIES[op](rhs_array.dtype)
            lhs_array.fill(np.array(fill_value, lhs_array.dtype))

            # If output dims is not 0, then we must have axes
            assert axes is not None
            # Reduction to a smaller array
            result = lhs_array.base  # type: ignore
            if keepdims:
                for axis in axes:
                    result = result.project(axis, 0)
            for axis in axes:
                result = result.promote(axis, rhs_array.shape[axis])
            # Iterate over all the dimension(s) being collapsed and build a
            # temporary field that we will reduce down to the final value
            if len(axes) > 1:
                raise NotImplementedError(
                    "Need support for reducing multiple dimensions"
                )

            with Annotation({"OpCode": op.name, "ArgRed?": str(argred)}):
                task = self.context.create_auto_task(CuNumericOpCode.UNARY_RED)

                task.add_input(rhs_array.base)
                task.add_reduction(result, _UNARY_RED_TO_REDUCTION_OPS[op])
                task.add_scalar_arg(axis, ty.int32)
                task.add_scalar_arg(op, ty.int32)
                task.add_scalar_arg(is_where, ty.bool_)
                if is_where:
                    task.add_input(where.base)
                    task.add_alignment(rhs_array.base, where.base)

                self.add_arguments(task, args)

                task.add_alignment(result, rhs_array.base)

                task.execute()

        if argred:
            self.unary_op(
                UnaryOpCode.GETARG,
                lhs_array,
                True,
                [],
            )

    def isclose(
        self, rhs1: Any, rhs2: Any, rtol: float, atol: float, equal_nan: bool
    ) -> None:
        assert not equal_nan
        args = (
            np.array(rtol, dtype=np.float64),
            np.array(atol, dtype=np.float64),
        )
        self.binary_op(BinaryOpCode.ISCLOSE, rhs1, rhs2, True, args)

    # Perform the binary operation and put the result in the lhs array
    @auto_convert("src1", "src2")
    def binary_op(
        self,
        op_code: BinaryOpCode,
        src1: Any,
        src2: Any,
        where: Any,
        args: Any,
    ) -> None:
        lhs = self.base
        src1 = src1._copy_if_overlapping(self)
        rhs1 = src1._broadcast(lhs.shape)
        src2 = src2._copy_if_overlapping(self)
        rhs2 = src2._broadcast(lhs.shape)

        with Annotation({"OpCode": op_code.name}):
            # Populate the Legate launcher
            task = self.context.create_auto_task(CuNumericOpCode.BINARY_OP)
            task.add_output(lhs)
            task.add_input(rhs1)
            task.add_input(rhs2)
            task.add_scalar_arg(op_code.value, ty.int32)
            self.add_arguments(task, args)

            task.add_alignment(lhs, rhs1)
            task.add_alignment(lhs, rhs2)

            task.execute()

    @auto_convert("src1", "src2")
    def binary_reduction(
        self,
        op: BinaryOpCode,
        src1: Any,
        src2: Any,
        broadcast: Union[NdShape, None],
        args: Any,
    ) -> None:
        lhs = self.base
        rhs1 = src1.base
        rhs2 = src2.base
        assert lhs.scalar

        if broadcast is not None:
            rhs1 = rhs1._broadcast(broadcast)
            rhs2 = rhs2._broadcast(broadcast)

        # Populate the Legate launcher
        if op == BinaryOpCode.NOT_EQUAL:
            redop = ReductionOp.ADD
            self.fill(np.array(False))
        else:
            redop = ReductionOp.MUL
            self.fill(np.array(True))
        task = self.context.create_auto_task(CuNumericOpCode.BINARY_RED)
        task.add_reduction(lhs, redop)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_scalar_arg(op.value, ty.int32)
        self.add_arguments(task, args)

        task.add_alignment(rhs1, rhs2)

        task.execute()

    @auto_convert("src1", "src2", "src3")
    def where(self, src1: Any, src2: Any, src3: Any) -> None:
        lhs = self.base
        rhs1 = src1._broadcast(lhs.shape)
        rhs2 = src2._broadcast(lhs.shape)
        rhs3 = src3._broadcast(lhs.shape)

        # Populate the Legate launcher
        task = self.context.create_auto_task(CuNumericOpCode.WHERE)
        task.add_output(lhs)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_input(rhs3)

        task.add_alignment(lhs, rhs1)
        task.add_alignment(lhs, rhs2)
        task.add_alignment(lhs, rhs3)

        task.execute()

    def argwhere(self) -> NumPyThunk:
        result = self.runtime.create_unbound_thunk(ty.int64, ndim=2)

        task = self.context.create_auto_task(CuNumericOpCode.ARGWHERE)

        task.add_output(result.base)
        task.add_input(self.base)
        task.add_broadcast(self.base, axes=range(1, self.ndim))

        task.execute()

        return result

    # A helper method for attaching arguments
    def add_arguments(
        self,
        task: Union[AutoTask, ManualTask],
        args: Optional[Sequence[npt.NDArray[Any]]],
    ) -> None:
        if args is None:
            return
        for numpy_array in args:
            assert numpy_array.size == 1
            scalar = self.runtime.create_wrapped_scalar(
                numpy_array.data,
                numpy_array.dtype,
                shape=(1,),
            )
            task.add_input(scalar.base)

    @staticmethod
    def compute_strides(shape: NdShape) -> tuple[int, ...]:
        stride = 1
        result: NdShape = ()
        for dim in reversed(shape):
            result = (stride,) + result
            stride *= dim
        return result

    @auto_convert("src")
    def cholesky(self, src: Any, no_tril: bool = False) -> None:
        cholesky(self, src, no_tril)

    @auto_convert("a", "b")
    def solve(self, a: Any, b: Any) -> None:
        solve(self, a, b)

    @auto_convert("rhs")
    def scan(
        self,
        op: int,
        rhs: Any,
        axis: int,
        dtype: Optional[npt.DTypeLike],
        nan_to_identity: bool,
    ) -> None:
        # local sum
        # storage for local sums accessible
        temp = self.runtime.create_unbound_thunk(
            dtype=self.base.type, ndim=self.ndim
        )

        if axis == rhs.ndim - 1:
            input = rhs
            output = self
        else:
            # swap axes, always performing scan along last axis
            swapped = rhs.swapaxes(axis, rhs.ndim - 1)
            input = self.runtime.create_empty_thunk(
                swapped.shape, dtype=rhs.base.type, inputs=(rhs, swapped)
            )
            input.copy(swapped, deep=True)
            output = input

        task = output.context.create_auto_task(CuNumericOpCode.SCAN_LOCAL)
        task.add_output(output.base)
        task.add_input(input.base)
        task.add_output(temp.base)
        task.add_scalar_arg(op, ty.int32)
        task.add_scalar_arg(nan_to_identity, ty.bool_)

        task.add_alignment(input.base, output.base)

        task.execute()
        # Global sum
        # NOTE: Assumes the partitioning stays the same from previous task.
        # NOTE: Each node will do a sum up to its index, alternatively could
        # do one centralized scan and broadcast (slightly less redundant work)
        task = output.context.create_auto_task(CuNumericOpCode.SCAN_GLOBAL)
        task.add_input(output.base)
        task.add_input(temp.base)
        task.add_output(output.base)
        task.add_scalar_arg(op, ty.int32)

        task.add_broadcast(temp.base)

        task.execute()

        # if axes were swapped, turn them back
        if output is not self:
            swapped = output.swapaxes(rhs.ndim - 1, axis)
            assert self.shape == swapped.shape
            self.copy(swapped, deep=True)

    def unique(self) -> NumPyThunk:
        result = self.runtime.create_unbound_thunk(self.base.type)

        task = self.context.create_auto_task(CuNumericOpCode.UNIQUE)

        task.add_output(result.base)
        task.add_input(self.base)

        if self.runtime.num_gpus > 0:
            task.add_nccl_communicator()

        task.execute()

        if self.runtime.num_gpus == 0 and self.runtime.num_procs > 1:
            result.base = self.context.tree_reduce(
                CuNumericOpCode.UNIQUE_REDUCE, result.base
            )

        return result

    @auto_convert("rhs", "v")
    def searchsorted(self, rhs: Any, v: Any, side: SortSide = "left") -> None:
        task = self.context.create_auto_task(CuNumericOpCode.SEARCHSORTED)

        is_left = side == "left"

        if is_left:
            self.fill(np.array(rhs.size, self.dtype))
            task.add_reduction(self.base, ReductionOp.MIN)
        else:
            self.fill(np.array(0, self.dtype))
            task.add_reduction(self.base, ReductionOp.MAX)

        task.add_input(rhs.base)
        task.add_input(v.base)

        # every partition needs the value information
        task.add_broadcast(v.base)
        task.add_broadcast(self.base)
        task.add_alignment(self.base, v.base)

        task.add_scalar_arg(is_left, ty.bool_)
        task.add_scalar_arg(rhs.size, ty.int64)
        task.execute()

    @auto_convert("rhs")
    def sort(
        self,
        rhs: Any,
        argsort: bool = False,
        axis: Union[int, None] = -1,
        kind: SortType = "quicksort",
        order: Union[None, str, list[str]] = None,
    ) -> None:
        if kind == "stable":
            stable = True
        else:
            stable = False

        if order is not None:
            raise NotImplementedError(
                "cuNumeric does not support sorting with 'order' as "
                "ndarray only supports numeric values"
            )
        if axis is not None and (axis >= rhs.ndim or axis < -rhs.ndim):
            raise ValueError("invalid axis")

        sort(self, rhs, argsort, axis, stable)

    @auto_convert("rhs")
    def partition(
        self,
        rhs: Any,
        kth: Union[int, Sequence[int]],
        argpartition: bool = False,
        axis: Union[int, None] = -1,
        kind: SelectKind = "introselect",
        order: Union[None, str, list[str]] = None,
    ) -> None:
        if order is not None:
            raise NotImplementedError(
                "cuNumeric does not support partitioning with 'order' as "
                "ndarray only supports numeric values"
            )
        if axis is not None and (axis >= rhs.ndim or axis < -rhs.ndim):
            raise ValueError("invalid axis")

        # fallback to sort for now
        sort(self, rhs, argpartition, axis, False)

    def create_window(self, op_code: WindowOpCode, M: int, *args: Any) -> None:
        task = self.context.create_auto_task(CuNumericOpCode.WINDOW)
        task.add_output(self.base)
        task.add_scalar_arg(op_code, ty.int32)
        task.add_scalar_arg(M, ty.int64)
        for arg in args:
            task.add_scalar_arg(arg, ty.float64)
        task.execute()

    @auto_convert("src")
    def packbits(
        self, src: Any, axis: Union[int, None], bitorder: BitOrder
    ) -> None:
        bitorder_code = getattr(Bitorder, bitorder.upper())
        task = self.context.create_auto_task(CuNumericOpCode.PACKBITS)
        p_out = task.declare_partition(self.base)
        p_in = task.declare_partition(src.base)
        task.add_output(self.base, partition=p_out)
        task.add_input(src.base, partition=p_in)
        task.add_scalar_arg(axis, ty.uint32)
        task.add_scalar_arg(bitorder_code, ty.uint32)
        scale = tuple(8 if dim == axis else 1 for dim in range(src.ndim))
        task.add_constraint(p_in <= p_out * scale)  # type: ignore
        task.execute()

    @auto_convert("src")
    def unpackbits(
        self, src: Any, axis: Union[int, None], bitorder: BitOrder
    ) -> None:
        bitorder_code = getattr(Bitorder, bitorder.upper())
        task = self.context.create_auto_task(CuNumericOpCode.UNPACKBITS)
        p_out = task.declare_partition(self.base)
        p_in = task.declare_partition(src.base)
        task.add_output(self.base, partition=p_out)
        task.add_input(src.base, partition=p_in)
        task.add_scalar_arg(axis, ty.uint32)
        task.add_scalar_arg(bitorder_code, ty.uint32)
        scale = tuple(8 if dim == axis else 1 for dim in range(src.ndim))
        task.add_constraint(p_out <= p_in * scale)  # type: ignore
        task.execute()

    @auto_convert("src")
    def _wrap(self, src: Any, new_len: int) -> None:
        if src.base.kind == Future or src.base.transformed:
            change_shape = src.base.kind == Future
            src = src._convert_future_to_regionfield(change_shape)

        # first, we create indirect array with PointN type that
        # (len,) shape and is used to copy data from original array
        # to the target 1D wrapped array
        N = src.ndim
        pointN_dtype = self.runtime.get_point_type(N)
        indirect = cast(
            DeferredArray,
            self.runtime.create_empty_thunk(
                shape=(new_len,),
                dtype=pointN_dtype,
                inputs=[src],
            ),
        )

        task = self.context.create_auto_task(CuNumericOpCode.WRAP)
        task.add_output(indirect.base)
        task.add_scalar_arg(src.shape, (ty.int64,))
        task.add_scalar_arg(False, ty.bool_)  # has_input
        task.add_scalar_arg(False, ty.bool_)  # check bounds
        task.execute()

        copy = self.context.create_copy()
        copy.set_target_indirect_out_of_range(False)
        copy.add_input(src.base)
        copy.add_source_indirect(indirect.base)
        copy.add_output(self.base)
        copy.execute()

    # Perform a histogram operation on the array
    @auto_convert("src", "bins", "weights")
    def histogram(self, src: Any, bins: Any, weights: Any) -> None:
        weight_array = weights
        src_array = src
        bins_array = bins
        dst_array = self
        assert src_array.size > 0
        assert dst_array.ndim == 1
        assert (
            (len(src_array.shape) == 1)
            and (len(weight_array.shape) == 1)
            and (src_array.size == weight_array.size)
        )

        dst_array.fill(np.array(0, dst_array.dtype))

        task = self.context.create_auto_task(CuNumericOpCode.HISTOGRAM)
        task.add_reduction(dst_array.base, ReductionOp.ADD)
        task.add_input(src_array.base)
        task.add_input(bins_array.base)
        task.add_input(weight_array.base)

        task.add_broadcast(bins_array.base)
        task.add_broadcast(dst_array.base)
        task.add_alignment(src_array.base, weight_array.base)

        task.execute()
