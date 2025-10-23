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

import operator
from functools import reduce
from math import prod as builtin_prod
from typing import TYPE_CHECKING, Any, Sequence, cast

import legate.core.types as ty
import numpy as np
from legate.core import Field, LogicalArray, Scalar
from legate.core.utils import OrderedSet
from numpy.exceptions import AxisError

from .. import _ufunc
from .._utils import is_np2
from .._utils.array import max_identity, min_identity, to_core_type
from .._utils.coverage import (
    clone_class,
    is_implemented,
    issue_fallback_warning,
)
from .._utils.linalg import dot_modes
from .._utils.structure import deep_apply
from ..config import (
    FFTDirection,
    FFTNormalization,
    FFTType,
    ScanCode,
    TransferType,
    UnaryOpCode,
    UnaryRedCode,
)
from ..lib.array_utils import normalize_axis_index, normalize_axis_tuple
from ..runtime import runtime
from .flags import flagsobj
from .thunk import perform_scan, perform_unary_op, perform_unary_reduction
from .util import (
    add_boilerplate,
    broadcast_where,
    check_writeable,
    convert_to_cupynumeric_ndarray,
    maybe_convert_to_np_ndarray,
    sanitize_shape,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    from typing_extensions import CapsuleType

    from .._thunk.thunk import NumPyThunk
    from ..types import (
        BoundsMode,
        CastingKind,
        OrderType,
        NdShape,
        NdShapeLike,
        SelectKind,
        SortSide,
        SortType,
    )

    import sys
    import mmap
    from numpy import generic

    if sys.version_info >= (3, 12):
        from collections.abc import Buffer
    else:
        Buffer = (
            bytes
            | bytearray
            | memoryview
            | npt.NDArray[Any]
            | mmap.mmap
            | generic
        )

from math import prod

NDARRAY_INTERNAL = {
    "__array_finalize__",
    "__array_function__",
    "__array_interface__",
    "__array_prepare__",
    "__array_priority__",
    "__array_struct__",
    "__array_ufunc__",
    "__array_wrap__",
    # Avoid auto-wrapping Array API specifics:
    "__array_namespace__",
    "device",
    "to_device",
}


def _warn_and_convert(array: ndarray, dtype: np.dtype[Any]) -> ndarray:
    if array.dtype != dtype:
        runtime.warn(
            f"converting array to {dtype} type", category=RuntimeWarning
        )
        return array.astype(dtype)
    else:
        return array


@clone_class(np.ndarray, NDARRAY_INTERNAL, maybe_convert_to_np_ndarray)
class ndarray:
    _thunk: NumPyThunk
    _legate_data: dict[str, Any] | None

    def __init__(
        self,
        shape: NdShapeLike,
        dtype: npt.DTypeLike = np.float64,
        buffer: Buffer | None = None,
        offset: int = 0,
        strides: tuple[int, ...] | None = None,
        order: OrderType | None = None,
        *,
        writeable: bool = True,
    ) -> None:
        assert shape is not None

        if buffer is not None:
            self._init_from_buffer(
                shape=shape,
                dtype=dtype,
                buffer=buffer,
                offset=offset,
                strides=strides,
                order=order,
                writeable=writeable,
            )

        else:
            self._init_from_inputs(
                shape=shape, dtype=dtype, inputs=(), writeable=writeable
            )

    @staticmethod
    def _from_thunk(thunk: NumPyThunk, *, writeable: bool = True) -> ndarray:
        array: ndarray = object.__new__(ndarray)
        array._init_from_thunk(thunk, writeable=writeable)
        return array

    @staticmethod
    def _from_buffer(
        shape: NdShapeLike,
        dtype: npt.DTypeLike,
        buffer: Buffer,
        offset: int,
        strides: tuple[int, ...] | None,
        order: OrderType | None,
        *,
        writeable: bool = True,
    ) -> ndarray:
        array: ndarray = object.__new__(ndarray)
        array._init_from_buffer(
            shape=shape,
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order,
            writeable=writeable,
        )
        return array

    @staticmethod
    def _from_inputs(
        shape: NdShapeLike,
        dtype: npt.DTypeLike = np.float64,
        inputs: Sequence[Any] = (),
        *,
        writeable: bool = True,
    ) -> ndarray:
        array: ndarray = object.__new__(ndarray)
        array._init_from_inputs(shape, dtype, inputs, writeable=writeable)
        return array

    def _init_from_thunk(
        self, thunk: NumPyThunk, *, writeable: bool = True
    ) -> None:
        self._thunk = thunk
        self._legate_data = None
        self._writeable = writeable

    def _init_from_buffer(
        self,
        shape: Any,
        dtype: npt.DTypeLike,
        buffer: Buffer,
        offset: int,
        strides: tuple[int, ...] | None,
        order: OrderType | None,
        *,
        writeable: bool = True,
    ) -> None:
        arr = np.ndarray(
            shape=sanitize_shape(shape),
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order,
        )

        thunk = runtime.find_or_create_array_thunk(arr, TransferType.SHARE)

        self._init_from_thunk(thunk, writeable=writeable)

    def _init_from_inputs(
        self,
        shape: NdShapeLike,
        dtype: npt.DTypeLike,
        inputs: Sequence[Any],
        *,
        writeable: bool = True,
    ) -> None:
        from .._thunk.thunk import NumPyThunk

        shape_tuple = sanitize_shape(shape)

        filtered_inputs: list[NumPyThunk] = []
        for inp in inputs:
            if isinstance(inp, ndarray):
                filtered_inputs.append(inp._thunk)
            elif isinstance(inp, NumPyThunk):
                filtered_inputs.append(inp)

        core_dtype = to_core_type(np.dtype(dtype))
        eager_shape = runtime.is_eager_shape(shape_tuple)
        eager_inputs = runtime.are_all_eager_inputs(filtered_inputs)

        if eager_shape and eager_inputs:
            thunk = runtime.create_eager_thunk(
                shape_tuple, core_dtype.to_numpy_dtype()
            )
            self._init_from_thunk(thunk, writeable=writeable)
        else:
            thunk = runtime.create_deferred_thunk(shape_tuple, core_dtype)
            self._init_from_thunk(thunk, writeable=writeable)

    # Support for the Legate data interface
    @property
    def __legate_data_interface__(self) -> dict[str, Any]:
        if self._legate_data is None:
            # If the thunk is an eager array, we need to convert it to a
            # deferred array so we can extract a legate store
            deferred_thunk = runtime.to_deferred_array(
                self._thunk, read_only=False
            )
            # We don't have nullable data for the moment
            # until we support masked arrays
            dtype = deferred_thunk.base.type
            array = LogicalArray.from_store(deferred_thunk.base)
            self._legate_data = dict()
            self._legate_data["version"] = 1
            field = Field("cuPyNumeric Array", dtype)
            self._legate_data["data"] = {field: array}
        return self._legate_data

    def __dlpack__(
        self,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType:
        return self._thunk.__dlpack__(
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._thunk.__dlpack_device__()

    # Properties for ndarray

    # Disable these since they seem to cause problems
    # when our arrays do not last long enough, instead
    # users will go through the __array__ method

    # @property
    # def __array_interface__(self):
    #    return self.__array__().__array_interface__

    # @property
    # def __array_priority__(self):
    #    return self.__array__().__array_priority__

    # @property
    # def __array_struct__(self):
    #    return self.__array__().__array_struct__

    def __array_function__(
        self, func: Any, types: Any, args: tuple[Any], kwargs: dict[str, Any]
    ) -> Any:
        import cupynumeric as cn

        what = func.__name__

        for t in types:
            # Be strict about which types we support.  Accept superclasses
            # (for basic subclassing support) and NumPy.
            if not issubclass(type(self), t) and t is not np.ndarray:
                return NotImplemented

        # We are wrapping all NumPy modules, so we can expect to find every
        # NumPy API call in cuPyNumeric, even if just an "unimplemented" stub.
        module = reduce(getattr, func.__module__.split(".")[1:], cn)
        cn_func = getattr(module, func.__name__)

        # We can't immediately forward to the corresponding cuPyNumeric
        # entrypoint. Say that we reached this point because the user code
        # invoked `np.foo(x, bar=True)` where `x` is a `cupynumeric.ndarray`.
        # If our implementation of `foo` is not complete, and cannot handle
        # `bar=True`, then forwarding this call to `cn.foo` would fail. This
        # goes against the semantics of `__array_function__`, which shouldn't
        # fail if the custom implementation cannot handle the provided
        # arguments. Conversely, if the user calls `cn.foo(x, bar=True)`
        # directly, that means they requested the cuPyNumeric implementation
        # specifically, and the `NotImplementedError` should not be hidden.
        if is_implemented(cn_func):
            try:
                return cn_func(*args, **kwargs)
            except NotImplementedError:
                # Inform the user that we support the requested API in general,
                # but not this specific combination of arguments.
                what = f"the requested combination of arguments to {what}"

        # We cannot handle this call, so we will fall back to NumPy.
        issue_fallback_warning(what=what)
        args = deep_apply(args, maybe_convert_to_np_ndarray)
        kwargs = deep_apply(kwargs, maybe_convert_to_np_ndarray)
        return func(*args, **kwargs)

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        from .. import _ufunc

        # Check whether we should handle the arguments
        array_args = inputs
        array_args += kwargs.get("out", ())
        if (where := kwargs.get("where", True)) is not True:
            array_args += (where,)

        for arg in array_args:
            if not hasattr(arg, "__array_ufunc__"):
                continue

            t = type(arg)
            # Reject arguments we do not know (see __array_function__)
            if not issubclass(type(self), t) and t is not np.ndarray:
                return NotImplemented

        # TODO: The logic below should be moved to a "clone_ufunc" wrapper.

        what = f"{ufunc.__name__}.{method}"

        # special case for @ matmul
        if what == "matmul.__call__":
            x = convert_to_cupynumeric_ndarray(inputs[0])
            return x._thunk._matmul(*inputs[1:], **kwargs)

        if hasattr(_ufunc, ufunc.__name__):
            cn_ufunc = getattr(_ufunc, ufunc.__name__)
            if hasattr(cn_ufunc, method):
                cn_method = getattr(cn_ufunc, method)
                # Similar to __array_function__, we need to gracefully fall
                # back to NumPy if we can't handle the provided combination of
                # arguments.
                try:
                    return cn_method(*inputs, **kwargs)
                except NotImplementedError:
                    what = f"the requested combination of arguments to {what}"

        # We cannot handle this ufunc call, so we will fall back to NumPy.
        issue_fallback_warning(what=what)
        inputs = deep_apply(inputs, maybe_convert_to_np_ndarray)
        kwargs = deep_apply(kwargs, maybe_convert_to_np_ndarray)
        return getattr(ufunc, method)(*inputs, **kwargs)

    @property
    def T(self) -> ndarray:
        """

        The transposed array.

        Same as ``self.transpose()``.

        See Also
        --------
        cupynumeric.transpose
        ndarray.transpose

        """
        return self.transpose()

    @property
    def base(self) -> npt.NDArray[Any] | None:
        """
        Base object if memory is from some other object.
        """
        raise NotImplementedError(
            "cupynumeric.ndarray doesn't keep track of the array view hierarchy yet"
        )

    @property
    def data(self) -> memoryview:
        """
        Python buffer object pointing to the start of the array's data.

        Notes
        -----
        This causes the entire (potentially distributed) array to be collected
        into one memory.
        """
        return self.__array__().data

    def __buffer__(self, flags: int, /) -> memoryview:
        """
        Python buffer object pointing to the start of the array's data.

        Notes
        -----
        This causes the entire (potentially distributed) array to be collected
        into one memory.
        """
        return self.__array__().__buffer__(flags)  # type: ignore

    @property
    def dtype(self) -> np.dtype[Any]:
        """
        Data-type of the array's elements.

        See Also
        --------
        astype : Cast the values contained in the array to a new data-type.
        view : Create a view of the same data but a different data-type.
        numpy.dtype

        """
        return self._thunk.dtype

    @property
    def flags(self) -> Any:
        """
        Information about the memory layout of the array.

        These flags don't reflect the properties of the cuPyNumeric array, but
        rather the NumPy array that will be produced if the cuPyNumeric array
        is materialized on a single node.

        Attributes
        ----------
        C_CONTIGUOUS (C)
            The data is in a single, C-style contiguous segment.
        F_CONTIGUOUS (F)
            The data is in a single, Fortran-style contiguous segment.
        OWNDATA (O)
            The array owns the memory it uses or borrows it from another
            object.
        WRITEABLE (W)
            The data area can be written to.  Setting this to False locks
            the data, making it read-only.  A view (slice, etc.) inherits
            WRITEABLE from its base array at creation time, but a view of a
            writeable array may be subsequently locked while the base array
            remains writeable. (The opposite is not true, in that a view of a
            locked array may not be made writeable.  However, currently,
            locking a base object does not lock any views that already
            reference it, so under that circumstance it is possible to alter
            the contents of a locked array via a previously created writeable
            view onto it.)  Attempting to change a non-writeable array raises
            a RuntimeError exception.
        ALIGNED (A)
            The data and all elements are aligned appropriately for the
            hardware.
        WRITEBACKIFCOPY (X)
            This array is a copy of some other array. The C-API function
            PyArray_ResolveWritebackIfCopy must be called before deallocating
            to the base array will be updated with the contents of this array.
        FNC
            F_CONTIGUOUS and not C_CONTIGUOUS.
        FORC
            F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).
        BEHAVED (B)
            ALIGNED and WRITEABLE.
        CARRAY (CA)
            BEHAVED and C_CONTIGUOUS.
        FARRAY (FA)
            BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.

        Notes
        -----
        The `flags` object can be accessed dictionary-like (as in
        ``a.flags['WRITEABLE']``), or by using lowercased attribute names (as
        in ``a.flags.writeable``). Short flag names are only supported in
        dictionary access.

        Only the WRITEBACKIFCOPY, WRITEABLE, and ALIGNED flags can be
        changed by the user, via direct assignment to the attribute or
        dictionary entry, or by calling `ndarray.setflags`.

        The array flags cannot be set arbitrarily:
        - WRITEBACKIFCOPY can only be set ``False``.
        - ALIGNED can only be set ``True`` if the data is truly aligned.
        - WRITEABLE can only be set ``True`` if the array owns its own memory
        or the ultimate owner of the memory exposes a writeable buffer
        interface or is a string.

        Arrays can be both C-style and Fortran-style contiguous
        simultaneously. This is clear for 1-dimensional arrays, but can also
        be true for higher dimensional arrays.

        Even for contiguous arrays a stride for a given dimension
        ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``
        or the array has no elements.
        It does not generally hold that ``self.strides[-1] == self.itemsize``
        for C-style contiguous arrays or ``self.strides[0] == self.itemsize``
        for Fortran-style contiguous arrays is true.
        """
        return flagsobj(self)

    @property
    def flat(self) -> np.flatiter[npt.NDArray[Any]]:
        """
        A 1-D iterator over the array.

        See Also
        --------
        flatten : Return a copy of the array collapsed into one dimension.

        Availability
        ------------
        Single CPU

        """
        return self.__array__().flat

    @property
    def imag(self) -> ndarray:
        """
        The imaginary part of the array.

        """
        if self.dtype.kind == "c":
            return ndarray._from_thunk(self._thunk.imag())
        else:
            result = ndarray(self.shape, self.dtype)
            result.fill(0)
            return result

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions.

        """
        return self._thunk.ndim

    @property
    def real(self) -> ndarray:
        """

        The real part of the array.

        """
        if self.dtype.kind == "c":
            return ndarray._from_thunk(self._thunk.real())
        else:
            return self

    @property
    def shape(self) -> NdShape:
        """

        Tuple of array dimensions.

        See Also
        --------
        shape : Equivalent getter function.
        reshape : Function forsetting ``shape``.
        ndarray.reshape : Method for setting ``shape``.

        """
        return self._thunk.shape

    @property
    def size(self) -> int:
        """

        Number of elements in the array.

        Equal to ``np.prod(a.shape)``, i.e., the product of the array's
        dimensions.

        Notes
        -----
        `a.size` returns a standard arbitrary precision Python integer. This
        may not be the case with other methods of obtaining the same value
        (like the suggested ``np.prod(a.shape)``, which returns an instance
        of ``np.int_``), and may be relevant if the value is used further in
        calculations that may overflow a fixed size integer type.

        """
        s = 1
        if self.ndim == 0:
            return s
        for p in self.shape:
            s *= p
        return s

    @property
    def itemsize(self) -> int:
        """

        The element size of this data-type object.

        For 18 of the 21 types this number is fixed by the data-type.
        For the flexible data-types, this number can be anything.

        """
        return self._thunk.dtype.itemsize

    @property
    def nbytes(self) -> int:
        """

        Total bytes consumed by the elements of the array.

        Notes
        -----
        Does not include memory consumed by non-element attributes of the
        array object.

        """
        return self.itemsize * self.size

    @property
    def strides(self) -> tuple[int, ...]:
        """

        Tuple of bytes to step in each dimension when traversing an array.

        The byte offset of element ``(i[0], i[1], ..., i[n])`` in an array
        `a` is::

            offset = sum(np.array(i) * a.strides)

        A more detailed explanation of strides can be found in the
        "ndarray.rst" file in the NumPy reference guide.

        Notes
        -----
        Imagine an array of 32-bit integers (each 4 bytes)::

            x = np.array([[0, 1, 2, 3, 4],
                         [5, 6, 7, 8, 9]], dtype=np.int32)

        This array is stored in memory as 40 bytes, one after the other
        (known as a contiguous block of memory).  The strides of an array tell
        us how many bytes we have to skip in memory to move to the next
        position along a certain axis.  For example, we have to skip 4 bytes
        (1 value) to move to the next column, but 20 bytes (5 values) to get
        to the same position in the next row.  As such, the strides for the
        array `x` will be ``(20, 4)``.

        """
        return self.__array__().strides

    @property
    def ctypes(self) -> Any:
        """

        An object to simplify the interaction of the array with the ctypes
        module.

        This attribute creates an object that makes it easier to use arrays
        when calling shared libraries with the ctypes module. The returned
        object has, among others, data, shape, and strides attributes (see
        :external+numpy:attr:`numpy.ndarray.ctypes` for details) which
        themselves return ctypes objects that can be used as arguments to a
        shared library.

        Parameters
        ----------
        None

        Returns
        -------
        c : object
            Possessing attributes data, shape, strides, etc.

        """
        return self.__array__().ctypes

    # Methods for ndarray

    def __abs__(self) -> ndarray:
        """a.__abs__(/)

        Return ``abs(self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # Handle the nice case of it being unsigned
        return convert_to_cupynumeric_ndarray(self._thunk._absolute())

    def __add__(self, rhs: Any) -> ndarray:
        """a.__add__(value, /)

        Return ``self+value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._add(rhs))

    def __and__(self, rhs: Any) -> ndarray:
        """a.__and__(value, /)

        Return ``self&value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._bitwise_and(rhs))

    def __array__(
        self, dtype: np.dtype[Any] | None = None
    ) -> npt.NDArray[Any]:
        """a.__array__([dtype], /)

        Returns either a new reference to self if dtype is not given or a new
        array of provided data type if dtype is different from the current
        dtype of the array.

        """
        numpy_array = self._thunk.__numpy_array__()
        if numpy_array.flags.writeable and not self._writeable:
            numpy_array.flags.writeable = False
        if dtype is not None:
            numpy_array = numpy_array.astype(dtype)
        return numpy_array

    # def __array_prepare__(self, *args, **kwargs):
    #    return self.__array__().__array_prepare__(*args, **kwargs)

    # def __array_wrap__(self, *args, **kwargs):
    #    return self.__array__().__array_wrap__(*args, **kwargs)

    def __bool__(self) -> bool:
        """a.__bool__(/)

        Return ``self!=0``

        """
        return bool(self.__array__())

    def __complex__(self) -> complex:
        """a.__complex__(/)"""
        return complex(self.__array__())

    def __contains__(self, item: Any) -> ndarray:
        """a.__contains__(key, /)

        Return ``key in self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if isinstance(item, np.ndarray):
            args = (item.astype(self.dtype),)
        else:  # Otherwise convert it to a scalar numpy array of our type
            args = (np.array(item, dtype=self.dtype),)
        if args[0].size != 1:
            raise ValueError("contains needs scalar item")
        core_dtype = to_core_type(self.dtype)
        return perform_unary_reduction(
            UnaryRedCode.CONTAINS,
            self,
            axis=None,
            args=(Scalar(args[0].squeeze()[()], core_dtype),),
        )

    def __copy__(self) -> ndarray:
        """a.__copy__()

        Used if :func:`copy.copy` is called on an array. Returns a copy
        of the array.

        Equivalent to ``a.copy(order='K')``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        result = ndarray._from_inputs(self.shape, self.dtype, inputs=(self,))
        result._thunk.copy(self._thunk, deep=False)
        return result

    def __deepcopy__(self, memo: Any | None = None) -> ndarray:
        """a.__deepcopy__(memo, /)

        Deep copy of array.

        Used if :func:`copy.deepcopy` is called on an array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        result = ndarray._from_inputs(self.shape, self.dtype, inputs=(self,))
        result._thunk.copy(self._thunk, deep=True)
        return result

    def __div__(self, rhs: Any) -> ndarray:
        """a.__div__(value, /)

        Return ``self/value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self.__truediv__(rhs))

    def __divmod__(self, rhs: Any) -> ndarray:
        """a.__divmod__(value, /)

        Return ``divmod(self, value)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        raise NotImplementedError(
            "cupynumeric.ndarray doesn't support __divmod__ yet"
        )

    def __eq__(self, rhs: object) -> ndarray:  # type: ignore [override]
        """a.__eq__(value, /)

        Return ``self==value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._equal(rhs))

    def __float__(self) -> float:
        """a.__float__(/)

        Return ``float(self)``.

        """
        return float(self.__array__())

    def __floordiv__(self, rhs: Any) -> ndarray:
        """a.__floordiv__(value, /)

        Return ``self//value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._floor_divide(rhs))

    def __format__(self, *args: Any, **kwargs: Any) -> str:
        return self.__array__().__format__(*args, **kwargs)

    def __ge__(self, rhs: Any) -> ndarray:
        """a.__ge__(value, /)

        Return ``self>=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._greater_equal(rhs))

    # __getattribute__

    def _convert_key(self, key: Any, first: bool = True) -> Any:
        # Convert any arrays stored in a key to a cuPyNumeric array
        if isinstance(key, slice):
            key = slice(
                operator.index(key.start) if key.start is not None else None,
                operator.index(key.stop) if key.stop is not None else None,
                operator.index(key.step) if key.step is not None else None,
            )
        if (
            key is np.newaxis
            or key is Ellipsis
            or np.isscalar(key)
            or isinstance(key, slice)
        ):
            return (key,) if first else key
        elif isinstance(key, tuple) and first:
            return tuple(self._convert_key(k, first=False) for k in key)
        else:
            # Otherwise convert it to a cuPyNumeric array, check types
            # and get the thunk
            key = convert_to_cupynumeric_ndarray(key)
            if key.dtype != bool and not np.issubdtype(key.dtype, np.integer):
                raise TypeError("index arrays should be int or bool type")
            if key.dtype != bool:
                key = _warn_and_convert(key, np.dtype(np.int64))

            return key._thunk

    @add_boilerplate()
    def __getitem__(self, key: Any) -> ndarray:
        """a.__getitem__(key, /)

        Return ``self[key]``.

        """
        key = self._convert_key(key)
        return ndarray._from_thunk(self._thunk.get_item(key))

    def __gt__(self, rhs: Any) -> ndarray:
        """a.__gt__(value, /)

        Return ``self>value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._greater(rhs))

    def __hash__(self) -> int:
        raise TypeError("unhashable type: cupynumeric.ndarray")

    def __iadd__(self, rhs: Any) -> ndarray:
        """a.__iadd__(value, /)

        Return ``self+=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._add(rhs, out=self))

    def __iand__(self, rhs: Any) -> ndarray:
        """a.__iand__(value, /)

        Return ``self&=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._bitwise_and(rhs, out=self)
        )

    def __idiv__(self, rhs: Any) -> ndarray:
        """a.__idiv__(value, /)

        Return ``self/=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self.__itruediv__(rhs))

    def __ifloordiv__(self, rhs: Any) -> ndarray:
        """a.__ifloordiv__(value, /)

        Return ``self//=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._floor_divide(rhs, out=self)
        )

    def __ilshift__(self, rhs: Any) -> ndarray:
        """a.__ilshift__(value, /)

        Return ``self<<=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._left_shift(rhs, out=self)
        )

    def __imatmul__(self, rhs: Any) -> ndarray:
        """a.__imatmul__(value, /)

        Return ``self@=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._matmul(rhs, out=self)
        )

    def __imod__(self, rhs: Any) -> ndarray:
        """a.__imod__(value, /)

        Return ``self%=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._remainder(rhs, out=self)
        )

    def __imul__(self, rhs: Any) -> ndarray:
        """a.__imul__(value, /)

        Return ``self*=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._multiply(rhs, out=self)
        )

    def __index__(self) -> int:
        return self.__array__().__index__()

    def __int__(self) -> int:
        """a.__int__(/)

        Return ``int(self)``.

        """
        return int(self.__array__())

    def __invert__(self) -> ndarray:
        """a.__invert__(/)

        Return ``~self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.dtype == bool:
            # Boolean values are special, just do logical NOT
            return convert_to_cupynumeric_ndarray(self._thunk._logical_not())
        else:
            return convert_to_cupynumeric_ndarray(self._thunk._invert())

    def __ior__(self, rhs: Any) -> ndarray:
        """a.__ior__(/)

        Return ``self|=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._bitwise_or(rhs, out=self)
        )

    def __ipow__(self, rhs: float) -> ndarray:
        """a.__ipow__(/)

        Return ``self**=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._power(rhs, out=self)
        )

    def __irshift__(self, rhs: Any) -> ndarray:
        """a.__irshift__(/)

        Return ``self>>=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._right_shift(rhs, out=self)
        )

    def __iter__(self) -> Any:
        """a.__iter__(/)"""
        return self.__array__().__iter__()

    def __isub__(self, rhs: Any) -> ndarray:
        """a.__isub__(/)

        Return ``self-=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._subtract(rhs, out=self)
        )

    def __itruediv__(self, rhs: Any) -> ndarray:
        """a.__itruediv__(/)

        Return ``self/=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._true_divide(rhs, out=self)
        )

    def __ixor__(self, rhs: Any) -> ndarray:
        """a.__ixor__(/)

        Return ``self^=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(
            self._thunk._bitwise_xor(rhs, out=self)
        )

    def __le__(self, rhs: Any) -> ndarray:
        """a.__le__(value, /)

        Return ``self<=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._less_equal(rhs))

    def __len__(self) -> int:
        """a.__len__(/)

        Return ``len(self)``.

        """
        return self.shape[0]

    def __lshift__(self, rhs: Any) -> ndarray:
        """a.__lshift__(value, /)

        Return ``self<<value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._left_shift(rhs))

    def __lt__(self, rhs: Any) -> ndarray:
        """a.__lt__(value, /)

        Return ``self<value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._less(rhs))

    def __matmul__(self, value: Any) -> ndarray:
        """a.__matmul__(value, /)

        Return ``self@value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._matmul(value))

    def __mod__(self, rhs: Any) -> ndarray:
        """a.__mod__(value, /)

        Return ``self%value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._remainder(rhs))

    def __mul__(self, rhs: Any) -> ndarray:
        """a.__mul__(value, /)

        Return ``self*value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._multiply(rhs))

    def __ne__(self, rhs: object) -> ndarray:  # type: ignore [override]
        """a.__ne__(value, /)

        Return ``self!=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._not_equal(rhs))

    def __neg__(self) -> ndarray:
        """a.__neg__(value, /)

        Return ``-self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._negative())

    # __new__

    @add_boilerplate()
    def nonzero(self) -> tuple[ndarray, ...]:
        """a.nonzero()

        Return the indices of the elements that are non-zero.

        Refer to :func:`cupynumeric.nonzero` for full documentation.

        See Also
        --------
        cupynumeric.nonzero : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        thunks = self._thunk.nonzero()
        return tuple(ndarray._from_thunk(thunk) for thunk in thunks)

    def __or__(self, rhs: Any) -> ndarray:
        """a.__or__(value, /)

        Return ``self|value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._bitwise_or(rhs))

    def __pos__(self) -> ndarray:
        """a.__pos__(value, /)

        Return ``+self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # the positive opeartor is equivalent to copy
        return convert_to_cupynumeric_ndarray(self._thunk._positive())

    def __pow__(self, rhs: float) -> ndarray:
        """a.__pow__(value, /)

        Return ``pow(self, value)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._power(rhs))

    def __radd__(self, lhs: Any) -> ndarray:
        """a.__radd__(value, /)

        Return ``value+self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # order doesn't matter for add
        return convert_to_cupynumeric_ndarray(self._thunk._add(lhs))

    def __rand__(self, lhs: Any) -> ndarray:
        """a.__rand__(value, /)

        Return ``value&self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # order doesn't matter for bitwise_and
        return convert_to_cupynumeric_ndarray(self._thunk._bitwise_and(lhs))

    def __rdiv__(self, lhs: Any) -> ndarray:
        """a.__rdiv__(value, /)

        Return ``value/self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(
                lhs._thunk._true_divide(self)
            )
        return _ufunc.true_divide(lhs, self)

    def __rdivmod__(self, lhs: Any) -> ndarray:
        """a.__rdivmod__(value, /)

        Return ``divmod(value, self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        raise NotImplementedError(
            "cupynumeric.ndarray doesn't support __rdivmod__ yet"
        )

    def __reduce__(self, *args: Any, **kwargs: Any) -> str | tuple[str, ...]:
        """a.__reduce__(/)

        For pickling.

        """
        return self.__array__().__reduce__(*args, **kwargs)

    def __reduce_ex__(
        self, *args: Any, **kwargs: Any
    ) -> str | tuple[str, ...]:
        return self.__array__().__reduce_ex__(*args, **kwargs)

    def __repr__(self) -> str:
        """a.__repr__(/)

        Return ``repr(self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return repr(self.__array__())

    def __rfloordiv__(self, lhs: Any) -> ndarray:
        """a.__rfloordiv__(value, /)

        Return ``value//self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(
                lhs._thunk._floor_divide(self)
            )
        return _ufunc.floor_divide(lhs, self)

    def __rmatmul__(self, lhs: Any) -> ndarray:
        """a.__rmatmul__(value, /)

        Return ``value@self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from .._module.linalg_mvp import matmul

        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(lhs._thunk._matmul(self))
        return convert_to_cupynumeric_ndarray(matmul(lhs, self))

    def __rmod__(self, lhs: Any) -> ndarray:
        """a.__rmod__(value, /)

        Return ``value%self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(lhs._thunk._remainder(self))
        return _ufunc.remainder(lhs, self)

    def __rmul__(self, lhs: Any) -> ndarray:
        """a.__rmul__(value, /)

        Return ``value*self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(lhs._thunk._multiply(self))
        return convert_to_cupynumeric_ndarray(self._thunk._multiply(lhs))

    def __ror__(self, lhs: Any) -> ndarray:
        """a.__ror__(value, /)

        Return ``value|self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # order doesn't matter here
        return convert_to_cupynumeric_ndarray(self._thunk._bitwise_or(lhs))

    def __rpow__(self, lhs: Any) -> ndarray:
        """__rpow__(value, /)

        Return ``pow(value, self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(lhs._thunk._power(self))
        return _ufunc.power(lhs, self)

    def __rshift__(self, rhs: Any) -> ndarray:
        """a.__rshift__(value, /)

        Return ``self>>value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._right_shift(rhs))

    def __rsub__(self, lhs: Any) -> ndarray:
        """a.__rsub__(value, /)

        Return ``value-self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(lhs._thunk._subtract(self))
        return _ufunc.subtract(lhs, self)

    def __rtruediv__(self, lhs: Any) -> ndarray:
        """a.__rtruediv__(value, /)

        Return ``value/self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(
                lhs._thunk._true_divide(self)
            )
        return _ufunc.true_divide(lhs, self)

    def __rxor__(self, lhs: Any) -> ndarray:
        """a.__rxor__(value, /)

        Return ``value^self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if hasattr(lhs, "_thunk"):
            return convert_to_cupynumeric_ndarray(
                lhs._thunk._bitwise_xor(self)
            )
        return _ufunc.bitwise_xor(lhs, self)

    # __setattr__
    @add_boilerplate("value")
    def __setitem__(self, key: Any, value: ndarray) -> None:
        """__setitem__(key, value, /)

        Set ``self[key]=value``.

        """
        check_writeable(self)

        if value.dtype != self.dtype:
            temp = ndarray._from_inputs(
                value.shape, dtype=self.dtype, inputs=(value,)
            )
            temp._thunk.convert(value._thunk)
            value = temp

        if len(value.shape) > len(self.shape):
            N = len(value.shape) - len(self.shape)
            first = value.shape[:N]
            if builtin_prod(first) == 1:
                value = value.squeeze(tuple(range(N)))
            else:
                raise ValueError(
                    "could not broadcast input array from shape "
                    f"{value.shape} into shape {self.shape}"
                )

        key = self._convert_key(key)
        self._thunk.set_item(key, value._thunk)

    def __setstate__(self, state: Any) -> None:
        """a.__setstate__(state, /)

        For unpickling.

        The `state` argument must be a sequence that contains the following
        elements:

        Parameters
        ----------
        version : int
            optional pickle version. If omitted defaults to 0.
        shape : tuple
        dtype : data-type
        isFortran : bool
        rawdata : str or list
            a binary string with the data, or a list if 'a' is an object array

        """
        self.__array__().__setstate__(state)

    def __sizeof__(self, *args: Any, **kwargs: Any) -> int:
        return self.__array__().__sizeof__(*args, **kwargs)

    def __sub__(self, rhs: Any) -> ndarray:
        """a.__sub__(value, /)

        Return ``self-value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._subtract(rhs))

    def __str__(self) -> str:
        """a.__str__(/)

        Return ``str(self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return str(self.__array__())

    def __truediv__(self, rhs: Any) -> ndarray:
        """a.__truediv__(value, /)

        Return ``self/value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._true_divide(rhs))

    def __xor__(self, rhs: Any) -> ndarray:
        """a.__xor__(value, /)

        Return ``self^value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return convert_to_cupynumeric_ndarray(self._thunk._bitwise_xor(rhs))

    # scalar functions on 0d arrays
    def __round__(self, ndigits: int | None = None) -> Any:
        if self.ndim == 0:
            value = self.__array__().item()
            if not isinstance(value, (int, float, np.number)):
                raise TypeError(
                    f"Rounding not supported for type: {self.dtype}"
                )
            return round(float(value), ndigits)
        else:
            raise ValueError(
                "Python's round method can be called only on scalars"
            )

    @add_boilerplate()
    def all(
        self,
        axis: Any = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        initial: int | float | None = None,
        where: ndarray | None = None,
    ) -> ndarray:
        """a.all(axis=None, out=None, keepdims=False, initial=None, where=True)

        Returns True if all elements evaluate to True.

        Refer to :func:`cupynumeric.all` for full documentation.

        See Also
        --------
        cupynumeric.all : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return perform_unary_reduction(
            UnaryRedCode.ALL,
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def any(
        self,
        axis: Any = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        initial: int | float | None = None,
        where: ndarray | None = None,
    ) -> ndarray:
        """a.any(axis=None, out=None, keepdims=False, initial=None, where=True)

        Returns True if any of the elements of `a` evaluate to True.

        Refer to :func:`cupynumeric.any` for full documentation.

        See Also
        --------
        cupynumeric.any : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return perform_unary_reduction(
            UnaryRedCode.ANY,
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def argmax(
        self,
        axis: Any = None,
        out: ndarray | None = None,
        keepdims: bool = False,
    ) -> ndarray:
        """a.argmax(axis=None, out=None)

        Return indices of the maximum values along the given axis.

        Refer to :func:`cupynumeric.argmax` for full documentation.

        See Also
        --------
        cupynumeric.argmax : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if out is not None and out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
        if axis is not None and not isinstance(axis, int):
            raise ValueError("axis must be an integer")
        return perform_unary_reduction(
            UnaryRedCode.ARGMAX, self, axis=axis, out=out, keepdims=keepdims
        )

    @add_boilerplate()
    def argmin(
        self,
        axis: Any = None,
        out: ndarray | None = None,
        keepdims: bool = False,
    ) -> ndarray:
        """a.argmin(axis=None, out=None)

        Return indices of the minimum values along the given axis.

        Refer to :func:`cupynumeric.argmin` for detailed documentation.

        See Also
        --------
        cupynumeric.argmin : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if out is not None and out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
        if axis is not None and not isinstance(axis, int):
            raise ValueError("axis must be an integer")
        return perform_unary_reduction(
            UnaryRedCode.ARGMIN, self, axis=axis, out=out, keepdims=keepdims
        )

    def astype(
        self,
        dtype: npt.DTypeLike,
        order: OrderType = "C",
        casting: CastingKind = "unsafe",
        subok: bool = True,
        copy: bool = False,
    ) -> ndarray:
        """a.astype(dtype, order='C', casting='unsafe', subok=True, copy=True)

        Copy of the array, cast to a specified type.

        Parameters
        ----------
        dtype : str or data-type
            Typecode or data-type to which the array is cast.

        order : ``{'C', 'F', 'A', 'K'}``, optional
            Controls the memory layout order of the result.
            'C' means C order, 'F' means Fortran order, 'A'
            means 'F' order if all the arrays are Fortran contiguous,
            'C' order otherwise, and 'K' means as close to the
            order the array elements appear in memory as possible.
            Default is 'K'.

        casting : ``{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}``, optional
            Controls what kind of data casting may occur. Defaults to 'unsafe'
            for backwards compatibility.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
                like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.

        subok : bool, optional
            If True, then sub-classes will be passed-through (default),
            otherwise the returned array will be forced to be a base-class
            array.

        copy : bool, optional
            By default, astype does not returns a newly allocated array. If
            this is set to True, a copy is made and returned, instead of the
            input array.

        Notes
        -----
        The default value for the ``copy`` argument is the opposite of Numpy.
        Avoiding copies reduces memory pressure.

        Returns
        -------
        arr_t : ndarray
            Unless `copy` is False and the other conditions for returning the
            input array are satisfied (see description for `copy` input
            parameter), `arr_t` is a new array of the same shape as the input
            array, with dtype, order given by `dtype`, `order`.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._astype(dtype, order, casting, subok, copy, False)

    def _astype(
        self,
        dtype: npt.DTypeLike,
        order: OrderType = "C",
        casting: CastingKind = "unsafe",
        subok: bool = True,
        copy: bool = True,
        temporary: bool = False,
    ) -> ndarray:
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self

        casting_allowed = np.can_cast(self.dtype, dtype, casting)
        if casting_allowed:
            # For numeric to non-numeric casting, the dest dtype should be
            # retrived from 'promote_types' to preserve values
            # e.g. ) float 64 to str, np.dtype(dtype) == '<U'
            # this dtype is not safe to store
            if dtype == np.dtype("str"):
                dtype = np.promote_types(self.dtype, dtype)
        else:
            raise TypeError(
                f"Cannot cast array data"
                f"from '{self.dtype}' to '{dtype}' "
                f"to the rule '{casting}'"
            )
        result = ndarray._from_inputs(self.shape, dtype=dtype, inputs=(self,))
        result._thunk.convert(self._thunk, warn=False, temporary=temporary)
        return result

    @add_boilerplate()
    def take(
        self,
        indices: int | npt.ArrayLike,
        axis: int | None = None,
        out: ndarray | None = None,
        mode: BoundsMode = "raise",
    ) -> ndarray:
        """a.take(indices, axis=None, out=None, mode="raise")

        Take elements from an array along an axis.

        Refer to :func:`cupynumeric.take` for full documentation.

        See Also
        --------
        cupynumeric.take : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # garbage check axis and indices early before temporary arrays might be created
        if axis is not None:
            axis = normalize_axis_index(axis, self.ndim)

        is_scalar = np.isscalar(indices)
        ind_shape: tuple[int, ...]
        ind_thunk: Any
        if is_scalar:
            # avoid converting indices to an array to avoid overhead
            ind = indices
            ind_dtype = np.dtype(type(indices))
            ind_size = 1
            ind_shape = ()
            ind_thunk = indices
        else:
            ind = convert_to_cupynumeric_ndarray(indices)
            ind_dtype = ind.dtype
            ind_size = ind.size
            ind_shape = ind.shape
            ind_thunk = ind._thunk
        if not np.issubdtype(ind_dtype, np.integer):
            raise TypeError("indices array should be integers")

        lim = self.size if axis is None else self.shape[axis]
        if lim == 0 and ind_size != 0:
            raise IndexError(
                "Cannot do a non-empty take() from an empty axis."
            )

        if mode not in {"raise", "wrap", "clip"}:
            raise ValueError(f"Invalid mode '{mode}' for take operation")

        if axis is None:
            out_shape = ind_shape
        else:
            in_shape = self.shape
            out_shape = in_shape[:axis] + ind_shape + in_shape[(axis + 1) :]

        out_thunk = None
        if out is not None:
            if out.shape != out_shape:
                raise ValueError(
                    f"Shape mismatch: output array has shape {out.shape}, expected {out_shape}"
                )
            if out.dtype != self.dtype:
                raise TypeError("Type mismatch: out array has the wrong type")
            out_thunk = out._thunk

        if np.prod(out_shape) == 0:
            if out is None:
                inputs = (self,) if is_scalar else (self, ind)
                out = ndarray._from_inputs(
                    shape=out_shape, dtype=self.dtype, inputs=inputs
                )
            return out

        result_thunk = self._thunk.take(
            ind_thunk, axis, out=out_thunk, mode=mode
        )
        if out is not None:
            assert result_thunk is out_thunk
            return out
        return ndarray._from_thunk(result_thunk)

    @add_boilerplate()
    def choose(
        self,
        choices: Any,
        out: ndarray | None = None,
        mode: BoundsMode = "raise",
    ) -> ndarray:
        """a.choose(choices, out=None, mode='raise')

        Use an index array to construct a new array from a set of choices.

        Refer to :func:`cupynumeric.choose` for full documentation.

        See Also
        --------
        cupynumeric.choose : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        a = self

        if isinstance(choices, list):
            choices = tuple(choices)
        is_tuple = isinstance(choices, tuple)
        if is_tuple:
            if (n := len(choices)) == 0:
                raise ValueError("invalid entry in choice array")
            dtypes = [ch.dtype for ch in choices]
            ch_dtype = np.result_type(*dtypes)
            choices = tuple(
                convert_to_cupynumeric_ndarray(choices[i]).astype(ch_dtype)
                for i in range(n)
            )

        else:
            choices = convert_to_cupynumeric_ndarray(choices)
            n = choices.shape[0]
            ch_dtype = choices.dtype
            choices = tuple(choices[i, ...] for i in range(n))

        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError("a array should be integer type")

        if self.dtype != np.int64:
            a = a.astype(np.int64)

        if mode not in ["raise", "wrap", "clip"]:
            mode = "raise"

        if mode == "raise":
            if (a < 0).any() or (a >= n).any():
                raise ValueError("invalid entry in choice array")
        elif mode == "wrap":
            a = a % n
            if not isinstance(a, ndarray):
                a = convert_to_cupynumeric_ndarray(a)  # type: ignore [unreachable] # noqa: E501
        elif mode == "clip":
            a = a.clip(0, n - 1)

        # we need to broadcast all arrays in choices with
        # input and output arrays
        if out is not None:
            out_shape = np.broadcast_shapes(
                a.shape, choices[0].shape, out.shape
            )
        else:
            out_shape = np.broadcast_shapes(a.shape, choices[0].shape)

        for c in choices:
            out_shape = np.broadcast_shapes(out_shape, c.shape)

        # if output is provided, it shape should be the same as out_shape
        if out is not None and out.shape != out_shape:
            raise ValueError(
                f"non-broadcastable output operand with shape "
                f" {str(out.shape)}"
                f" doesn't match the broadcast shape {out_shape}"
            )

        if out is not None and out.dtype == ch_dtype:
            out_arr = out

        else:
            # no output, create one
            out_arr = ndarray._from_inputs(
                shape=out_shape, dtype=ch_dtype, inputs=(a, choices)
            )

        ch = tuple(c._thunk for c in choices)
        out_arr._thunk.choose(a._thunk, *ch)

        if out is not None and out.dtype != ch_dtype:
            out._thunk.convert(out_arr._thunk)
            return out

        return out_arr

    @add_boilerplate()
    def compress(
        self, condition: ndarray, axis: Any = None, out: ndarray | None = None
    ) -> ndarray:
        """a.compress(self, condition, axis=None, out=None)

        Return selected slices of an array along given axis.

        Refer to :func:`cupynumeric.compress` for full documentation.

        See Also
        --------
        cupynumeric.compress : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        a = self
        try:
            if condition.ndim != 1:
                raise ValueError(
                    "Dimension mismatch: condition must be a 1D array"
                )
        except AttributeError:
            raise ValueError(
                "Dimension mismatch: condition must be a 1D array"
            )
        condition = _warn_and_convert(condition, np.dtype(bool))

        if axis is None:
            axis = 0
            a = self.ravel()
        else:
            axis = normalize_axis_index(axis, self.ndim)

        if a.shape[axis] < condition.shape[0]:
            raise ValueError(
                "Shape mismatch: condition contains entries that are out of bounds"
            )
        elif a.shape[axis] > condition.shape[0]:
            slice_tuple = tuple(slice(None) for ax in range(axis)) + (
                slice(0, condition.shape[0]),
            )
            a = a[slice_tuple]

        index_tuple: tuple[Any, ...] = tuple(slice(None) for ax in range(axis))
        index_tuple += (condition,)

        if out is not None:
            out[:] = a[index_tuple]
            return out
        else:
            res = a[index_tuple]
            return res

    @add_boilerplate()
    def clip(
        self,
        min: int | float | npt.ArrayLike | None = None,
        max: int | float | npt.ArrayLike | None = None,
        out: ndarray | None = None,
    ) -> ndarray:
        """a.clip(min=None, max=None, out=None)

        Return an array whose values are limited to ``[min, max]``.

        One of max or min must be given.

        Refer to :func:`cupynumeric.clip` for full documentation.

        See Also
        --------
        cupynumeric.clip : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        min = max_identity(self.dtype) if min is None else min
        max = min_identity(self.dtype) if max is None else max

        def ensure_compatible(val: Any) -> npt.NDArray[Any]:
            if np.isscalar(val):
                return np.array(val, dtype=self.dtype)
            arr = np.array(val)
            if arr.shape == ():
                return arr.astype(self.dtype)
            return np.array(val, dtype=self.dtype)

        min = ensure_compatible(min)
        max = ensure_compatible(max)

        if min.size != 1 or max.size != 1:
            runtime.warn(
                "cuPyNumeric has not implemented clip with array-like "
                "arguments and is falling back to canonical numpy. You "
                "may notice significantly decreased performance for this "
                "function call.",
                category=RuntimeWarning,
            )
            if out is not None:
                self.__array__().clip(min, max, out=out.__array__())
                return out
            else:
                return convert_to_cupynumeric_ndarray(
                    self.__array__().clip(min, max)
                )
        core_dtype = to_core_type(self.dtype)
        extra_args = (Scalar(min, core_dtype), Scalar(max, core_dtype))
        return perform_unary_op(
            UnaryOpCode.CLIP, self, out=out, extra_args=extra_args
        )

    @add_boilerplate()
    def round(self, decimals: int = 0, out: ndarray | None = None) -> ndarray:
        """a.round(decimals=0, out=None)

        Return a with each element rounded to the given number of decimals.

        Refer to :func:`cupynumeric.round` for full documentation.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        extra_args = (
            Scalar(decimals, ty.int64),
            Scalar(10 ** abs(decimals), ty.int64),
        )
        return perform_unary_op(
            UnaryOpCode.ROUND, self, out=out, extra_args=extra_args
        )

    def conj(self) -> ndarray:
        """a.conj()

        Complex-conjugate all elements.

        Refer to :func:`cupynumeric.conjugate` for full documentation.

        See Also
        --------
        cupynumeric.conjugate : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.dtype.kind == "c":
            return ndarray._from_thunk(self._thunk.conj())
        else:
            return self

    def conjugate(self) -> ndarray:
        """a.conjugate()

        Return the complex conjugate, element-wise.

        Refer to :func:`cupynumeric.conjugate` for full documentation.

        See Also
        --------
        cupynumeric.conjugate : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.conj()

    def copy(self, order: OrderType = "C") -> ndarray:
        """copy()

        Get a copy of the iterator as a 1-D array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # We don't care about dimension order in cuPyNumeric
        return self.__copy__()

    @add_boilerplate()
    def cumsum(
        self,
        axis: Any = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
    ) -> ndarray:
        return perform_scan(
            ScanCode.SUM,
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            nan_to_identity=False,
        )

    @add_boilerplate()
    def cumprod(
        self,
        axis: Any = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
    ) -> ndarray:
        return perform_scan(
            ScanCode.PROD,
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            nan_to_identity=False,
        )

    @add_boilerplate()
    def nancumsum(
        self,
        axis: Any = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
    ) -> ndarray:
        return perform_scan(
            ScanCode.SUM,
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            nan_to_identity=True,
        )

    @add_boilerplate()
    def nancumprod(
        self,
        axis: Any = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
    ) -> ndarray:
        return perform_scan(
            ScanCode.PROD,
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            nan_to_identity=True,
        )

    # diagonal helper. Will return diagonal for arbitrary number of axes;
    # currently offset option is implemented only for the case of number of
    # axes=2. This restriction can be lifted in the future if there is a
    # use case of having arbitrary number of offsets
    def _diag_helper(
        self,
        offset: int = 0,
        axes: Any | None = None,
        extract: bool = True,
        trace: bool = False,
        out: ndarray | None = None,
        dtype: np.dtype[Any] | None = None,
    ) -> ndarray:
        # _diag_helper can be used only for arrays with dim>=1
        if self.ndim < 1:
            raise ValueError("_diag_helper is implemented for dim>=1")
        # out should be passed only for Trace
        if out is not None and not trace:
            raise ValueError("_diag_helper supports out only for trace=True")
        # dtype should be passed only for Trace
        if dtype is not None and not trace:
            raise ValueError("_diag_helper supports dtype only for trace=True")

        if self.ndim == 1:
            if axes is not None:
                raise ValueError(
                    "Axes shouldn't be specified when getting diagonal for 1D array"
                )
            m = self.shape[0] + np.abs(offset)
            res = ndarray._from_inputs(
                (m, m), dtype=self.dtype, inputs=(self,)
            )
            diag_size = self.shape[0]
            res._thunk._diag_helper(
                self._thunk, offset=offset, naxes=0, extract=False, trace=False
            )
        else:
            assert axes is not None
            N = len(axes)
            if len(axes) != len(OrderedSet(axes)):
                raise ValueError("axis1 and axis2 cannot be the same")
            if self.ndim < N:
                raise ValueError(
                    "Dimension of input array shouldn't be less than number of axes"
                )
            # pack the axes that are not going to change
            transpose_axes = tuple(
                ax for ax in range(self.ndim) if ax not in axes
            )
            # only 2 axes provided, we transpose based on the offset
            if N == 2:
                if offset >= 0:
                    a = self.transpose(transpose_axes + (axes[0], axes[1]))
                else:
                    a = self.transpose(transpose_axes + (axes[1], axes[0]))
                    offset = -offset

                if offset >= a.shape[self.ndim - 1]:
                    return ndarray(shape=(0,))

                diag_size = max(0, min(a.shape[-2], a.shape[-1] - offset))
            # more than 2 axes provided:
            elif N > 2:
                # offsets are supported only when naxes=2
                if offset != 0:
                    raise ValueError(
                        "offset supported for number of axes == 2"
                    )
                # sort axes along which diagonal is calculated by size
                axes = sorted(axes, reverse=True, key=lambda i: self.shape[i])
                axes = tuple(axes)
                # transpose a so axes for which diagonal is calculated are at
                #  at the end
                a = self.transpose(transpose_axes + axes)
                diag_size = a.shape[a.ndim - 1]
            elif N < 2:
                raise ValueError(
                    "number of axes passed to the _diag_helper should be more than 1"
                )

            tr_shape = tuple(a.shape[i] for i in range(a.ndim - N))
            # calculate shape of the output array
            out_shape: NdShape
            if trace:
                if N != 2:
                    raise ValueError(
                        " exactly 2 axes should be passed to trace"
                    )
                if self.ndim == 2:
                    out_shape = (1,)
                elif self.ndim > 2:
                    out_shape = tr_shape
                else:
                    raise ValueError(
                        "dimension of the array for trace operation: should be >=2"
                    )
            else:
                out_shape = tr_shape + (diag_size,)

            out_dtype = (
                dtype
                if dtype is not None
                else out.dtype
                if out is not None
                else a.dtype
            )
            a = a._maybe_convert(out_dtype, (a,))
            if out is not None and out.shape != out_shape:
                raise ValueError("output array has the wrong shape")
            if out is not None and out.dtype == out_dtype:
                res = out
            else:
                res = ndarray._from_inputs(
                    shape=out_shape, dtype=out_dtype, inputs=(self,)
                )

            res._thunk._diag_helper(
                a._thunk, offset=offset, naxes=N, extract=extract, trace=trace
            )
            if out is not None and out is not res:
                out._thunk.convert(res._thunk)
                res = out

        return res

    def diagonal(
        self,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        extract: bool = True,
    ) -> ndarray:
        """a.diagonal(offset=0, axis1=None, axis2=None)

        Return specified diagonals.

        Refer to :func:`cupynumeric.diagonal` for full documentation.

        See Also
        --------
        cupynumeric.diagonal : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.ndim == 1:
            if extract is True:
                raise ValueError("extract can be true only for Ndim >=2")
            axes = None
        else:
            if axis1 is None or axis2 is None:
                raise TypeError(
                    "'NoneType' object cannot be interpreted as an integer"
                )
            if isinstance(axis1, float) or isinstance(axis2, float):
                raise TypeError(
                    "'float' object cannot be interpreted as an integer"
                )
            if isinstance(offset, float):
                raise TypeError("integer argument expected, got float")
            axes = (axis1, axis2)
        return self._diag_helper(offset=offset, axes=axes, extract=extract)

    @add_boilerplate("indices", "values")
    def put(
        self, indices: ndarray, values: ndarray, mode: str = "raise"
    ) -> None:
        """
        Replaces specified elements of the array with given values.

        Refer to :func:`cupynumeric.put` for full documentation.

        See Also
        --------
        cupynumeric.put : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        check_writeable(self)

        if values.size == 0 or indices.size == 0 or self.size == 0:
            return

        if mode not in ("raise", "wrap", "clip"):
            raise ValueError(
                f"mode must be one of 'clip', 'raise', or 'wrap' (got  {mode})"
            )

        if mode == "wrap":
            indices = indices % self.size
        elif mode == "clip":
            indices = indices.clip(0, self.size - 1)

        indices = _warn_and_convert(indices, np.dtype(np.int64))
        values = _warn_and_convert(values, self.dtype)

        if indices.ndim > 1:
            indices = indices.ravel()

        if self.shape == ():
            if mode == "raise":
                if indices.min() < -1 or indices.max() > 0:
                    raise IndexError("Indices out of bounds")
            if values.shape == ():
                v = values
            else:
                v = values[0]
            self._thunk.copy(v._thunk, deep=False)
            return

        # indices might have taken an eager path above
        indices = convert_to_cupynumeric_ndarray(indices)

        # call _wrap on the values if they need to be wrapped
        if values.ndim != indices.ndim or values.size != indices.size:
            values = values._wrap(indices.size)

        self._thunk.put(indices._thunk, values._thunk, mode == "raise")

    @add_boilerplate()
    def trace(
        self,
        offset: int = 0,
        axis1: Any = None,
        axis2: Any = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
    ) -> ndarray:
        """a.trace(offset=0, axis1=None, axis2=None, dtype = None, out = None)

        Return the sum along diagonals of the array.

        Refer to :func:`cupynumeric.trace` for full documentation.

        See Also
        --------
        cupynumeric.trace : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.ndim < 2:
            raise ValueError(
                "trace operation can't be called on a array with DIM<2"
            )

        axes: tuple[int, ...] = ()
        if (axis1 is None) and (axis2 is None):
            # default values for axis
            axes = (0, 1)
        elif (axis1 is None) or (axis2 is None):
            raise TypeError("both axes should be passed")
        else:
            axes = (axis1, axis2)

        res = self._diag_helper(
            offset=offset, axes=axes, trace=True, out=out, dtype=dtype
        )

        # for 2D arrays we must return scalar
        if self.ndim == 2:
            res = res[0]

        return res

    @add_boilerplate("rhs")
    def dot(self, rhs: ndarray, out: ndarray | None = None) -> ndarray:
        """a.dot(rhs, out=None)

        Return the dot product of this array with ``rhs``.

        Refer to :func:`cupynumeric.dot` for full documentation.

        See Also
        --------
        cupynumeric.dot : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # work around circular import
        from .._module.linalg_mvp import _contract

        if self.ndim == 0 or rhs.ndim == 0:
            return convert_to_cupynumeric_ndarray(
                self._thunk._multiply(rhs, out=out)
            )

        (self_modes, rhs_modes, out_modes) = dot_modes(self.ndim, rhs.ndim)
        return _contract(
            self_modes,
            rhs_modes,
            out_modes,
            self,
            rhs,
            out=out,
            casting="unsafe",
        )

    def dump(self, file: str | Path) -> None:
        """a.dump(file)

        Dump a pickle of the array to the specified file.

        The array can be read back with pickle.load or cupynumeric.load.

        Parameters
        ----------
        file : str or `pathlib.Path`
            A string naming the dump file.

        Availability
        --------
        Single CPU

        """
        self.__array__().dump(file=file)

    def dumps(self) -> bytes:
        """a.dumps()

        Returns the pickle of the array as a string.

        pickle.loads will convert the string back to an array.

        Parameters
        ----------
        None

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().dumps()

    def _normalize_axes_shape(
        self, axes: Any, s: Any
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        user_axes = axes is not None
        user_sizes = s is not None
        if user_axes and user_sizes and len(axes) != len(s):
            raise ValueError("Shape and axes have different lengths")
        if user_axes:
            fft_axes = [ax if ax >= 0 else ax + self.ndim for ax in axes]
            if min(fft_axes) < 0 or max(fft_axes) >= self.ndim:
                raise ValueError(
                    "Axis is out of bounds for array of size {}".format(
                        self.ndim
                    )
                )
        else:
            fft_axes = list(range(len(s)) if user_sizes else range(self.ndim))

        fft_s = list(self.shape)
        if user_sizes:
            for idx, ax in enumerate(fft_axes):
                fft_s[ax] = s[idx]
        return np.asarray(fft_axes), np.asarray(fft_s)

    def fft(
        self,
        s: Any,
        axes: Sequence[int] | None,
        kind: FFTType,
        direction: FFTDirection,
        norm: Any,
    ) -> ndarray:
        """a.fft(s, axes, kind, direction, norm)

        Return the ``kind`` ``direction`` FFT of this array
        with normalization ``norm``.

        Common entrypoint for FFT functionality in cupynumeric.fft module.

        Notes
        -----
        Multi-GPU usage is limited to data parallel axis-wise batching.

        See Also
        --------
        cupynumeric.fft : FFT functions for different ``kind`` and
        ``direction`` arguments

        Availability
        --------
        Multiple GPUs (partial)

        """
        # Type
        fft_output_type = kind.output_dtype

        # Axes and sizes
        user_sizes = s is not None
        fft_axes, fft_s = self._normalize_axes_shape(axes, s)

        # Shape
        fft_input = self
        fft_input_shape = np.asarray(self.shape)
        fft_output_shape = np.asarray(self.shape)
        if user_sizes:
            # Zero padding if any of the user sizes is larger than input
            zeropad_input = self
            if np.any(np.greater(fft_s, fft_input_shape)):
                # Create array with superset shape, fill with zeros,
                # and copy input in
                max_size = np.maximum(fft_s, fft_input_shape)
                zeropad_input = ndarray(shape=max_size, dtype=fft_input.dtype)
                zeropad_input.fill(0)
                slices = tuple(slice(0, i) for i in fft_input.shape)
                zeropad_input._thunk.set_item(slices, fft_input._thunk)

            # Slicing according to final shape
            for idx, ax in enumerate(fft_axes):
                fft_input_shape[ax] = s[idx]
            # TODO: always copying is not the best idea,
            # sometimes a view of the original input will do
            slices = tuple(slice(0, i) for i in fft_s)
            fft_input = ndarray._from_thunk(
                zeropad_input._thunk.get_item(slices)
            )
            fft_output_shape = np.copy(fft_input_shape)

        # R2C/C2R require different output shapes
        if fft_output_type != self.dtype:
            # R2C/C2R dimension is the last axis
            lax = fft_axes[-1]
            if direction == FFTDirection.FORWARD:
                fft_output_shape[lax] = fft_output_shape[lax] // 2 + 1
            else:
                if user_sizes:
                    fft_output_shape[lax] = s[-1]
                else:
                    fft_output_shape[lax] = 2 * (fft_input.shape[lax] - 1)

        # Execute FFT backend
        out = ndarray(shape=tuple(fft_output_shape), dtype=fft_output_type)
        out._thunk.fft(
            fft_input._thunk, cast(Sequence[int], fft_axes), kind, direction
        )

        # Normalization
        fft_norm = FFTNormalization.from_string(norm)
        do_normalization = any(
            (
                fft_norm == FFTNormalization.ORTHOGONAL,
                fft_norm == FFTNormalization.FORWARD
                and direction == FFTDirection.FORWARD,
                fft_norm == FFTNormalization.INVERSE
                and direction == FFTDirection.INVERSE,
            )
        )
        if do_normalization:
            if direction == FFTDirection.FORWARD:
                norm_shape = fft_input.shape
            else:
                norm_shape = out.shape
            norm_shape_along_axes = [norm_shape[ax] for ax in fft_axes]
            factor = np.prod(norm_shape_along_axes)
            if fft_norm == FFTNormalization.ORTHOGONAL:
                factor = np.sqrt(factor)
            return convert_to_cupynumeric_ndarray(
                out / factor.astype(fft_output_type)
            )

        return out

    def fill(self, value: float) -> None:
        """a.fill(value)

        Fill the array with a scalar value.

        Parameters
        ----------
        value : scalar
            All elements of `a` will be assigned this value.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        check_writeable(self)
        val = np.array(value, dtype=self.dtype)
        self._thunk.fill(val)

    def flatten(self, order: OrderType = "C") -> ndarray:
        """a.flatten(order='C')

        Return a copy of the array collapsed into one dimension.

        Parameters
        ----------
        order : ``{'C', 'F', 'A', 'K'}``, optional
            'C' means to flatten in row-major (C-style) order.
            'F' means to flatten in column-major (Fortran-
            style) order. 'A' means to flatten in column-major
            order if `a` is Fortran *contiguous* in memory,
            row-major order otherwise. 'K' means to flatten
            `a` in the order the elements occur in memory.
            The default is 'C'.

        Returns
        -------
        y : ndarray
            A copy of the input array, flattened to one dimension.

        See Also
        --------
        ravel : Return a flattened array.
        flat : A 1-D flat iterator over the array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # Reshape first and make a copy if the output is a view of the src
        # the output always should be a copy of the src array
        result = self.reshape(-1, order=order)
        if self.ndim <= 1:
            result = result.copy()
        return result

    def getfield(self, dtype: np.dtype[Any], offset: int = 0) -> None:
        raise NotImplementedError(
            "cuPyNumeric does not currently support type reinterpretation "
            "for ndarray.getfield"
        )

    def _convert_singleton_key(self, args: tuple[Any, ...]) -> Any:
        if len(args) == 0 and self.size == 1:
            return (0,) * self.ndim
        if len(args) == 1 and isinstance(args[0], int):
            flat_idx = args[0]
            if flat_idx >= self.size:
                raise IndexError(
                    f"index {flat_idx} is out of bounds for size {self.size}"
                )
            result: tuple[int, ...] = ()
            for dim_size in reversed(self.shape):
                result = (flat_idx % dim_size,) + result
                flat_idx //= dim_size
            return result
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        if len(args) != self.ndim or any(not isinstance(x, int) for x in args):
            raise ValueError(
                "can only convert an array of size 1 to a Python scalar"
            )
        return args

    def item(self, *args: Any) -> Any:
        """a.item(*args)

        Copy an element of an array to a standard Python scalar and return it.

        Parameters
        ----------
        \\*args :

            * none: in this case, the method only works for arrays
                with one element (`a.size == 1`), which element is
                copied into a standard Python scalar object and returned.
            * int_type: this argument is interpreted as a flat index into
                the array, specifying which element to copy and return.
            * tuple of int_types: functions as does a single int_type
                argument, except that the argument is interpreted as an
                nd-index into the array.

        Returns
        -------
        z : scalar
            A copy of the specified element of the array as a suitable
            Python scalar

        Notes
        -----
        When the data type of `a` is longdouble or clongdouble, item() returns
        a scalar array object because there is no available Python scalar that
        would not lose information. Void arrays return a buffer object for
        item(), unless fields are defined, in which case a tuple is returned.
        `item` is very similar to a[args], except, instead of an array scalar,
        a standard Python scalar is returned. This can be useful for speeding
        up access to elements of the array and doing arithmetic on elements of
        the array using Python's optimized math.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        key = self._convert_singleton_key(args)
        result = self[key]
        assert result.shape == ()
        return result._thunk.__numpy_array__()

    if not is_np2:

        def itemset(self, *args: Any) -> None:
            """a.itemset(*args)

            Insert scalar into an array (scalar is cast to array's dtype,
            if possible)

            There must be at least 1 argument, and define the last argument
            as *item*.  Then, ``a.itemset(*args)`` is equivalent to but faster
            than ``a[args] = item``.  The item should be a scalar value and
            `args` must select a single item in the array `a`.

            Parameters
            ----------
            \\*args :
                If one argument: a scalar, only used in case `a` is of size 1.
                If two arguments: the last argument is the value to be set
                and must be a scalar, the first argument specifies a single
                array element location. It is either an int or a tuple.

            Notes
            -----
            Compared to indexing syntax, `itemset` provides some speed increase
            for placing a scalar into a particular location in an `ndarray`,
            if you must do this.  However, generally this is discouraged:
            among other problems, it complicates the appearance of the code.
            Also, when using `itemset` (and `item`) inside a loop, be sure
            to assign the methods to a local variable to avoid the attribute
            look-up at each loop iteration.

            Availability
            --------
            Multiple GPUs, Multiple CPUs

            """
            if len(args) == 0:
                raise KeyError("itemset() requires at least one argument")
            value = args[-1]
            args = args[:-1]
            key = self._convert_singleton_key(args)
            self[key] = value

    @add_boilerplate()
    def max(
        self,
        axis: Any = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        initial: int | float | None = None,
        where: ndarray | None = None,
    ) -> ndarray:
        """a.max(axis=None, out=None, keepdims=False, initial=<no value>,
                 where=True)

        Return the maximum along a given axis.

        Refer to :func:`cupynumeric.amax` for full documentation.

        See Also
        --------
        cupynumeric.amax : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return perform_unary_reduction(
            UnaryRedCode.MAX,
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def _count_nonzero(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> int | ndarray:
        if self.size == 0:
            if axis is None:
                return 0

            from .thunk import normalize_axis_tuple

            axis = normalize_axis_tuple(axis, self.ndim)
            if keepdims:
                out_shape = tuple(
                    1 if i in axis else s for i, s in enumerate(self.shape)
                )
            else:
                out_shape = tuple(
                    s for i, s in enumerate(self.shape) if i not in axis
                )
            if not out_shape:
                return 0
            result = ndarray(shape=out_shape, dtype=np.uint64)
            result.fill(0)
            return result

        return perform_unary_reduction(
            UnaryRedCode.COUNT_NONZERO, self, axis=axis, keepdims=keepdims
        )

    def _summation_dtype(self, dtype: np.dtype[Any] | None) -> np.dtype[Any]:
        # Pick our dtype if it wasn't picked yet
        if dtype is None:
            if self.dtype.kind != "f" and self.dtype.kind != "c":
                return np.dtype(np.float64)
            else:
                return self.dtype
        return dtype

    def _normalize_summation(
        self,
        sum_array: Any,
        axis: Any,
        ddof: int = 0,
        keepdims: bool = False,
        where: ndarray | None = None,
    ) -> None:
        dtype = sum_array.dtype
        if axis is None:
            if where is not None:
                divisor = where._count_nonzero() - ddof
            else:
                divisor = reduce(lambda x, y: x * y, self.shape, 1) - ddof
        else:
            if where is not None:
                divisor = where.sum(axis=axis, dtype=dtype, keepdims=keepdims)
                if not isinstance(divisor, ndarray):
                    divisor = convert_to_cupynumeric_ndarray(divisor)  # type: ignore [unreachable] # noqa: E501
                if ddof != 0 and not np.isscalar(divisor):
                    mask = divisor != 0
                    if not isinstance(mask, ndarray):
                        mask = convert_to_cupynumeric_ndarray(mask)  # type: ignore [unreachable] # noqa: E501
                    values = divisor - ddof
                    if not isinstance(values, ndarray):
                        values = convert_to_cupynumeric_ndarray(values)  # type: ignore [unreachable] # noqa: E501
                    divisor._thunk.putmask(mask._thunk, values._thunk)
                else:
                    divisor -= ddof
            else:
                axis = normalize_axis_tuple(axis, self.ndim)
                divisor = (
                    reduce(
                        lambda x, y: x * y, (self.shape[d] for d in axis), 1
                    )
                    - ddof
                )

        # Divide by the number of things in the collapsed dimensions
        # Pick the right kinds of division based on the dtype
        if isinstance(divisor, ndarray):
            divisor = divisor.astype(dtype)
        else:
            divisor = np.array(divisor, dtype=dtype)  # type: ignore [assignment] # noqa

        if dtype.kind == "f" or dtype.kind == "c":
            sum_array.__itruediv__(divisor)
        else:
            sum_array.__ifloordiv__(divisor)

    @add_boilerplate()
    def mean(
        self,
        axis: Any = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        where: ndarray | None = None,
    ) -> ndarray:
        """a.mean(axis=None, dtype=None, out=None, keepdims=False)

        Returns the average of the array elements along given axis.

        Refer to :func:`cupynumeric.mean` for full documentation.

        See Also
        --------
        cupynumeric.mean : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        dtype = self._summation_dtype(dtype)
        where_array = broadcast_where(where, self.shape)

        # Do the sum
        sum_array = (
            self.sum(
                axis=axis,
                out=out,
                keepdims=keepdims,
                dtype=dtype,
                where=where_array,
            )
            if out is not None and out.dtype == dtype
            else self.sum(
                axis=axis, keepdims=keepdims, dtype=dtype, where=where_array
            )
        )

        self._normalize_summation(
            sum_array, axis, keepdims=keepdims, where=where_array
        )

        # Convert to the output we didn't already put it there
        if out is not None and sum_array is not out:
            assert out.dtype != sum_array.dtype
            out._thunk.convert(sum_array._thunk)
            return out
        else:
            return sum_array

    def _nanmean(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        where: ndarray | None = None,
    ) -> ndarray:
        if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, bool):
            return self.mean(
                axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
            )

        nan_mask = _ufunc.bit_twiddling.bitwise_not(
            _ufunc.floating.isnan(self)
        )
        if where is not None:
            nan_mask &= where
        return self.mean(
            axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=nan_mask
        )

    @add_boilerplate()
    def var(
        self,
        axis: int | tuple[int, ...] | None = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
        ddof: int = 0,
        keepdims: bool = False,
        *,
        where: ndarray | None = None,
    ) -> ndarray:
        """a.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)

        Returns the variance of the array elements along given axis.

        Refer to :func:`cupynumeric.var` for full documentation.

        See Also
        --------
        cupynumeric.var : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # this could be computed as a single pass through the array
        # by computing both <x^2> and <x> and then computing <x^2> - <x>^2.
        # this would takee the difference of two large numbers and is unstable
        # the mean needs to be computed first and the variance computed
        # directly as <(x-mu)^2>, which then requires two passes through the
        # data to first compute the mean and then compute the variance
        # see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # TODO(https://github.com/nv-legate/cupynumeric/issues/590)

        dtype = self._summation_dtype(dtype)
        # calculate the mean, but keep the dimensions so that the
        # mean can be broadcast against the original array
        mu = self.mean(axis=axis, dtype=dtype, keepdims=True, where=where)

        where_array = broadcast_where(where, self.shape)

        # 1D arrays (or equivalent) should benefit from this unary reduction:
        if axis is None:
            # this is a scalar reduction and we can optimize this as a single
            # pass through a scalar reduction
            result = perform_unary_reduction(
                UnaryRedCode.VARIANCE,
                self,
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
                where=where_array,
                # FIXME(wonchanl): the following code blocks on mu to convert
                # it to a Scalar object. We need to get rid of this blocking by
                # allowing the extra arguments to be Legate stores
                args=(Scalar(mu.__array__(), to_core_type(self.dtype)),),
            )
        else:
            # TODO(https://github.com/nv-legate/cupynumeric/issues/591)
            # there isn't really support for generic binary reductions
            # right now all of the current binary reductions are boolean
            # reductions like allclose. To implement this a single pass would
            # require a variant of einsum/dot that produces
            # (self-mu)*(self-mu) rather than self*mu. For now, we have to
            # compute delta = self-mu in a first pass and then compute
            # delta*delta in second pass
            delta = self - mu

            result = perform_unary_reduction(
                UnaryRedCode.SUM_SQUARES,
                delta,
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
                where=where_array,
            )

        self._normalize_summation(
            result, axis=axis, ddof=ddof, keepdims=keepdims, where=where_array
        )

        return result

    @add_boilerplate()
    def min(
        self,
        axis: Any = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        initial: int | float | None = None,
        where: ndarray | None = None,
    ) -> ndarray:
        """a.min(axis=None, out=None, keepdims=False, initial=<no value>,
                 where=True)

        Return the minimum along a given axis.

        Refer to :func:`cupynumeric.amin` for full documentation.

        See Also
        --------
        cupynumeric.amin : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return perform_unary_reduction(
            UnaryRedCode.MIN,
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def partition(
        self,
        kth: int | Sequence[int],
        axis: Any = -1,
        kind: SelectKind = "introselect",
        order: OrderType | None = None,
    ) -> None:
        """a.partition(kth, axis=-1, kind='introselect', order=None)

        Partition of an array in-place.

        Refer to :func:`cupynumeric.partition` for full documentation.

        See Also
        --------
        cupynumeric.partition : equivalent function

        Availability
        --------
        Multiple GPUs, Single CPU

        """
        check_writeable(self)
        self._thunk.partition(
            rhs=self._thunk, kth=kth, axis=axis, kind=kind, order=order
        )

    @add_boilerplate()
    def argpartition(
        self,
        kth: int | Sequence[int],
        axis: Any = -1,
        kind: SelectKind = "introselect",
        order: OrderType | None = None,
    ) -> ndarray:
        """a.argpartition(kth, axis=-1, kind='introselect', order=None)

        Returns the indices that would partition this array.

        Refer to :func:`cupynumeric.argpartition` for full documentation.

        See Also
        --------
        cupynumeric.argpartition : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        result = ndarray(self.shape, np.int64)
        result._thunk.partition(
            rhs=self._thunk,
            argpartition=True,
            kth=kth,
            axis=axis,
            kind=kind,
            order=order,
        )
        return result

    @add_boilerplate()
    def prod(
        self,
        axis: Any = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        initial: int | float | None = None,
        where: ndarray | None = None,
    ) -> ndarray:
        """a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1,
        where=True)

        Return the product of the array elements over the given axis

        Refer to :func:`cupynumeric.prod` for full documentation.

        See Also
        --------
        cupynumeric.prod : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return perform_unary_reduction(
            UnaryRedCode.PROD,
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def ravel(self, order: OrderType = "C") -> ndarray:
        """a.ravel(order="C")

        Return a flattened array.

        Refer to :func:`cupynumeric.ravel` for full documentation.

        See Also
        --------
        cupynumeric.ravel : equivalent function
        ndarray.flat : a flat iterator on the array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.reshape(-1, order=order)

    def reshape(self, *args: Any, order: OrderType = "C") -> ndarray:
        """a.reshape(shape, order='C')

        Returns an array containing the same data with a new shape.

        Refer to :func:`cupynumeric.reshape` for full documentation.

        See Also
        --------
        cupynumeric.reshape : equivalent function


        Availability
        --------
        Multiple GPUs, Multiple CPUs
        """
        if len(args) == 0:
            raise TypeError("reshape() takes exactly 1 argument (0 given)")
        elif len(args) == 1:
            shape = (args[0],) if isinstance(args[0], int) else args[0]
        else:
            shape = args

        if self.size == 0 and self.ndim > 1:
            if shape == (-1,):
                shape = (0,)
            new_size = prod(shape)
            if new_size > 0:
                raise ValueError("new shape has bigger size than original")
            result = ndarray._from_inputs(
                shape=shape, dtype=self.dtype, inputs=(self,)
            )
            result.fill(0)
            return result

        computed_shape = tuple(operator.index(extent) for extent in shape)

        num_unknowns = sum(extent < 0 for extent in computed_shape)
        if num_unknowns > 1:
            raise ValueError("can only specify one unknown dimension")

        knowns = filter(lambda x: x >= 0, computed_shape)
        known_volume = reduce(lambda x, y: x * y, knowns, 1)

        # Can't have an unknown if the known shape has 0 size
        if num_unknowns > 0 and known_volume == 0:
            raise ValueError(
                f"cannot reshape array of size {self.size} into shape {computed_shape}"
            )

        size = self.size
        unknown_extent = 1 if num_unknowns == 0 else size // known_volume

        if unknown_extent * known_volume != size:
            raise ValueError(
                f"cannot reshape array of size {size} into shape {computed_shape}"
            )

        computed_shape = tuple(
            unknown_extent if extent < 0 else extent
            for extent in computed_shape
        )

        # Handle an easy case
        if computed_shape == self.shape:
            return self

        return ndarray._from_thunk(self._thunk.reshape(computed_shape, order))

    def setfield(
        self, val: Any, dtype: npt.DTypeLike, offset: int = 0
    ) -> None:
        raise NotImplementedError(
            "cuPyNumeric does not currently support type reinterpretation "
            "for ndarray.setfield"
        )

    def setflags(
        self,
        write: bool | None = None,
        align: bool | None = None,
        uic: bool | None = None,
    ) -> None:
        """a.setflags(write=None, align=None, uic=None)

        Set array flags WRITEABLE, ALIGNED, WRITEBACKIFCOPY,
        respectively.

        These Boolean-valued flags affect how numpy interprets the memory
        area used by `a` (see Notes below). The ALIGNED flag can only
        be set to True if the data is actually aligned according to the type.
        The WRITEBACKIFCOPY and flag can never be set
        to True. The flag WRITEABLE can only be set to True if the array owns
        its own memory, or the ultimate owner of the memory exposes a
        writeable buffer interface, or is a string. (The exception for string
        is made so that unpickling can be done without copying memory.)

        Parameters
        ----------
        write : bool, optional
            Describes whether or not `a` can be written to.
        align : bool, optional
            Describes whether or not `a` is aligned properly for its type.
        uic : bool, optional
            Describes whether or not `a` is a copy of another "base" array.

        Notes
        -----
        Array flags provide information about how the memory area used
        for the array is to be interpreted. There are 7 Boolean flags
        in use, only four of which can be changed by the user:
        WRITEBACKIFCOPY, WRITEABLE, and ALIGNED.

        WRITEABLE (W) the data area can be written to;

        ALIGNED (A) the data and strides are aligned appropriately for the
        hardware (as determined by the compiler);

        WRITEBACKIFCOPY (X) this array is a copy of some other array
        (referenced by .base). When the C-API function
        PyArray_ResolveWritebackIfCopy is called, the base array will be
        updated with the contents of this array.

        All flags can be accessed using the single (upper case) letter as well
        as the full name.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # Be a bit more careful here, and only pass params that are explicitly
        # set by the caller. The numpy interface specifies only bool values,
        # despite its None defaults.
        if write is not None:
            self.flags["W"] = write
        if align is not None:
            self.flags["A"] = align
        if uic is not None:
            self.flags["X"] = uic

    @add_boilerplate()
    def searchsorted(
        self: ndarray,
        v: int | float | ndarray,
        side: SortSide = "left",
        sorter: ndarray | None = None,
    ) -> int | ndarray:
        """a.searchsorted(v, side='left', sorter=None)

        Find the indices into a sorted array a such that, if the corresponding
        elements in v were inserted before the indices, the order of a would be
        preserved.

        Parameters
        ----------
        v : scalar or array_like
            Values to insert into a.
        side : ``{'left', 'right'}``, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index. If there is no suitable
            index, return either 0 or N (where N is the length of a).
        sorter : 1-D array_like, optional
            Optional array of integer indices that sort array a into ascending
            order. They are typically the result of argsort.

        Returns
        -------
        indices : int or array_like[int]
            Array of insertion points with the same shape as v, or an integer
            if v is a scalar.

        Availability
        --------
        Multiple GPUs, Multiple CPUs
        """

        if self.ndim != 1:
            raise ValueError("Dimension mismatch: self must be a 1D array")

        # this is needed in case v is a scalar
        v_ndarray = convert_to_cupynumeric_ndarray(v)

        a = self
        # in case we have different dtypes we ned to find a common type
        if a.dtype != v_ndarray.dtype:
            ch_dtype = np.result_type(a.dtype, v_ndarray.dtype)

            if v_ndarray.dtype != ch_dtype:
                v_ndarray = v_ndarray.astype(ch_dtype)
            if a.dtype != ch_dtype:
                a = a.astype(ch_dtype)

        if sorter is not None and a.shape[0] > 1:
            if sorter.ndim != 1:
                raise ValueError(
                    "Dimension mismatch: sorter must be a 1D array"
                )
            if sorter.shape != a.shape:
                raise ValueError(
                    "Shape mismatch: sorter must have the same shape as self"
                )
            if not np.issubdtype(sorter.dtype, np.integer):
                raise ValueError(
                    "Dtype mismatch: sorter must be of integer type"
                )
            a = a.take(sorter).copy()

        result = ndarray._from_inputs(
            v_ndarray.shape, np.int64, inputs=(a, v_ndarray, sorter)
        )

        result._thunk.searchsorted(a._thunk, v_ndarray._thunk, side)
        return result

    def sort(
        self,
        axis: Any = -1,
        kind: SortType = "quicksort",
        order: OrderType | None = None,
    ) -> None:
        """a.sort(axis=-1, kind=None, order=None)

        Sort an array in-place.

        Refer to :func:`cupynumeric.sort` for full documentation.

        See Also
        --------
        cupynumeric.sort : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        check_writeable(self)
        self._thunk.sort(rhs=self._thunk, axis=axis, kind=kind, order=order)

    def argsort(
        self,
        axis: Any = -1,
        kind: SortType = "quicksort",
        order: OrderType | None = None,
    ) -> ndarray:
        """a.argsort(axis=-1, kind=None, order=None)

        Returns the indices that would sort this array.

        Refer to :func:`cupynumeric.argsort` for full documentation.

        See Also
        --------
        cupynumeric.argsort : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        result = ndarray(self.shape, np.int64)
        result._thunk.sort(
            rhs=self._thunk, argsort=True, axis=axis, kind=kind, order=order
        )
        return result

    def squeeze(self, axis: Any = None) -> ndarray:
        """a.squeeze(axis=None)

        Remove axes of length one from `a`.

        Refer to :func:`cupynumeric.squeeze` for full documentation.

        See Also
        --------
        cupynumeric.squeeze : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if axis is not None:
            computed_axis = normalize_axis_tuple(axis, self.ndim)
            if any(self.shape[ax] != 1 for ax in computed_axis):
                raise ValueError(
                    "can only select axes to squeeze out with size equal to one"
                )
        else:
            computed_axis = None

        thunk = self._thunk.squeeze(computed_axis)
        if self._thunk is thunk:
            return self
        return ndarray._from_thunk(thunk)

    @add_boilerplate()
    def sum(
        self,
        axis: Any = None,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        initial: int | float | None = None,
        where: ndarray | None = None,
    ) -> ndarray:
        """a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0,
        where=None)

        Return the sum of the array elements over the given axis.

        Refer to :func:`cupynumeric.sum` for full documentation.

        See Also
        --------
        cupynumeric.sum : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return perform_unary_reduction(
            UnaryRedCode.SUM,
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def _nansum(
        self,
        axis: Any = None,
        dtype: Any = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        initial: int | float | None = None,
        where: ndarray | None = None,
    ) -> ndarray:
        # Note that np.nansum and np.sum allow complex datatypes
        # so there are no "disallowed types" for this API

        if self.dtype.kind in ("f", "c"):
            unary_red_code = UnaryRedCode.NANSUM
        else:
            unary_red_code = UnaryRedCode.SUM

        return perform_unary_reduction(
            unary_red_code,
            self,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def swapaxes(self, axis1: Any, axis2: Any) -> ndarray:
        """a.swapaxes(axis1, axis2)

        Return a view of the array with `axis1` and `axis2` interchanged.

        Refer to :func:`cupynumeric.swapaxes` for full documentation.

        See Also
        --------
        cupynumeric.swapaxes : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if axis1 >= self.ndim:
            raise ValueError(
                "axis1=" + str(axis1) + " is too large for swapaxes"
            )
        if axis2 >= self.ndim:
            raise ValueError(
                "axis2=" + str(axis2) + " is too large for swapaxes"
            )
        return ndarray._from_thunk(self._thunk.swapaxes(axis1, axis2))

    def tofile(self, fid: Any, sep: str = "", format: str = "%s") -> None:
        """a.tofile(fid, sep="", format="%s")

        Write array to a file as text or binary (default).

        Data is always written in 'C' order, independent of the order of `a`.
        The data produced by this method can be recovered using the function
        fromfile().

        Parameters
        ----------
        fid : ``file`` or str or pathlib.Path
            An open file object, or a string containing a filename.
        sep : str
            Separator between array items for text output.
            If "" (empty), a binary file is written, equivalent to
            ``file.write(a.tobytes())``.
        format : str
            Format string for text file output.
            Each entry in the array is formatted to text by first converting
            it to the closest Python type, and then using "format" % item.

        Notes
        -----
        This is a convenience function for quick storage of array data.
        Information on endianness and precision is lost, so this method is not
        a good choice for files intended to archive data or transport data
        between machines with different endianness. Some of these problems can
        be overcome by outputting the data as text files, at the expense of
        speed and file size.

        When fid is a file object, array contents are directly written to the
        file, bypassing the file object's ``write`` method. As a result,
        tofile cannot be used with files objects supporting compression (e.g.,
        GzipFile) or file-like objects that do not support ``fileno()`` (e.g.,
        BytesIO).

        Availability
        --------
        Single CPU

        """
        return self.__array__().tofile(fid, sep=sep, format=format)

    def tobytes(self, order: OrderType = "C") -> bytes:
        """a.tobytes(order='C')

        Construct Python bytes containing the raw data bytes in the array.

        Constructs Python bytes showing a copy of the raw contents of
        data memory. The bytes object is produced in C-order by default.

        This behavior is controlled by the ``order`` parameter.

        Parameters
        ----------
        order : ``{'C', 'F', 'A'}``, optional
            Controls the memory layout of the bytes object. 'C' means C-order,
            'F' means F-order, 'A' (short for *Any*) means 'F' if `a` is
            Fortran contiguous, 'C' otherwise. Default is 'C'.

        Returns
        -------
        s : bytes
            Python bytes exhibiting a copy of `a`'s raw data.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().tobytes(order=order)

    def tolist(self) -> Any:
        """a.tolist()

        Return the array as an ``a.ndim``-levels deep nested list of Python
        scalars.

        Return a copy of the array data as a (nested) Python list.
        Data items are converted to the nearest compatible builtin Python
        type, via the `~cupynumeric.ndarray.item` function.

        If ``a.ndim`` is 0, then since the depth of the nested list is 0, it
        will not be a list at all, but a simple Python scalar.

        Parameters
        ----------
        None

        Returns
        -------
        y : Any
            The possibly nested list of array elements. (object, or list of
            object, or list of list of object, or ...)

        Notes
        -----
        The array may be recreated via ``a = cupynumeric.array(a.tolist())``,
        although this may sometimes lose precision.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().tolist()

    def tostring(self, order: OrderType = "C") -> bytes:
        """a.tostring(order='C')

        A compatibility alias for `tobytes`, with exactly the same behavior.
        Despite its name, it returns `bytes` not `str`.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__array__().tobytes(order=order)

    def transpose(self, axes: Any = None) -> ndarray:
        """a.transpose(axes=None)

        Returns a view of the array with axes transposed.

        For a 1-D array this has no effect, as a transposed vector is simply
        the same vector. To convert a 1-D array into a 2D column vector, an
        additional dimension must be added. `np.atleast2d(a).T` achieves this,
        as does `a[:, np.newaxis]`.

        For a 2-D array, this is a standard matrix transpose.

        For an n-D array, if axes are given, their order indicates how the
        axes are permuted (see Examples). If axes are not provided and
        ``a.shape = (i[0], i[1], ... i[n-2], i[n-1])``, then
        ``a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])``.

        Parameters
        ----------
        axes : None or tuple[int]

            * None or no argument: reverses the order of the axes.
            * tuple of ints: `i` in the `j`-th place in the tuple means `a`'s
                `i`-th axis becomes `a.transpose()`'s `j`-th axis.

        Returns
        -------
        out : ndarray
            View of `a`, with axes suitably permuted.

        See Also
        --------
        transpose : Equivalent function
        ndarray.T : Array property returning the array transposed.
        ndarray.reshape : Give a new shape to an array without changing its
            data.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.ndim == 1:
            return self
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        elif any(a >= self.ndim for a in axes):
            raise AxisError(
                "axis is out of bounds for array of dimension "
                + str(self.ndim)
            )
        elif len(axes) != self.ndim:
            raise ValueError(
                "axes must be the same size as ndim for transpose"
            )
        return ndarray._from_thunk(
            self._thunk.transpose(axes), writeable=self._writeable
        )

    def flip(self, axis: Any = None) -> ndarray:
        """
        Reverse the order of elements in an array along the given axis.

        The shape of the array is preserved, but the elements are reordered.

        Parameters
        ----------
        axis : None or int or tuple[int], optional
            Axis or axes along which to flip over. The default, axis=None, will
            flip over all of the axes of the input array.  If axis is negative
            it counts from the last to the first axis.

            If axis is a tuple of ints, flipping is performed on all of the
            axes specified in the tuple.

        Returns
        -------
        out : array_like
            A view of `m` with the entries of axis reversed.  Since a view is
            returned, this operation is done in constant time.

        Availability
        --------
        Single GPU, Single CPU

        """
        result = ndarray._from_inputs(
            shape=self.shape, dtype=self.dtype, inputs=(self,)
        )
        result._thunk.flip(self._thunk, axis)
        return result

    def view(
        self, dtype: npt.DTypeLike | None = None, type: type | None = None
    ) -> ndarray:
        """
        New view of array with the same data.

        Parameters
        ----------
        dtype : data-type or ndarray sub-class, optional
            Data-type descriptor of the returned view, e.g., float32 or int16.
            Omitting it results in the view having the same data-type as the
            input array. This argument can also be specified as an ndarray
            sub-class, which then specifies the type of the returned object
            (this is equivalent to setting the ``type`` parameter).
        type : ndarray sub-class, optional
            Type of the returned view, e.g., ndarray or matrix. Again, omission
            of the parameter results in type preservation.

        Notes
        -----
        cuPyNumeric does not currently support type reinterpretation, or
        conversion to ndarray sub-classes; use :func:`ndarray.__array__()` to
        convert to `numpy.ndarray`.

        See Also
        --------
        numpy.ndarray.view

        Availability
        --------
        Multiple GPUs, Multiple CPUs
        """
        if dtype is not None and dtype != self.dtype:
            raise NotImplementedError(
                "cuPyNumeric does not currently support type reinterpretation"
            )
        if type is not None:
            raise NotImplementedError(
                "cuPyNumeric does not currently support conversion to ndarray "
                "sub-classes; use __array__() to convert to numpy.ndarray"
            )
        return ndarray._from_thunk(self._thunk, writeable=self._writeable)

    def unique(self) -> ndarray:
        """a.unique()

        Find the unique elements of an array.

        Refer to :func:`cupynumeric.unique` for full documentation.

        See Also
        --------
        cupynumeric.unique : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return ndarray._from_thunk(self._thunk.unique())

    def _maybe_convert(self, dtype: np.dtype[Any], hints: Any) -> ndarray:
        if self.dtype == dtype:
            return self
        copy = ndarray._from_inputs(
            shape=self.shape, dtype=dtype, inputs=hints
        )
        copy._thunk.convert(self._thunk)
        return copy

    def _wrap(self, new_len: int) -> ndarray:
        if new_len == 1:
            idxs = tuple(0 for i in range(self.ndim))
            return self[idxs]

        out = ndarray._from_inputs(
            shape=(new_len,), dtype=self.dtype, inputs=(self,)
        )
        out._thunk._wrap(src=self._thunk, new_len=new_len)
        return out

    def stencil_hint(
        self, low_offsets: tuple[int, ...], high_offsets: tuple[int, ...]
    ) -> None:
        """
        Inform cuPyNumeric that this array will be used in a stencil
        computation in the following code.

        This allows cuPyNumeric to allocate space for the "ghost" elements
        ahead of time, rather than discovering the full extent of accesses
        incrementally, and thus avoid intermediate copies.

        For example, let's say we have a 1-D array A of size 10 and we want to
        partition A across two GPUs. By default, A would be partitioned equally
        and each GPU gets an instance of size 5 (GPU0 gets elements 0-4, and
        GPU1 gets 5-9 inclusive). Suppose we use A in the stencil computation
        `B = A[:9] + A[1:]`. The runtime would now need to adjust the
        partitioning such that GPU0 has elements 0-5 and GPU1 has elements 4-9
        inclusive. Since the original instance on GPU0 does not cover index 5,
        cuPyNumeric needs to allocate a full new instance that covers 0-5,
        leading to an extra copy. In this case, if the code calls
        `A.stencil_hint([1], [1])` to pre-allocate instances that contain the
        extra elements before it uses A, the extra copies can be avoided.

        Parameters
        ----------
        low_offsets: tuple[int]
            Stencil offsets towards the negative direction.
        high_offsets: tuple[int]
            Stencil offsets towards the positive direction.

        Notes
        -----
        This function currently does not behave as expected in the case where
        multiple CPU/OpenMP processors use the same system memory.

        Availability
        --------
        Multiple CPUs, Multiple GPUs
        """
        self._thunk.stencil_hint(low_offsets, high_offsets)
