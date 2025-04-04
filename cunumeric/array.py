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

import operator
from functools import reduce, wraps
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from legate.core import Array, Field
from legate.core.utils import OrderedSet
from numpy.core.multiarray import (  # type: ignore [attr-defined]
    normalize_axis_index,
)
from numpy.core.numeric import (  # type: ignore [attr-defined]
    normalize_axis_tuple,
)
from typing_extensions import ParamSpec

from .config import (
    BinaryOpCode,
    ConvertCode,
    FFTDirection,
    FFTNormalization,
    FFTType,
    ScanCode,
    UnaryOpCode,
    UnaryRedCode,
)
from .coverage import FALLBACK_WARNING, clone_class, is_implemented
from .runtime import runtime
from .types import NdShape
from .utils import (
    calculate_volume,
    deep_apply,
    dot_modes,
    to_core_dtype,
    tuple_pop,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from .thunk import NumPyThunk
    from .types import (
        BoundsMode,
        CastingKind,
        NdShapeLike,
        OrderType,
        SelectKind,
        SortSide,
        SortType,
    )

from math import prod

R = TypeVar("R")
P = ParamSpec("P")


_WARN_SINGLE_ELEM_ACCESS = (
    "cuNumeric detected a single-element access, and is proactively pulling "
    "the entire array onto a single memory, to serve this and future "
    "accesses. This may result in blocking and increased memory consumption. "
    "Looping over ndarray elements is not efficient, please consider using "
    "full-array operations instead."
)


def add_boilerplate(
    *array_params: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Adds required boilerplate to the wrapped cunumeric.ndarray or module-level
    function.

    Every time the wrapped function is called, this wrapper will:
    * Convert all specified array-like parameters, plus the special "out"
      parameter (if present), to cuNumeric ndarrays.
    * Convert the special "where" parameter (if present) to a valid predicate.
    """
    keys = OrderedSet(array_params)
    assert len(keys) == len(array_params)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        assert not hasattr(
            func, "__wrapped__"
        ), "this decorator must be the innermost"

        # For each parameter specified by name, also consider the case where
        # it's passed as a positional parameter.
        indices: OrderedSet[int] = OrderedSet()
        where_idx: Optional[int] = None
        out_idx: Optional[int] = None
        params = signature(func).parameters
        extra = keys - OrderedSet(params)
        assert len(extra) == 0, f"unknown parameter(s): {extra}"
        for idx, param in enumerate(params):
            if param == "where":
                where_idx = idx
            elif param == "out":
                out_idx = idx
            elif param in keys:
                indices.add(idx)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            assert (where_idx is None or len(args) <= where_idx) and (
                out_idx is None or len(args) <= out_idx
            ), "'where' and 'out' should be passed as keyword arguments"

            # Convert relevant arguments to cuNumeric ndarrays
            args = tuple(
                convert_to_cunumeric_ndarray(arg)
                if idx in indices and arg is not None
                else arg
                for (idx, arg) in enumerate(args)
            )
            for k, v in kwargs.items():
                if v is None:
                    continue
                elif k == "out":
                    kwargs[k] = convert_to_cunumeric_ndarray(v, share=True)
                    if not kwargs[k].flags.writeable:
                        raise ValueError("out is not writeable")
                elif (k in keys) or (k == "where"):
                    kwargs[k] = convert_to_cunumeric_ndarray(v)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def convert_to_cunumeric_ndarray(obj: Any, share: bool = False) -> ndarray:
    # If this is an instance of one of our ndarrays then we're done
    if isinstance(obj, ndarray):
        return obj
    # Ask the runtime to make a numpy thunk for this object
    thunk = runtime.get_numpy_thunk(obj, share=share)
    writeable = (
        obj.flags.writeable if isinstance(obj, np.ndarray) and share else True
    )
    return ndarray(shape=None, thunk=thunk, writeable=writeable)


def maybe_convert_to_np_ndarray(obj: Any) -> Any:
    """
    Converts cuNumeric arrays into NumPy arrays, otherwise has no effect.
    """
    from .ma import MaskedArray

    if isinstance(obj, (ndarray, MaskedArray)):
        return obj.__array__()
    return obj


def check_writeable(arr: Union[ndarray, tuple[ndarray, ...], None]) -> None:
    """
    Check if the current array is writeable
    This check needs to be manually inserted
    with consideration on the behavior of the corresponding method
    """
    if arr is None:
        return
    check_list = (arr,) if not isinstance(arr, tuple) else arr
    if any(not arr.flags.writeable for arr in check_list):
        raise ValueError("array is not writeable")


def broadcast_where(
    where: Union[ndarray, None], shape: NdShape
) -> Union[ndarray, None]:
    if where is not None and where.shape != shape:
        from .module import broadcast_to

        where = broadcast_to(where, shape)
    return where


class flagsobj:
    """
    Information about the memory layout of the array.

    These flags don't reflect the properties of the cuNumeric array, but
    rather the NumPy array that will be produced if the cuNumeric array is
    materialized on a single node.
    """

    def __init__(self, array: ndarray) -> None:
        # prevent infinite __setattr__ recursion
        object.__setattr__(self, "_array", array)

    def __repr__(self) -> str:
        return f"""\
  C_CONTIGUOUS : {self["C"]}
  F_CONTIGUOUS : {self["F"]}
  OWNDATA : {self["O"]}
  WRITEABLE : {self["W"]}
  ALIGNED : {self["A"]}
  WRITEBACKIFCOPY : {self["X"]}
"""

    def __eq__(self, other: Any) -> bool:
        flags = ("C", "F", "O", "W", "A", "X")
        if not isinstance(other, (flagsobj, np.core.multiarray.flagsobj)):
            return False

        return all(self[f] == other[f] for f in flags)  # type: ignore [index]

    def __getattr__(self, name: str) -> Any:
        if name == "writeable":
            return self._array._writeable
        flags = self._array.__array__().flags
        return getattr(flags, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "writeable":
            self._check_writeable(value)
            self._array._writeable = bool(value)
        else:
            flags = self._array.__array__().flags
            setattr(flags, name, value)

    def __getitem__(self, key: Any) -> bool:
        if key == "W":
            return self._array._writeable
        flags = self._array.__array__().flags
        return flags[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "W":
            self._check_writeable(value)
            self._array._writeable = bool(value)
        else:
            flags = self._array.__array__().flags
            flags[key] = value

    def _check_writeable(self, value: Any) -> None:
        if value and not self._array._writeable:
            raise ValueError(
                "non-writeable cunumeric arrays cannot be made writeable"
            )


NDARRAY_INTERNAL = {
    "__array_finalize__",
    "__array_function__",
    "__array_interface__",
    "__array_prepare__",
    "__array_priority__",
    "__array_struct__",
    "__array_ufunc__",
    "__array_wrap__",
}


@clone_class(np.ndarray, NDARRAY_INTERNAL, maybe_convert_to_np_ndarray)
class ndarray:
    def __init__(
        self,
        shape: Any,
        dtype: npt.DTypeLike = np.float64,
        buffer: Union[Any, None] = None,
        offset: int = 0,
        strides: Union[tuple[int], None] = None,
        order: Union[OrderType, None] = None,
        thunk: Union[NumPyThunk, None] = None,
        inputs: Union[Any, None] = None,
        writeable: bool = True,
    ) -> None:
        # `inputs` being a cuNumeric ndarray is definitely a bug
        assert not isinstance(inputs, ndarray)
        if thunk is None:
            assert shape is not None
            sanitized_shape = self._sanitize_shape(shape)
            if not isinstance(dtype, np.dtype):
                dtype = np.dtype(dtype)
            if buffer is not None:
                # Make a normal numpy array for this buffer
                np_array: npt.NDArray[Any] = np.ndarray(
                    shape=sanitized_shape,
                    dtype=dtype,
                    buffer=buffer,
                    offset=offset,
                    strides=strides,
                    order=order,
                )
                self._thunk = runtime.find_or_create_array_thunk(
                    np_array, share=False
                )
            else:
                # Filter the inputs if necessary
                if inputs is not None:
                    inputs = [
                        inp._thunk
                        for inp in inputs
                        if isinstance(inp, ndarray)
                    ]
                core_dtype = to_core_dtype(dtype)
                self._thunk = runtime.create_empty_thunk(
                    sanitized_shape, core_dtype, inputs
                )
        else:
            self._thunk = thunk
        self._legate_data: Union[dict[str, Any], None] = None

        self._writeable = writeable

    @staticmethod
    def _sanitize_shape(
        shape: Union[NdShapeLike, Sequence[Any], npt.NDArray[Any], ndarray]
    ) -> NdShape:
        seq: tuple[Any, ...]
        if isinstance(shape, (ndarray, np.ndarray)):
            if shape.ndim == 0:
                seq = (shape.__array__().item(),)
            else:
                seq = tuple(shape.__array__())
        elif np.isscalar(shape):
            seq = (shape,)
        else:
            seq = tuple(cast(NdShape, shape))
        try:
            # Unfortunately, we can't do this check using
            # 'isinstance(value, int)', as the values in a NumPy ndarray
            # don't satisfy the predicate (they have numpy value types,
            # such as numpy.int64).
            result = tuple(operator.index(value) for value in seq)
        except TypeError:
            raise TypeError(
                "expected a sequence of integers or a single integer, "
                f"got {shape!r}"
            )
        return result

    # Support for the Legate data interface
    @property
    def __legate_data_interface__(self) -> dict[str, Any]:
        if self._legate_data is None:
            # If the thunk is an eager array, we need to convert it to a
            # deferred array so we can extract a legate store
            deferred_thunk = runtime.to_deferred_array(self._thunk)
            # We don't have nullable data for the moment
            # until we support masked arrays
            dtype = deferred_thunk.base.type
            array = Array(dtype, [None, deferred_thunk.base])
            self._legate_data = dict()
            self._legate_data["version"] = 1
            field = Field("cuNumeric Array", dtype)
            self._legate_data["data"] = {field: array}
        return self._legate_data

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
        import cunumeric as cn

        what = func.__name__

        for t in types:
            # Be strict about which types we support.  Accept superclasses
            # (for basic subclassing support) and NumPy.
            if not issubclass(type(self), t) and t is not np.ndarray:
                return NotImplemented

        # We are wrapping all NumPy modules, so we can expect to find every
        # NumPy API call in cuNumeric, even if just an "unimplemented" stub.
        module = reduce(getattr, func.__module__.split(".")[1:], cn)
        cn_func = getattr(module, func.__name__)

        # We can't immediately forward to the corresponding cuNumeric
        # entrypoint. Say that we reached this point because the user code
        # invoked `np.foo(x, bar=True)` where `x` is a `cunumeric.ndarray`. If
        # our implementation of `foo` is not complete, and cannot handle
        # `bar=True`, then forwarding this call to `cn.foo` would fail. This
        # goes against the semantics of `__array_function__`, which shouldn't
        # fail if the custom implementation cannot handle the provided
        # arguments. Conversely, if the user calls `cn.foo(x, bar=True)`
        # directly, that means they requested the cuNumeric implementation
        # specifically, and the `NotImplementedError` should not be hidden.
        if is_implemented(cn_func):
            try:
                return cn_func(*args, **kwargs)
            except NotImplementedError:
                # Inform the user that we support the requested API in general,
                # but not this specific combination of arguments.
                what = f"the requested combination of arguments to {what}"

        # We cannot handle this call, so we will fall back to NumPy.
        runtime.warn(
            FALLBACK_WARNING.format(what=what),
            category=RuntimeWarning,
        )
        args = deep_apply(args, maybe_convert_to_np_ndarray)
        kwargs = deep_apply(kwargs, maybe_convert_to_np_ndarray)
        return func(*args, **kwargs)

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        from . import _ufunc

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
        runtime.warn(
            FALLBACK_WARNING.format(what=what),
            category=RuntimeWarning,
        )
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
        cunumeric.transpose
        ndarray.transpose

        """
        return self.transpose()

    @property
    def base(self) -> Union[npt.NDArray[Any], None]:
        """
        Returns dtype for the base element of the subarrays,
        regardless of their dimension or shape.

        See Also
        --------
        numpy.dtype.subdtype

        """
        return self.__array__().base

    @property
    def data(self) -> memoryview:
        """
        Python buffer object pointing to the start of the array's data.

        """
        return self.__array__().data

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

        These flags don't reflect the properties of the cuNumeric array, but
        rather the NumPy array that will be produced if the cuNumeric array is
        materialized on a single node.

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
        --------
        Single CPU

        """
        return self.__array__().flat

    @property
    def imag(self) -> ndarray:
        """
        The imaginary part of the array.

        """
        if self.dtype.kind == "c":
            return ndarray(shape=self.shape, thunk=self._thunk.imag())
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
            return ndarray(shape=self.shape, thunk=self._thunk.real())
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
        from ._ufunc import absolute

        return absolute(self)

    def __add__(self, rhs: Any) -> ndarray:
        """a.__add__(value, /)

        Return ``self+value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import add

        return add(self, rhs)

    def __and__(self, rhs: Any) -> ndarray:
        """a.__and__(value, /)

        Return ``self&value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_and

        return bitwise_and(self, rhs)

    def __array__(
        self, dtype: Union[np.dtype[Any], None] = None
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
        return self._perform_unary_reduction(
            UnaryRedCode.CONTAINS,
            self,
            axis=None,
            res_dtype=bool,
            args=args,
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
        result = ndarray(self.shape, self.dtype, inputs=(self,))
        result._thunk.copy(self._thunk, deep=False)
        return result

    def __deepcopy__(self, memo: Union[Any, None] = None) -> ndarray:
        """a.__deepcopy__(memo, /)

        Deep copy of array.

        Used if :func:`copy.deepcopy` is called on an array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        result = ndarray(self.shape, self.dtype, inputs=(self,))
        result._thunk.copy(self._thunk, deep=True)
        return result

    def __div__(self, rhs: Any) -> ndarray:
        """a.__div__(value, /)

        Return ``self/value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__truediv__(rhs)

    def __divmod__(self, rhs: Any) -> ndarray:
        """a.__divmod__(value, /)

        Return ``divmod(self, value)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        raise NotImplementedError(
            "cunumeric.ndarray doesn't support __divmod__ yet"
        )

    def __eq__(self, rhs: object) -> ndarray:  # type: ignore [override]
        """a.__eq__(value, /)

        Return ``self==value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import equal

        return equal(self, rhs)

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
        from ._ufunc import floor_divide

        return floor_divide(self, rhs)

    def __format__(self, *args: Any, **kwargs: Any) -> str:
        return self.__array__().__format__(*args, **kwargs)

    def __ge__(self, rhs: Any) -> ndarray:
        """a.__ge__(value, /)

        Return ``self>=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import greater_equal

        return greater_equal(self, rhs)

    # __getattribute__

    def _convert_key(self, key: Any, first: bool = True) -> Any:
        # Convert any arrays stored in a key to a cuNumeric array
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
            # Otherwise convert it to a cuNumeric array, check types
            # and get the thunk
            key = convert_to_cunumeric_ndarray(key)
            if key.dtype != bool and not np.issubdtype(key.dtype, np.integer):
                raise TypeError("index arrays should be int or bool type")
            if key.dtype != bool:
                key = key._warn_and_convert(np.dtype(np.int64))

            return key._thunk

    def _is_single_elem_access(self, key: Any) -> bool:
        # Just do a quick check to catch literal uses of scalar indices
        return (
            isinstance(key, tuple)
            and len(key) == self.ndim
            and all(np.isscalar(k) for k in key)
        )

    @add_boilerplate()
    def __getitem__(self, key: Any) -> ndarray:
        """a.__getitem__(key, /)

        Return ``self[key]``.

        """
        key = self._convert_key(key)
        if self._is_single_elem_access(key):
            runtime.warn(
                _WARN_SINGLE_ELEM_ACCESS,
                category=RuntimeWarning,
            )
            return convert_to_cunumeric_ndarray(self.__array__()[key])
        return ndarray(shape=None, thunk=self._thunk.get_item(key))

    def __gt__(self, rhs: Any) -> ndarray:
        """a.__gt__(value, /)

        Return ``self>value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import greater

        return greater(self, rhs)

    def __hash__(self) -> int:
        raise TypeError("unhashable type: cunumeric.ndarray")

    def __iadd__(self, rhs: Any) -> ndarray:
        """a.__iadd__(value, /)

        Return ``self+=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import add

        return add(self, rhs, out=self)

    def __iand__(self, rhs: Any) -> ndarray:
        """a.__iand__(value, /)

        Return ``self&=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_and

        return bitwise_and(self, rhs, out=self)

    def __idiv__(self, rhs: Any) -> ndarray:
        """a.__idiv__(value, /)

        Return ``self/=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.__itruediv__(rhs)

    def __ifloordiv__(self, rhs: Any) -> ndarray:
        """a.__ifloordiv__(value, /)

        Return ``self//=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import floor_divide

        return floor_divide(self, rhs, out=self)

    def __ilshift__(self, rhs: Any) -> ndarray:
        """a.__ilshift__(value, /)

        Return ``self<<=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import left_shift

        return left_shift(self, rhs, out=self)

    def __imod__(self, rhs: Any) -> ndarray:
        """a.__imod__(value, /)

        Return ``self%=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import remainder

        return remainder(self, rhs, out=self)

    def __imul__(self, rhs: Any) -> ndarray:
        """a.__imul__(value, /)

        Return ``self*=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import multiply

        return multiply(self, rhs, out=self)

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
        if self.dtype == np.bool_:
            # Boolean values are special, just do logical NOT
            from ._ufunc import logical_not

            return logical_not(self)
        else:
            from ._ufunc import invert

            return invert(self)

    def __ior__(self, rhs: Any) -> ndarray:
        """a.__ior__(/)

        Return ``self|=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_or

        return bitwise_or(self, rhs, out=self)

    def __ipow__(self, rhs: float) -> ndarray:
        """a.__ipow__(/)

        Return ``self**=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import power

        return power(self, rhs, out=self)

    def __irshift__(self, rhs: Any) -> ndarray:
        """a.__irshift__(/)

        Return ``self>>=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import right_shift

        return right_shift(self, rhs, out=self)

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
        from ._ufunc import subtract

        return subtract(self, rhs, out=self)

    def __itruediv__(self, rhs: Any) -> ndarray:
        """a.__itruediv__(/)

        Return ``self/=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import true_divide

        return true_divide(self, rhs, out=self)

    def __ixor__(self, rhs: Any) -> ndarray:
        """a.__ixor__(/)

        Return ``self^=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_xor

        return bitwise_xor(self, rhs, out=self)

    def __le__(self, rhs: Any) -> ndarray:
        """a.__le__(value, /)

        Return ``self<=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import less_equal

        return less_equal(self, rhs)

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
        from ._ufunc import left_shift

        return left_shift(self, rhs)

    def __lt__(self, rhs: Any) -> ndarray:
        """a.__lt__(value, /)

        Return ``self<value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import less

        return less(self, rhs)

    def __matmul__(self, value: Any) -> ndarray:
        """a.__matmul__(value, /)

        Return ``self@value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.dot(value)

    def __mod__(self, rhs: Any) -> ndarray:
        """a.__mod__(value, /)

        Return ``self%value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import remainder

        return remainder(self, rhs)

    def __mul__(self, rhs: Any) -> ndarray:
        """a.__mul__(value, /)

        Return ``self*value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import multiply

        return multiply(self, rhs)

    def __ne__(self, rhs: object) -> ndarray:  # type: ignore [override]
        """a.__ne__(value, /)

        Return ``self!=value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import not_equal

        return not_equal(self, rhs)

    def __neg__(self) -> ndarray:
        """a.__neg__(value, /)

        Return ``-self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import negative

        return negative(self)

    # __new__

    @add_boilerplate()
    def nonzero(self) -> tuple[ndarray, ...]:
        """a.nonzero()

        Return the indices of the elements that are non-zero.

        Refer to :func:`cunumeric.nonzero` for full documentation.

        See Also
        --------
        cunumeric.nonzero : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        thunks = self._thunk.nonzero()
        return tuple(
            ndarray(shape=thunk.shape, thunk=thunk) for thunk in thunks
        )

    def __or__(self, rhs: Any) -> ndarray:
        """a.__or__(value, /)

        Return ``self|value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_or

        return bitwise_or(self, rhs)

    def __pos__(self) -> ndarray:
        """a.__pos__(value, /)

        Return ``+self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        # the positive opeartor is equivalent to copy
        from ._ufunc import positive

        return positive(self)

    def __pow__(self, rhs: float) -> ndarray:
        """a.__pow__(value, /)

        Return ``pow(self, value)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import power

        return power(self, rhs)

    def __radd__(self, lhs: Any) -> ndarray:
        """a.__radd__(value, /)

        Return ``value+self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import add

        return add(lhs, self)

    def __rand__(self, lhs: Any) -> ndarray:
        """a.__rand__(value, /)

        Return ``value&self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_and

        return bitwise_and(lhs, self)

    def __rdiv__(self, lhs: Any) -> ndarray:
        """a.__rdiv__(value, /)

        Return ``value/self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import true_divide

        return true_divide(lhs, self)

    def __rdivmod__(self, lhs: Any) -> ndarray:
        """a.__rdivmod__(value, /)

        Return ``divmod(value, self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        raise NotImplementedError(
            "cunumeric.ndarray doesn't support __rdivmod__ yet"
        )

    def __reduce__(
        self, *args: Any, **kwargs: Any
    ) -> Union[str, tuple[str, ...]]:
        """a.__reduce__(/)

        For pickling.

        """
        return self.__array__().__reduce__(*args, **kwargs)

    def __reduce_ex__(
        self, *args: Any, **kwargs: Any
    ) -> Union[str, tuple[str, ...]]:
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
        from ._ufunc import floor_divide

        return floor_divide(lhs, self)

    def __rmod__(self, lhs: Any) -> ndarray:
        """a.__rmod__(value, /)

        Return ``value%self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import remainder

        return remainder(lhs, self)

    def __rmul__(self, lhs: Any) -> ndarray:
        """a.__rmul__(value, /)

        Return ``value*self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import multiply

        return multiply(lhs, self)

    def __ror__(self, lhs: Any) -> ndarray:
        """a.__ror__(value, /)

        Return ``value|self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_or

        return bitwise_or(lhs, self)

    def __rpow__(self, lhs: Any) -> ndarray:
        """__rpow__(value, /)

        Return ``pow(value, self)``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import power

        return power(lhs, self)

    def __rshift__(self, rhs: Any) -> ndarray:
        """a.__rshift__(value, /)

        Return ``self>>value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import right_shift

        return right_shift(self, rhs)

    def __rsub__(self, lhs: Any) -> ndarray:
        """a.__rsub__(value, /)

        Return ``value-self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import subtract

        return subtract(lhs, self)

    def __rtruediv__(self, lhs: Any) -> ndarray:
        """a.__rtruediv__(value, /)

        Return ``value/self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import true_divide

        return true_divide(lhs, self)

    def __rxor__(self, lhs: Any) -> ndarray:
        """a.__rxor__(value, /)

        Return ``value^self``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_xor

        return bitwise_xor(lhs, self)

    # __setattr__
    def __setitem__(self, key: Any, raw_value: Any) -> None:
        """__setitem__(key, value, /)

        Set ``self[key]=value``.

        """
        check_writeable(self)
        key = self._convert_key(key)
        if self._is_single_elem_access(key):
            runtime.warn(
                _WARN_SINGLE_ELEM_ACCESS,
                category=RuntimeWarning,
            )
            if self._thunk.can_write_through_numpy_array:
                self.__array__()[key] = raw_value
                return
        value = convert_to_cunumeric_ndarray(raw_value)
        if value.dtype != self.dtype:
            temp = ndarray(value.shape, dtype=self.dtype, inputs=(value,))
            temp._thunk.convert(value._thunk)
            value = temp
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
        from ._ufunc import subtract

        return subtract(self, rhs)

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
        from ._ufunc import true_divide

        return true_divide(self, rhs)

    def __xor__(self, rhs: Any) -> ndarray:
        """a.__xor__(value, /)

        Return ``self^value``.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from ._ufunc import bitwise_xor

        return bitwise_xor(self, rhs)

    @add_boilerplate()
    def all(
        self,
        axis: Any = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        initial: Union[int, float, None] = None,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.all(axis=None, out=None, keepdims=False, initial=None, where=True)

        Returns True if all elements evaluate to True.

        Refer to :func:`cunumeric.all` for full documentation.

        See Also
        --------
        cunumeric.all : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._perform_unary_reduction(
            UnaryRedCode.ALL,
            self,
            axis=axis,
            res_dtype=bool,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def any(
        self,
        axis: Any = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        initial: Union[int, float, None] = None,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.any(axis=None, out=None, keepdims=False, initial=None, where=True)

        Returns True if any of the elements of `a` evaluate to True.

        Refer to :func:`cunumeric.any` for full documentation.

        See Also
        --------
        cunumeric.any : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._perform_unary_reduction(
            UnaryRedCode.ANY,
            self,
            axis=axis,
            res_dtype=bool,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    @add_boilerplate()
    def argmax(
        self,
        axis: Any = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """a.argmax(axis=None, out=None)

        Return indices of the maximum values along the given axis.

        Refer to :func:`cunumeric.argmax` for full documentation.

        See Also
        --------
        cunumeric.argmax : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if out is not None and out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
        if axis is not None and not isinstance(axis, int):
            raise ValueError("axis must be an integer")
        return self._perform_unary_reduction(
            UnaryRedCode.ARGMAX,
            self,
            axis=axis,
            res_dtype=np.dtype(np.int64),
            out=out,
            keepdims=keepdims,
        )

    @add_boilerplate()
    def argmin(
        self,
        axis: Any = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
    ) -> ndarray:
        """a.argmin(axis=None, out=None)

        Return indices of the minimum values along the given axis.

        Refer to :func:`cunumeric.argmin` for detailed documentation.

        See Also
        --------
        cunumeric.argmin : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if out is not None and out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
        if axis is not None and not isinstance(axis, int):
            raise ValueError("axis must be an integer")
        return self._perform_unary_reduction(
            UnaryRedCode.ARGMIN,
            self,
            axis=axis,
            res_dtype=np.dtype(np.int64),
            out=out,
            keepdims=keepdims,
        )

    def astype(
        self,
        dtype: npt.DTypeLike,
        order: OrderType = "C",
        casting: CastingKind = "unsafe",
        subok: bool = True,
        copy: bool = True,
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
            By default, astype always returns a newly allocated array. If this
            is set to false, and the `dtype`, `order`, and `subok`
            requirements are satisfied, the input array is returned instead
            of a copy.

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
        result = ndarray(self.shape, dtype=dtype, inputs=(self,))
        result._thunk.convert(self._thunk, warn=False, temporary=temporary)
        return result

    @add_boilerplate()
    def take(
        self,
        indices: Any,
        axis: Any = None,
        out: Union[ndarray, None] = None,
        mode: BoundsMode = "raise",
    ) -> ndarray:
        """a.take(indices, axis=None, out=None, mode="raise")

        Take elements from an array along an axis.

        Refer to :func:`cunumeric.take` for full documentation.

        See Also
        --------
        cunumeric.take : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if not np.isscalar(indices):
            # if indices is a tuple or list, bring sub-tuples to the same shape
            # and concatenate them
            indices = convert_to_cunumeric_ndarray(indices)

        if axis is None:
            self = self.ravel()
            axis = 0
        else:
            axis = normalize_axis_index(axis, self.ndim)

        # TODO remove "raise" logic when bounds check for advanced
        # indexing is implementd
        if mode == "raise":
            if np.isscalar(indices):
                if (indices < -self.shape[axis]) or (
                    indices >= self.shape[axis]
                ):
                    raise IndexError("invalid entry in indices array")
            else:
                if (indices < -self.shape[axis]).any() or (
                    indices >= self.shape[axis]
                ).any():
                    raise IndexError("invalid entry in indices array")
        elif mode == "wrap":
            indices = indices % self.shape[axis]
        elif mode == "clip":
            if np.isscalar(indices):
                if indices >= self.shape[axis]:
                    indices = self.shape[axis] - 1
                if indices < 0:
                    indices = 0
            else:
                indices = indices.clip(0, self.shape[axis] - 1)
        else:
            raise ValueError(
                "Invalid mode '{}' for take operation".format(mode)
            )
        if self.shape[axis] == 0:
            if indices.size != 0:
                raise IndexError(
                    "Cannot do a non-empty take() from an empty axis."
                )
            return self.copy()

        point_indices = tuple(slice(None) for i in range(0, axis))
        point_indices += (indices,)
        if out is not None:
            if out.dtype != self.dtype:
                raise TypeError("Type mismatch: out array has the wrong type")
            out[:] = self[point_indices]
            return out
        else:
            res = self[point_indices]
            if np.isscalar(indices):
                res = res.copy()
            return res

    @add_boilerplate()
    def choose(
        self,
        choices: Any,
        out: Union[ndarray, None] = None,
        mode: BoundsMode = "raise",
    ) -> ndarray:
        """a.choose(choices, out=None, mode='raise')

        Use an index array to construct a new array from a set of choices.

        Refer to :func:`cunumeric.choose` for full documentation.

        See Also
        --------
        cunumeric.choose : equivalent function

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
                convert_to_cunumeric_ndarray(choices[i]).astype(ch_dtype)
                for i in range(n)
            )

        else:
            choices = convert_to_cunumeric_ndarray(choices)
            n = choices.shape[0]
            ch_dtype = choices.dtype
            choices = tuple(choices[i, ...] for i in range(n))

        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError("a array should be integer type")

        if self.dtype != np.int64:
            a = a.astype(np.int64)
        if mode == "raise":
            if (a < 0).any() or (a >= n).any():
                raise ValueError("invalid entry in choice array")
        elif mode == "wrap":
            a = a % n
        elif mode == "clip":
            a = a.clip(0, n - 1)
        else:
            raise ValueError(
                f"mode={mode} not understood. Must "
                "be 'raise', 'wrap', or 'clip'"
            )

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
            out_arr = ndarray(
                shape=out_shape,
                dtype=ch_dtype,
                inputs=(a, choices),
            )

        ch = tuple(c._thunk for c in choices)
        out_arr._thunk.choose(a._thunk, *ch)

        if out is not None and out.dtype != ch_dtype:
            out._thunk.convert(out_arr._thunk)
            return out

        return out_arr

    @add_boilerplate()
    def compress(
        self,
        condition: ndarray,
        axis: Any = None,
        out: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.compress(self, condition, axis=None, out=None)

        Return selected slices of an array along given axis.

        Refer to :func:`cunumeric.compress` for full documentation.

        See Also
        --------
        cunumeric.compress : equivalent function

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
        condition = condition._warn_and_convert(np.dtype(bool))

        if axis is None:
            axis = 0
            a = self.ravel()
        else:
            axis = normalize_axis_index(axis, self.ndim)

        if a.shape[axis] < condition.shape[0]:
            raise ValueError(
                "Shape mismatch: "
                "condition contains entries that are out of bounds"
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
        min: Union[int, float, npt.ArrayLike, None] = None,
        max: Union[int, float, npt.ArrayLike, None] = None,
        out: Union[npt.NDArray[Any], ndarray, None] = None,
    ) -> ndarray:
        """a.clip(min=None, max=None, out=None)

        Return an array whose values are limited to ``[min, max]``.

        One of max or min must be given.

        Refer to :func:`cunumeric.clip` for full documentation.

        See Also
        --------
        cunumeric.clip : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        min = (
            min
            if min is not None
            else np.iinfo(self.dtype).min
            if self.dtype.kind == "i"
            else -np.inf
        )
        max = (
            max
            if max is not None
            else np.iinfo(self.dtype).max
            if self.dtype.kind == "i"
            else np.inf
        )
        args = (
            np.array(min, dtype=self.dtype),
            np.array(max, dtype=self.dtype),
        )
        if args[0].size != 1 or args[1].size != 1:
            runtime.warn(
                "cuNumeric has not implemented clip with array-like "
                "arguments and is falling back to canonical numpy. You "
                "may notice significantly decreased performance for this "
                "function call.",
                category=RuntimeWarning,
            )
            if isinstance(out, np.ndarray):
                self.__array__().clip(args[0], args[1], out=out)
                return convert_to_cunumeric_ndarray(out, share=True)
            elif isinstance(out, ndarray):
                self.__array__().clip(args[0], args[1], out=out.__array__())
                return out
            else:
                return convert_to_cunumeric_ndarray(
                    self.__array__().clip(args[0], args[1])
                )
        return self._perform_unary_op(
            UnaryOpCode.CLIP, self, out=out, extra_args=args
        )

    def conj(self) -> ndarray:
        """a.conj()

        Complex-conjugate all elements.

        Refer to :func:`cunumeric.conjugate` for full documentation.

        See Also
        --------
        cunumeric.conjugate : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.dtype.kind == "c":
            result = self._thunk.conj()
            return ndarray(self.shape, dtype=self.dtype, thunk=result)
        else:
            return self

    def conjugate(self) -> ndarray:
        """a.conjugate()

        Return the complex conjugate, element-wise.

        Refer to :func:`cunumeric.conjugate` for full documentation.

        See Also
        --------
        cunumeric.conjugate : equivalent function

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
        # We don't care about dimension order in cuNumeric
        return self.__copy__()

    @add_boilerplate()
    def cumsum(
        self,
        axis: Any = None,
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
    ) -> ndarray:
        return self._perform_scan(
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
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
    ) -> ndarray:
        return self._perform_scan(
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
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
    ) -> ndarray:
        return self._perform_scan(
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
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
    ) -> ndarray:
        return self._perform_scan(
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
        axes: Union[Any, None] = None,
        extract: bool = True,
        trace: bool = False,
        out: Union[ndarray, None] = None,
        dtype: Union[np.dtype[Any], None] = None,
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
                    "Axes shouldn't be specified when getting "
                    "diagonal for 1D array"
                )
            m = self.shape[0] + np.abs(offset)
            res = ndarray((m, m), dtype=self.dtype, inputs=(self,))
            diag_size = self.shape[0]
            res._thunk._diag_helper(
                self._thunk, offset=offset, naxes=0, extract=False, trace=False
            )
        else:
            assert axes is not None
            N = len(axes)
            if len(axes) != len(OrderedSet(axes)):
                raise ValueError(
                    "axes passed to _diag_helper should be all different"
                )
            if self.ndim < N:
                raise ValueError(
                    "Dimension of input array shouldn't be less "
                    "than number of axes"
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
                    raise ValueError(
                        "'offset' for diag or diagonal must be in range"
                    )

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
                    "number of axes passed to the _diag_helper"
                    " should be more than 1"
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
                        "dimension of the array for trace operation:"
                        " should be >=2"
                    )
            else:
                out_shape = tr_shape + (diag_size,)

            res_dtype = (
                dtype
                if dtype is not None
                else out.dtype
                if out is not None
                else a.dtype
            )
            a = a._maybe_convert(res_dtype, (a,))
            if out is not None and out.shape != out_shape:
                raise ValueError("output array has the wrong shape")
            if out is not None and out.dtype == res_dtype:
                res = out
            else:
                res = ndarray(shape=out_shape, dtype=res_dtype, inputs=(self,))

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

        Refer to :func:`cunumeric.diagonal` for full documentation.

        See Also
        --------
        cunumeric.diagonal : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.ndim == 1:
            if extract is True:
                raise ValueError("extract can be true only for Ndim >=2")
            axes = None
        else:
            axes = (axis1, axis2)
        return self._diag_helper(offset=offset, axes=axes, extract=extract)

    @add_boilerplate("indices", "values")
    def put(
        self, indices: ndarray, values: ndarray, mode: str = "raise"
    ) -> None:
        """
        Replaces specified elements of the array with given values.

        Refer to :func:`cunumeric.put` for full documentation.

        See Also
        --------
        cunumeric.put : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        check_writeable(self)

        if values.size == 0 or indices.size == 0 or self.size == 0:
            return

        if mode not in ("raise", "wrap", "clip"):
            raise ValueError(
                "mode must be one of 'clip', 'raise', or 'wrap' "
                f"(got  {mode})"
            )

        if mode == "wrap":
            indices = indices % self.size
        elif mode == "clip":
            indices = indices.clip(0, self.size - 1)

        indices = indices._warn_and_convert(np.dtype(np.int64))
        values = values._warn_and_convert(self.dtype)

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
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.trace(offset=0, axis1=None, axis2=None, dtype = None, out = None)

        Return the sum along diagonals of the array.

        Refer to :func:`cunumeric.trace` for full documentation.

        See Also
        --------
        cunumeric.trace : equivalent function

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
    def dot(self, rhs: ndarray, out: Union[ndarray, None] = None) -> ndarray:
        """a.dot(rhs, out=None)

        Return the dot product of this array with ``rhs``.

        Refer to :func:`cunumeric.dot` for full documentation.

        See Also
        --------
        cunumeric.dot : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        from .module import _contract  # work around circular import

        if self.ndim == 0 or rhs.ndim == 0:
            from ._ufunc import multiply

            return multiply(self, rhs, out=out)

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

    def dump(self, file: Union[str, Path]) -> None:
        """a.dump(file)

        Dump a pickle of the array to the specified file.

        The array can be read back with pickle.load or cunumeric.load.

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
        axes: Union[Sequence[int], None],
        kind: FFTType,
        direction: FFTDirection,
        norm: Any,
    ) -> ndarray:
        """a.fft(s, axes, kind, direction, norm)

        Return the ``kind`` ``direction`` FFT of this array
        with normalization ``norm``.

        Common entrypoint for FFT functionality in cunumeric.fft module.

        Notes
        -----
        Multi-GPU usage is limited to data parallel axis-wise batching.

        See Also
        --------
        cunumeric.fft : FFT functions for different ``kind`` and
        ``direction`` arguments

        Availability
        --------
        Multiple GPUs

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
            fft_input = ndarray(
                shape=fft_input_shape,
                thunk=zeropad_input._thunk.get_item(slices),
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
        out = ndarray(
            shape=fft_output_shape,
            dtype=fft_output_type,
        )
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
            return out / factor.astype(fft_output_type)

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
            "cuNumeric does not currently support type reinterpretation "
            "for ndarray.getfield"
        )

    def _convert_singleton_key(self, args: tuple[Any, ...]) -> Any:
        if len(args) == 0 and self.size == 1:
            return (0,) * self.ndim
        if len(args) == 1 and isinstance(args[0], int):
            flat_idx = args[0]
            result: tuple[int, ...] = ()
            for dim_size in reversed(self.shape):
                result = (flat_idx % dim_size,) + result
                flat_idx //= dim_size
            return result
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        if len(args) != self.ndim or any(not isinstance(x, int) for x in args):
            raise KeyError("invalid key")
        return args

    def item(self, *args: Any) -> npt.NDArray[Any]:
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

    def itemset(self, *args: Any) -> None:
        """a.itemset(*args)

        Insert scalar into an array (scalar is cast to array's dtype,
        if possible)

        There must be at least 1 argument, and define the last argument
        as *item*.  Then, ``a.itemset(*args)`` is equivalent to but faster
        than ``a[args] = item``.  The item should be a scalar value and `args`
        must select a single item in the array `a`.

        Parameters
        ----------
        \\*args :
            If one argument: a scalar, only used in case `a` is of size 1.
            If two arguments: the last argument is the value to be set
            and must be a scalar, the first argument specifies a single array
            element location. It is either an int or a tuple.

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
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        initial: Union[int, float, None] = None,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.max(axis=None, out=None, keepdims=False, initial=<no value>,
                 where=True)

        Return the maximum along a given axis.

        Refer to :func:`cunumeric.amax` for full documentation.

        See Also
        --------
        cunumeric.amax : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._perform_unary_reduction(
            UnaryRedCode.MAX,
            self,
            axis=axis,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )

    def _count_nonzero(self, axis: Any = None) -> Union[int, ndarray]:
        if self.size == 0:
            return 0
        return ndarray._perform_unary_reduction(
            UnaryRedCode.COUNT_NONZERO,
            self,
            res_dtype=np.dtype(np.uint64),
            axis=axis,
        )

    def _summation_dtype(
        self, dtype: Optional[np.dtype[Any]]
    ) -> np.dtype[Any]:
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
        where: Union[ndarray, None] = None,
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
                if ddof != 0 and not np.isscalar(divisor):
                    mask = divisor != 0
                    values = divisor - ddof
                    divisor._thunk.putmask(mask._thunk, values._thunk)
                else:
                    divisor -= ddof
            else:
                divisor = self.shape[axis] - ddof

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
        dtype: Optional[np.dtype[Any]] = None,
        out: Optional[ndarray] = None,
        keepdims: bool = False,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.mean(axis=None, dtype=None, out=None, keepdims=False)

        Returns the average of the array elements along given axis.

        Refer to :func:`cunumeric.mean` for full documentation.

        See Also
        --------
        cunumeric.mean : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if axis is not None and not isinstance(axis, int):
            raise NotImplementedError(
                "cunumeric.mean only supports int types for "
                "`axis` currently"
            )

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
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        from . import _ufunc

        if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
            return self.mean(
                axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
            )

        nan_mask = _ufunc.bit_twiddling.bitwise_not(
            _ufunc.floating.isnan(self)
        )
        if where is not None:
            nan_mask &= where
        return self.mean(
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=nan_mask,
        )

    @add_boilerplate()
    def var(
        self,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        dtype: Optional[np.dtype[Any]] = None,
        out: Optional[ndarray] = None,
        ddof: int = 0,
        keepdims: bool = False,
        *,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False)

        Returns the variance of the array elements along given axis.

        Refer to :func:`cunumeric.var` for full documentation.

        See Also
        --------
        cunumeric.var : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if axis is not None and not isinstance(axis, int):
            raise NotImplementedError(
                "cunumeric.var only supports int types for `axis` currently"
            )

        # this could be computed as a single pass through the array
        # by computing both <x^2> and <x> and then computing <x^2> - <x>^2.
        # this would takee the difference of two large numbers and is unstable
        # the mean needs to be computed first and the variance computed
        # directly as <(x-mu)^2>, which then requires two passes through the
        # data to first compute the mean and then compute the variance
        # see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # TODO(https://github.com/nv-legate/cunumeric/issues/590)

        dtype = self._summation_dtype(dtype)
        # calculate the mean, but keep the dimensions so that the
        # mean can be broadcast against the original array
        mu = self.mean(axis=axis, dtype=dtype, keepdims=True, where=where)

        where_array = broadcast_where(where, self.shape)

        # 1D arrays (or equivalent) should benefit from this unary reduction:
        if axis is None or calculate_volume(tuple_pop(self.shape, axis)) == 1:
            # this is a scalar reduction and we can optimize this as a single
            # pass through a scalar reduction
            result = self._perform_unary_reduction(
                UnaryRedCode.VARIANCE,
                self,
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
                where=where_array,
                args=(mu,),
            )
        else:
            # TODO(https://github.com/nv-legate/cunumeric/issues/591)
            # there isn't really support for generic binary reductions
            # right now all of the current binary reductions are boolean
            # reductions like allclose. To implement this a single pass would
            # require a variant of einsum/dot that produces
            # (self-mu)*(self-mu) rather than self*mu. For now, we have to
            # compute delta = self-mu in a first pass and then compute
            # delta*delta in second pass
            delta = self - mu

            result = self._perform_unary_reduction(
                UnaryRedCode.SUM_SQUARES,
                delta,
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=keepdims,
                where=where_array,
            )

        self._normalize_summation(
            result,
            axis=axis,
            ddof=ddof,
            keepdims=keepdims,
            where=where_array,
        )

        return result

    @add_boilerplate()
    def min(
        self,
        axis: Any = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        initial: Union[int, float, None] = None,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.min(axis=None, out=None, keepdims=False, initial=<no value>,
                 where=True)

        Return the minimum along a given axis.

        Refer to :func:`cunumeric.amin` for full documentation.

        See Also
        --------
        cunumeric.amin : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self._perform_unary_reduction(
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
        kth: Union[int, Sequence[int]],
        axis: Any = -1,
        kind: SelectKind = "introselect",
        order: Union[OrderType, None] = None,
    ) -> None:
        """a.partition(kth, axis=-1, kind='introselect', order=None)

        Partition of an array in-place.

        Refer to :func:`cunumeric.partition` for full documentation.

        See Also
        --------
        cunumeric.partition : equivalent function

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
        kth: Union[int, Sequence[int]],
        axis: Any = -1,
        kind: SelectKind = "introselect",
        order: Union[OrderType, None] = None,
    ) -> ndarray:
        """a.argpartition(kth, axis=-1, kind='introselect', order=None)

        Returns the indices that would partition this array.

        Refer to :func:`cunumeric.argpartition` for full documentation.

        See Also
        --------
        cunumeric.argpartition : equivalent function

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
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        initial: Union[int, float, None] = None,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.prod(axis=None, dtype=None, out=None, keepdims=False, initial=1,
        where=True)

        Return the product of the array elements over the given axis

        Refer to :func:`cunumeric.prod` for full documentation.

        See Also
        --------
        cunumeric.prod : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.dtype.type == np.bool_:
            temp = ndarray(
                shape=self.shape,
                dtype=np.dtype(np.int32),
                inputs=(self,),
            )
            temp._thunk.convert(self._thunk)
            self_array = temp
        else:
            self_array = self
        return self._perform_unary_reduction(
            UnaryRedCode.PROD,
            self_array,
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

        Refer to :func:`cunumeric.ravel` for full documentation.

        See Also
        --------
        cunumeric.ravel : equivalent function
        ndarray.flat : a flat iterator on the array.

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        return self.reshape(-1, order=order)

    def reshape(self, *args: Any, order: OrderType = "C") -> ndarray:
        """a.reshape(shape, order='C')

        Returns an array containing the same data with a new shape.

        Refer to :func:`cunumeric.reshape` for full documentation.

        See Also
        --------
        cunumeric.reshape : equivalent function


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
            result = ndarray(
                shape=shape,
                dtype=self.dtype,
                inputs=(self,),
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
                f"cannot reshape array of size {self.size} into "
                f"shape {computed_shape}"
            )

        size = self.size
        unknown_extent = 1 if num_unknowns == 0 else size // known_volume

        if unknown_extent * known_volume != size:
            raise ValueError(
                f"cannot reshape array of size {size} into "
                f"shape {computed_shape}"
            )

        computed_shape = tuple(
            unknown_extent if extent < 0 else extent
            for extent in computed_shape
        )

        # Handle an easy case
        if computed_shape == self.shape:
            return self

        return ndarray(
            shape=None,
            thunk=self._thunk.reshape(computed_shape, order),
        )

    def setfield(
        self, val: Any, dtype: npt.DTypeLike, offset: int = 0
    ) -> None:
        raise NotImplementedError(
            "cuNumeric does not currently support type reinterpretation "
            "for ndarray.setfield"
        )

    def setflags(
        self,
        write: Union[bool, None] = None,
        align: Union[bool, None] = None,
        uic: Union[bool, None] = None,
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
        v: Union[int, float, ndarray],
        side: SortSide = "left",
        sorter: Optional[ndarray] = None,
    ) -> Union[int, ndarray]:
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
        v_ndarray = convert_to_cunumeric_ndarray(v)

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

        result = ndarray(
            v_ndarray.shape, np.int64, inputs=(a, v_ndarray, sorter)
        )

        result._thunk.searchsorted(a._thunk, v_ndarray._thunk, side)
        return result

    def sort(
        self,
        axis: Any = -1,
        kind: SortType = "quicksort",
        order: Union[OrderType, None] = None,
    ) -> None:
        """a.sort(axis=-1, kind=None, order=None)

        Sort an array in-place.

        Refer to :func:`cunumeric.sort` for full documentation.

        See Also
        --------
        cunumeric.sort : equivalent function

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
        order: Union[OrderType, None] = None,
    ) -> ndarray:
        """a.argsort(axis=-1, kind=None, order=None)

        Returns the indices that would sort this array.

        Refer to :func:`cunumeric.argsort` for full documentation.

        See Also
        --------
        cunumeric.argsort : equivalent function

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

        Refer to :func:`cunumeric.squeeze` for full documentation.

        See Also
        --------
        cunumeric.squeeze : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if axis is not None:
            computed_axis = normalize_axis_tuple(axis, self.ndim)
            if any(self.shape[ax] != 1 for ax in computed_axis):
                raise ValueError(
                    "can only select axes to squeeze out with size "
                    "equal to one"
                )
        else:
            computed_axis = None

        thunk = self._thunk.squeeze(computed_axis)
        if self._thunk is thunk:
            return self
        return ndarray(shape=None, thunk=thunk)

    @add_boilerplate()
    def sum(
        self,
        axis: Any = None,
        dtype: Union[np.dtype[Any], None] = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        initial: Union[int, float, None] = None,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        """a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0,
        where=None)

        Return the sum of the array elements over the given axis.

        Refer to :func:`cunumeric.sum` for full documentation.

        See Also
        --------
        cunumeric.sum : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        if self.dtype.type == np.bool_:
            temp = ndarray(
                shape=self.shape,
                dtype=np.dtype(np.int32),
                inputs=(self,),
            )
            temp._thunk.convert(self._thunk)
            self_array = temp
        else:
            self_array = self
        return self._perform_unary_reduction(
            UnaryRedCode.SUM,
            self_array,
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
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        initial: Optional[Union[int, float]] = None,
        where: Optional[ndarray] = None,
    ) -> ndarray:
        # Note that np.nansum and np.sum allow complex datatypes
        # so there are no "disallowed types" for this API

        if self.dtype.kind in ("f", "c"):
            unary_red_code = UnaryRedCode.NANSUM
        else:
            unary_red_code = UnaryRedCode.SUM

        return self._perform_unary_reduction(
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

        Refer to :func:`cunumeric.swapaxes` for full documentation.

        See Also
        --------
        cunumeric.swapaxes : equivalent function

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
        return ndarray(shape=None, thunk=self._thunk.swapaxes(axis1, axis2))

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
        type, via the `~cunumeric.ndarray.item` function.

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
        The array may be recreated via ``a = cunumeric.array(a.tolist())``,
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
        elif len(axes) != self.ndim:
            raise ValueError(
                "axes must be the same size as ndim for transpose"
            )
        return ndarray(
            shape=None,
            thunk=self._thunk.transpose(axes),
            writeable=self._writeable,
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
        result = ndarray(
            shape=self.shape,
            dtype=self.dtype,
            inputs=(self,),
        )
        result._thunk.flip(self._thunk, axis)
        return result

    def view(
        self,
        dtype: Union[npt.DTypeLike, None] = None,
        type: Union[type, None] = None,
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
        cuNumeric does not currently support type reinterpretation, or
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
                "cuNumeric does not currently support type reinterpretation"
            )
        if type is not None:
            raise NotImplementedError(
                "cuNumeric does not currently support conversion to ndarray "
                "sub-classes; use __array__() to convert to numpy.ndarray"
            )
        return ndarray(
            shape=self.shape,
            dtype=self.dtype,
            thunk=self._thunk,
            writeable=self._writeable,
        )

    def unique(self) -> ndarray:
        """a.unique()

        Find the unique elements of an array.

        Refer to :func:`cunumeric.unique` for full documentation.

        See Also
        --------
        cunumeric.unique : equivalent function

        Availability
        --------
        Multiple GPUs, Multiple CPUs

        """
        thunk = self._thunk.unique()
        return ndarray(shape=thunk.shape, thunk=thunk)

    @classmethod
    def _get_where_thunk(
        cls, where: Union[None, ndarray], out_shape: NdShape
    ) -> Union[None, NumPyThunk]:
        if where is None:
            return where
        if (
            not isinstance(where, ndarray)
            or where.dtype != np.bool_
            or where.shape != out_shape
        ):
            raise RuntimeError("should have converted this earlier")
        return where._thunk

    @staticmethod
    def find_common_type(*args: ndarray) -> np.dtype[Any]:
        """Determine common type following NumPy's coercion rules.

        Parameters
        ----------
        *args : ndarray
            A list of ndarrays

        Returns
        -------
        datatype : data-type
            The type that results from applying the NumPy type promotion rules
            to the arguments.
        """
        array_types = list()
        scalars = list()
        for array in args:
            if array.ndim == 0:
                scalars.append(array.dtype.type(0))
            else:
                array_types.append(array.dtype)
        return np.result_type(*array_types, *scalars)

    def _maybe_convert(self, dtype: np.dtype[Any], hints: Any) -> ndarray:
        if self.dtype == dtype:
            return self
        copy = ndarray(shape=self.shape, dtype=dtype, inputs=hints)
        copy._thunk.convert(self._thunk)
        return copy

    def _warn_and_convert(self, dtype: np.dtype[Any]) -> ndarray:
        if self.dtype != dtype:
            runtime.warn(
                f"converting array to {dtype} type",
                category=RuntimeWarning,
            )
            return self.astype(dtype)
        else:
            return self

    # For performing normal/broadcast unary operations
    @classmethod
    def _perform_unary_op(
        cls,
        op: UnaryOpCode,
        src: ndarray,
        out: Union[Any, None] = None,
        extra_args: Any = None,
        dtype: Union[np.dtype[Any], None] = None,
        out_dtype: Union[np.dtype[Any], None] = None,
    ) -> ndarray:
        if out is not None:
            # If the shapes don't match see if we can broadcast
            # This will raise an exception if they can't be broadcast together
            if np.broadcast_shapes(src.shape, out.shape) != out.shape:
                raise ValueError(
                    f"non-broadcastable output operand with shape {out.shape} "
                    f"doesn't match the broadcast shape {src.shape}"
                )
        else:
            # No output yet, so make one
            out_shape = src.shape

            if dtype is not None:
                out = ndarray(
                    shape=out_shape,
                    dtype=dtype,
                    inputs=(src,),
                )
            elif out_dtype is not None:
                out = ndarray(
                    shape=out_shape,
                    dtype=out_dtype,
                    inputs=(src,),
                )
            else:
                out = ndarray(
                    shape=out_shape,
                    dtype=src.dtype
                    if src.dtype.kind != "c"
                    else np.dtype(np.float32)
                    if src.dtype == np.dtype(np.complex64)
                    else np.dtype(np.float64),
                    inputs=(src,),
                )

        if out_dtype is None:
            if out.dtype != src.dtype and not (
                op == UnaryOpCode.ABSOLUTE and src.dtype.kind == "c"
            ):
                temp = ndarray(
                    out.shape,
                    dtype=src.dtype,
                    inputs=(src,),
                )
                temp._thunk.unary_op(
                    op,
                    src._thunk,
                    True,
                    extra_args,
                )
                out._thunk.convert(temp._thunk)
            else:
                out._thunk.unary_op(
                    op,
                    src._thunk,
                    True,
                    extra_args,
                )
        else:
            if out.dtype != out_dtype:
                temp = ndarray(
                    out.shape,
                    dtype=out_dtype,
                    inputs=(src,),
                )
                temp._thunk.unary_op(
                    op,
                    src._thunk,
                    True,
                    extra_args,
                )
                out._thunk.convert(temp._thunk)
            else:
                out._thunk.unary_op(
                    op,
                    src._thunk,
                    True,
                    extra_args,
                )
        return out

    # For performing reduction unary operations
    @classmethod
    def _perform_unary_reduction(
        cls,
        op: UnaryRedCode,
        src: ndarray,
        axis: Any = None,
        dtype: Union[np.dtype[Any], None] = None,
        res_dtype: Union[npt.DTypeLike, None] = None,
        out: Union[ndarray, None] = None,
        keepdims: bool = False,
        args: Union[Any, None] = None,
        initial: Union[int, float, None] = None,
        where: Union[ndarray, None] = None,
    ) -> ndarray:
        # When 'res_dtype' is not None, the input and output of the reduction
        # have different types. Such reduction operators don't take a dtype of
        # the accumulator
        if res_dtype is not None:
            assert dtype is None
            dtype = src.dtype
        else:
            # If 'dtype' exists, that determines both the accumulation dtype
            # and the output dtype
            if dtype is not None:
                res_dtype = dtype
            elif out is not None:
                dtype = out.dtype
                res_dtype = out.dtype
            else:
                dtype = src.dtype
                res_dtype = src.dtype

        # TODO: Need to require initial to be given when the array is empty
        #       or a where mask is given.
        if (
            op
            in (
                UnaryRedCode.ARGMAX,
                UnaryRedCode.ARGMIN,
                UnaryRedCode.MAX,
                UnaryRedCode.MIN,
            )
            and src.dtype.kind == "c"
        ):
            raise NotImplementedError(
                "(arg)max/min not supported for complex-type arrays"
            )

        if axis is None:
            axes = tuple(range(src.ndim))
        else:
            axes = normalize_axis_tuple(axis, src.ndim)

        out_shape: NdShape = ()
        for dim in range(src.ndim):
            if dim not in axes:
                out_shape += (src.shape[dim],)
            elif keepdims:
                out_shape += (1,)

        if out is None:
            out = ndarray(
                shape=out_shape, dtype=res_dtype, inputs=(src, where)
            )
        elif out.shape != out_shape:
            errmsg = (
                f"the output shapes do not match: expected {out_shape} "
                f"but got {out.shape}"
            )
            raise ValueError(errmsg)

        if dtype != src.dtype:
            src = src.astype(dtype)

        if out.dtype == res_dtype:
            result = out
        else:
            result = ndarray(
                shape=out_shape, dtype=res_dtype, inputs=(src, where)
            )

        where_array = broadcast_where(where, src.shape)
        result._thunk.unary_reduction(
            op,
            src._thunk,
            cls._get_where_thunk(where_array, src.shape),
            axis,
            axes,
            keepdims,
            args,
            initial,
        )

        if result is not out:
            out._thunk.convert(result._thunk)

        return out

    @classmethod
    def _perform_binary_reduction(
        cls,
        op: BinaryOpCode,
        one: ndarray,
        two: ndarray,
        dtype: np.dtype[Any],
        extra_args: Union[tuple[Any, ...], None] = None,
    ) -> ndarray:
        args = (one, two)

        # We only handle bool types here for now
        assert dtype is not None and dtype == np.dtype(np.bool_)
        # Collapsing down to a single value in this case
        # Check to see if we need to broadcast between inputs
        if one.shape != two.shape:
            broadcast = np.broadcast_shapes(one.shape, two.shape)
        else:
            broadcast = None

        common_type = cls.find_common_type(one, two)
        one_thunk = one._maybe_convert(common_type, args)._thunk
        two_thunk = two._maybe_convert(common_type, args)._thunk

        dst = ndarray(shape=(), dtype=dtype, inputs=args)
        dst._thunk.binary_reduction(
            op,
            one_thunk,
            two_thunk,
            broadcast,
            extra_args,
        )
        return dst

    @classmethod
    def _perform_where(
        cls, mask: ndarray, one: ndarray, two: ndarray
    ) -> ndarray:
        args = (mask, one, two)

        mask = mask._maybe_convert(np.dtype(np.bool_), args)

        common_type = cls.find_common_type(one, two)
        one = one._maybe_convert(common_type, args)
        two = two._maybe_convert(common_type, args)

        # Compute the output shape
        out_shape = np.broadcast_shapes(mask.shape, one.shape, two.shape)
        out = ndarray(shape=out_shape, dtype=common_type, inputs=args)
        out._thunk.where(mask._thunk, one._thunk, two._thunk)
        return out

    @classmethod
    def _perform_scan(
        cls,
        op: ScanCode,
        src: ndarray,
        axis: Any = None,
        dtype: Union[npt.DTypeLike, None] = None,
        out: Union[ndarray, None] = None,
        nan_to_identity: bool = False,
    ) -> ndarray:
        if src.dtype.kind != "c" and src.dtype.kind != "f":
            nan_to_identity = False
        if dtype is None:
            if out is None:
                if src.dtype.kind == "i":
                    # Set dtype to default platform integer
                    dtype = np.int_
                else:
                    dtype = src.dtype
            else:
                dtype = out.dtype
        # flatten input when axis is None
        if axis is None:
            axis = 0
            src_arr = src.ravel()
        else:
            axis = normalize_axis_index(axis, src.ndim)
            src_arr = src
        if out is not None:
            if dtype != out.dtype:
                # if out array is specified, its type overrules dtype
                dtype = out.dtype
            if out.shape != src_arr.shape:
                raise NotImplementedError(
                    "Varried output shape not supported. Output must have "
                    "same shape as input (same size if no axis is provided"
                )
        else:
            out = ndarray(shape=src_arr.shape, dtype=dtype)

        if dtype != src_arr.dtype:
            if nan_to_identity:
                if op is ScanCode.SUM:
                    nan_op = ConvertCode.SUM
                else:
                    nan_op = ConvertCode.PROD
                # If convert is called, it will handle NAN conversion
                nan_to_identity = False
            else:
                nan_op = ConvertCode.NOOP
            # convert input to temporary for type conversion
            temp = ndarray(shape=src_arr.shape, dtype=dtype)
            temp._thunk.convert(src_arr._thunk, nan_op=nan_op)
            src_arr = temp

        out._thunk.scan(
            op,
            src_arr._thunk,
            axis=axis,
            dtype=dtype,
            nan_to_identity=nan_to_identity,
        )
        return out

    def _wrap(self, new_len: int) -> ndarray:
        if new_len == 1:
            idxs = tuple(0 for i in range(self.ndim))
            return self[idxs]

        out = ndarray(
            shape=(new_len,),
            dtype=self.dtype,
            inputs=(self,),
        )
        out._thunk._wrap(src=self._thunk, new_len=new_len)
        return out
