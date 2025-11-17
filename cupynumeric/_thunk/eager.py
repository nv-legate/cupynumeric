# Copyright 2025 NVIDIA Corporation
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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Sequence,
    cast,
    Literal,
)

import numpy as np
from legate.core import Scalar

from .. import _ufunc
from .._utils import is_np2
from .._utils.array import is_advanced_indexing
from ..config import (
    FFT_C2R,
    FFT_D2Z,
    FFT_R2C,
    FFT_Z2D,
    BinaryOpCode,
    ConvertCode,
    FFTDirection,
    ScanCode,
    TransferType,
    UnaryOpCode,
    UnaryRedCode,
    WindowOpCode,
)
from ..runtime import runtime
from .deferred import DeferredArray, IndexKey
from .thunk import NumPyThunk

if TYPE_CHECKING:
    import numpy.typing as npt
    from typing_extensions import CapsuleType

    from .._ufunc.ufunc import binary_ufunc, unary_ufunc
    from ..config import BitGeneratorType, FFTType
    from ..types import (
        BitOrder,
        BoundsMode,
        CastingKind,
        ConvolveMethod,
        ConvolveMode,
        NdShape,
        OrderType,
        SelectKind,
        SortSide,
        SortType,
    )


_UNARY_OPS: dict[UnaryOpCode, Any] = {
    UnaryOpCode.ABSOLUTE: np.absolute,
    UnaryOpCode.ARCCOS: np.arccos,
    UnaryOpCode.ARCCOSH: np.arccosh,
    UnaryOpCode.ARCSIN: np.arcsin,
    UnaryOpCode.ARCSINH: np.arcsinh,
    UnaryOpCode.ARCTAN: np.arctan,
    UnaryOpCode.ARCTANH: np.arctanh,
    UnaryOpCode.CBRT: np.cbrt,
    UnaryOpCode.CEIL: np.ceil,
    UnaryOpCode.CONJ: np.conj,
    UnaryOpCode.COS: np.cos,
    UnaryOpCode.COSH: np.cosh,
    UnaryOpCode.DEG2RAD: np.deg2rad,
    UnaryOpCode.EXP2: np.exp2,
    UnaryOpCode.EXP: np.exp,
    UnaryOpCode.EXPM1: np.expm1,
    UnaryOpCode.FLOOR: np.floor,
    UnaryOpCode.FREXP: np.frexp,
    UnaryOpCode.INVERT: np.invert,
    UnaryOpCode.ISFINITE: np.isfinite,
    UnaryOpCode.ISINF: np.isinf,
    UnaryOpCode.ISNAN: np.isnan,
    UnaryOpCode.LOG10: np.log10,
    UnaryOpCode.LOG1P: np.log1p,
    UnaryOpCode.LOG2: np.log2,
    UnaryOpCode.LOG: np.log,
    UnaryOpCode.LOGICAL_NOT: np.logical_not,
    UnaryOpCode.MODF: np.modf,
    UnaryOpCode.NEGATIVE: np.negative,
    UnaryOpCode.POSITIVE: np.positive,
    UnaryOpCode.RAD2DEG: np.rad2deg,
    UnaryOpCode.RECIPROCAL: np.reciprocal,
    UnaryOpCode.RINT: np.rint,
    UnaryOpCode.SIGN: np.sign,
    UnaryOpCode.SIGNBIT: np.signbit,
    UnaryOpCode.SIN: np.sin,
    UnaryOpCode.SINH: np.sinh,
    UnaryOpCode.SQRT: np.sqrt,
    UnaryOpCode.SQUARE: np.square,
    UnaryOpCode.TAN: np.tan,
    UnaryOpCode.TANH: np.tanh,
    UnaryOpCode.TRUNC: np.trunc,
}

# Unary reduction operations that don't return the argument of the
# reduction operation
_UNARY_RED_OPS_WITHOUT_ARG: dict[UnaryRedCode, Any] = {
    UnaryRedCode.ALL: np.all,
    UnaryRedCode.ANY: np.any,
    UnaryRedCode.MAX: np.max,
    UnaryRedCode.MIN: np.min,
    UnaryRedCode.PROD: np.prod,
    UnaryRedCode.SUM: np.sum,
    UnaryRedCode.NANMAX: np.nanmax,
    UnaryRedCode.NANMIN: np.nanmin,
    UnaryRedCode.NANPROD: np.nanprod,
    UnaryRedCode.NANSUM: np.nansum,
}

# Unary reduction operations that return the argument of the
# reduction operation
_UNARY_RED_OPS_WITH_ARG: dict[UnaryRedCode, Any] = {
    UnaryRedCode.ARGMIN: np.argmin,
    UnaryRedCode.ARGMAX: np.argmax,
    UnaryRedCode.NANARGMAX: np.nanargmax,
    UnaryRedCode.NANARGMIN: np.nanargmin,
}

_BINARY_OPS: dict[BinaryOpCode, Any] = {
    BinaryOpCode.ADD: np.add,
    BinaryOpCode.ARCTAN2: np.arctan2,
    BinaryOpCode.BITWISE_AND: np.bitwise_and,
    BinaryOpCode.BITWISE_OR: np.bitwise_or,
    BinaryOpCode.BITWISE_XOR: np.bitwise_xor,
    BinaryOpCode.COPYSIGN: np.copysign,
    BinaryOpCode.DIVIDE: np.divide,
    BinaryOpCode.EQUAL: np.equal,
    BinaryOpCode.FLOAT_POWER: np.float_power,
    BinaryOpCode.FLOOR_DIVIDE: np.floor_divide,
    BinaryOpCode.FMOD: np.fmod,
    BinaryOpCode.GCD: np.gcd,
    BinaryOpCode.GREATER: np.greater,
    BinaryOpCode.GREATER_EQUAL: np.greater_equal,
    BinaryOpCode.HYPOT: np.hypot,
    BinaryOpCode.LCM: np.lcm,
    BinaryOpCode.LDEXP: np.ldexp,
    BinaryOpCode.LEFT_SHIFT: np.left_shift,
    BinaryOpCode.LESS: np.less,
    BinaryOpCode.LESS_EQUAL: np.less_equal,
    BinaryOpCode.LOGADDEXP2: np.logaddexp2,
    BinaryOpCode.LOGADDEXP: np.logaddexp,
    BinaryOpCode.LOGICAL_AND: np.logical_and,
    BinaryOpCode.LOGICAL_OR: np.logical_or,
    BinaryOpCode.LOGICAL_XOR: np.logical_xor,
    BinaryOpCode.MAXIMUM: np.maximum,
    BinaryOpCode.MINIMUM: np.minimum,
    BinaryOpCode.MOD: np.mod,
    BinaryOpCode.MULTIPLY: np.multiply,
    BinaryOpCode.NEXTAFTER: np.nextafter,
    BinaryOpCode.NOT_EQUAL: np.not_equal,
    BinaryOpCode.POWER: np.power,
    BinaryOpCode.RIGHT_SHIFT: np.right_shift,
    BinaryOpCode.SUBTRACT: np.subtract,
}

_WINDOW_OPS: dict[
    WindowOpCode,
    Callable[[float], npt.NDArray[Any]]
    | Callable[[float, float], npt.NDArray[Any]],
] = {
    WindowOpCode.BARLETT: np.bartlett,
    WindowOpCode.BLACKMAN: np.blackman,
    WindowOpCode.HAMMING: np.hamming,
    WindowOpCode.HANNING: np.hanning,
    WindowOpCode.KAISER: np.kaiser,
}


def eye_reference(
    shape: NdShape, dtype: np.dtype[Any], axes: tuple[int, ...]
) -> npt.NDArray[Any]:
    n = min(shape[ax] for ax in axes)
    res = np.zeros(shape, dtype=dtype)
    for i in range(n):
        sl = tuple(
            i if ax in axes else slice(None) for ax in range(len(shape))
        )
        res[sl] = 1
    return res


def diagonal_reference(a: npt.NDArray[Any], axes: NdShape) -> npt.NDArray[Any]:
    transpose_axes = tuple(ax for ax in range(a.ndim) if ax not in axes)
    axes = tuple(sorted(axes, reverse=False, key=lambda i: a.shape[i]))
    a = a.transpose(transpose_axes + axes)
    diff = a.ndim - len(axes)
    axes = tuple((diff + ax) for ax in range(0, len(axes)))
    eye = eye_reference(a.shape, a.dtype, axes)
    res = a * eye
    for ax in tuple(reversed(sorted(axes)))[:-1]:
        res = res.sum(axis=ax)
    return np.array(res)


def _make_eager_unary_ufunc(
    np_ufunc: np.ufunc, deferred_ufunc: unary_ufunc
) -> Callable[..., Any]:
    def method(
        self: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any:
        """
        Helper method to apply unary ufunc operations.
        """
        from .._array.array import ndarray

        # Check if out contains a deferred thunk and convert self if needed
        if isinstance(out, ndarray) and runtime.is_deferred_array(out._thunk):
            if self.deferred is None:
                self.to_deferred_array(read_only=False)

        if self.deferred is not None:
            return deferred_ufunc._call_full(
                self.deferred,
                out=out,
                where=where,
                casting=casting,
                order=order,
                dtype=dtype,
            )

        out_array = (
            out._thunk.__numpy_array__() if isinstance(out, ndarray) else out
        )
        return np_ufunc(
            self.array,
            out=out_array,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
        )

    return method


def _make_eager_binary_ufunc(
    np_ufunc: np.ufunc, deferred_ufunc: binary_ufunc
) -> Callable[..., Any]:
    def method(
        self: Any,
        rhs: Any,
        out: Any | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> Any:
        """
        Helper method to apply binary ufunc operations.
        """
        from .._array.array import ndarray

        # Check if rhs or out contain deferred thunks and convert self if needed
        # We check the ._thunk attribute directly without extracting to avoid issues
        # Only convert if the thunk is truly deferred (not an eager array with .deferred set)
        # Also avoid converting if arrays are 0-dimensional (ndim=0) as deferred doesn't support them
        if isinstance(rhs, ndarray):
            rhs_thunk = rhs._thunk
            # Only convert if both self and rhs have at least 1 dimension and non-zero size
            can_convert = (
                self.ndim > 0
                and self.array.size > 0
                and rhs.ndim > 0
                and rhs.size > 0
            )
            if runtime.is_deferred_array(rhs_thunk):
                if self.deferred is None and can_convert:
                    self.to_deferred_array(read_only=False)
            elif (
                runtime.is_eager_array(rhs_thunk)
                and rhs_thunk.deferred is not None
            ):
                if self.deferred is None and can_convert:
                    self.to_deferred_array(read_only=False)
        if isinstance(out, ndarray):
            out_thunk = out._thunk
            # Only convert if both self and out have at least 1 dimension and non-zero size
            can_convert = (
                self.ndim > 0
                and self.array.size > 0
                and out.ndim > 0
                and out.size > 0
            )
            if runtime.is_deferred_array(out_thunk):
                if self.deferred is None and can_convert:
                    self.to_deferred_array(read_only=False)
            elif (
                runtime.is_eager_array(out_thunk)
                and out_thunk.deferred is not None
            ):
                if self.deferred is None and can_convert:
                    self.to_deferred_array(read_only=False)

        if self.deferred is not None:
            return deferred_ufunc._call_full(
                self.deferred,
                rhs,
                out=out,
                where=where,
                casting=casting,
                order=order,
                dtype=dtype,
            )

        rhs_array = (
            rhs._thunk.__numpy_array__() if isinstance(rhs, ndarray) else rhs
        )

        out_array = (
            out._thunk.__numpy_array__() if isinstance(out, ndarray) else out
        )
        return np_ufunc(
            self.array,
            rhs_array,
            out=out_array,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
        )

    return method


class EagerArray(NumPyThunk):
    """This is an eager thunk for describing NumPy computations.
    It is backed by a standard NumPy array that stores the result
    of the computation locally.

    :meta private:
    """

    def __init__(
        self,
        val: npt.ArrayLike,
        parent: EagerArray | None = None,
        key: tuple[Any, ...] | None = None,
    ) -> None:
        array = np.asarray(val)
        super().__init__(array.dtype)
        self.array: npt.NDArray[Any] = array
        self.parent: EagerArray | None = parent
        self.children: list[EagerArray] = []
        self.key: tuple[Any, ...] | None = key
        #: if this ever becomes set (to a DeferredArray), we forward all
        #: operations to it
        self.deferred: DeferredArray | None = None
        self.escaped = False

    @property
    def shape(self) -> NdShape:
        return self.array.shape

    def __numpy_array__(self) -> npt.NDArray[Any]:
        if self.deferred is not None:
            return self.deferred.__numpy_array__()
        # Track when this escapes. If it escapes we have
        # to be more careful in how we do our attach
        self.record_escape()
        return self.array.__array__()

    def __dlpack__(
        self,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType:
        if self.deferred is None:
            return self.array.__dlpack__(
                stream=stream,
                max_version=max_version,
                dl_device=dl_device,
                copy=copy,
            )

        return self.deferred.__dlpack__(
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        if self.deferred is None:
            return self.array.__dlpack_device__()

        return self.deferred.__dlpack_device__()

    def record_escape(self) -> None:
        if self.parent is None:
            self.escaped = True
        else:
            self.parent.record_escape()

    def check_eager_args(self, *args: Any) -> None:
        if self.deferred is not None:
            return
        for arg in args:
            if runtime.is_eager_array(arg):
                if arg.deferred is not None:
                    self.to_deferred_array(read_only=False)
                    break
            elif runtime.is_deferred_array(arg):
                self.to_deferred_array(read_only=False)
                break
            elif arg is None or not isinstance(arg, NumPyThunk):
                pass
            else:
                raise RuntimeError("bad argument type")

    def _convert_subtree(self) -> None:
        assert self.deferred is None
        if self.parent is None:
            transfer = (
                TransferType.SHARE
                if self.escaped
                # We can donate the base array, since it hasn't escaped to the
                # user, and we won't be using it anymore.
                else TransferType.DONATE
            )
            deferred = runtime.find_or_create_array_thunk(
                self.array, transfer=transfer, defer=True
            )
        else:
            parent = self.parent.deferred
            assert self.key is not None
            func = getattr(parent, self.key[0])
            args = self.key[1:]
            deferred = func(*args)
        self.deferred = cast(DeferredArray, deferred)
        for child in self.children:
            child._convert_subtree()

    def _convert_tree(self) -> None:
        """
        Convert the entire array tree to deferred arrays.

        We have to convert the whole tree when we convert even one node, to
        make sure any future use of any array in the tree will go through the
        deferred path, rather than use the original eager NumPy array, that we
        donated.
        """
        if self.parent is None:
            self._convert_subtree()
        else:
            self.parent._convert_tree()

    def to_deferred_array(self, read_only: bool) -> DeferredArray:
        """
        Convert this EagerArray into a DeferredArray.

        If `read_only` is `False`, the EagerArray's buffer is donated to
        initialize the DeferredArray, and the returned DeferredArray is used
        in place of the EagerArray going forward.
        """
        if self.deferred is not None:
            return self.deferred
        if read_only:
            deferred = cast(
                DeferredArray,
                runtime.find_or_create_array_thunk(
                    self.array,
                    transfer=TransferType.MAKE_COPY,
                    read_only=True,
                    defer=True,
                ),
            )
        else:
            self._convert_tree()
            deferred = cast(DeferredArray, self.deferred)
        return deferred

    def imag(self) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.imag()
        return EagerArray(self.array.imag)

    def real(self) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.real()
        return EagerArray(self.array.real)

    def conj(self) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.conj()

        return EagerArray(self.array.conj())

    def convolve(
        self,
        input: Any,
        filter: Any,
        mode: ConvolveMode,
        method: ConvolveMethod,
    ) -> None:
        self.check_eager_args(input, filter)
        if self.deferred is not None:
            self.deferred.convolve(input, filter, mode, method)
        else:
            if self.ndim == 1:
                if method != "auto":
                    runtime.warn(
                        f"the method {method} is ignored for the 1D convolution"
                    )
                self.array[:] = np.convolve(input.array, filter.array, mode)
            else:
                from scipy.signal import convolve  # type: ignore [import-untyped]

                self.array[...] = convolve(
                    input.array, filter.array, mode, method
                )

    def fft(
        self,
        rhs: Any,
        axes: Sequence[int],
        kind: FFTType,
        direction: FFTDirection,
    ) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.fft(rhs, axes, kind, direction)
        else:
            res: npt.NDArray[Any]
            if kind in (FFT_D2Z, FFT_R2C):
                res = np.fft.rfftn(rhs.array, axes=axes, norm="backward")
            elif kind in (FFT_Z2D, FFT_C2R):
                s = tuple(rhs.array.shape[i] for i in axes)
                res = np.fft.irfftn(rhs.array, s=s, axes=axes, norm="forward")
            else:
                if direction == FFTDirection.FORWARD:
                    res = np.fft.fftn(rhs.array, axes=axes, norm="backward")
                else:
                    res = np.fft.ifftn(rhs.array, axes=axes, norm="forward")
            if kind.is_single_precision:
                if res.dtype == np.complex128:
                    self.array[:] = res.astype(np.complex64)
                elif res.dtype == np.float64:
                    self.array[:] = res.astype(np.float32)
                else:
                    if not is_np2:
                        raise RuntimeError(
                            f"Unsupported data type {res.dtype!r} in eager FFT"
                        )
                    else:
                        self.array[:] = res

            else:
                self.array[:] = res

    def copy(self, rhs: Any, deep: bool = False) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.copy(rhs, deep=deep)
        else:
            if self.array.size == 1:
                self.array.fill(rhs.array.item())
            elif deep:
                self.array[:] = rhs.array.__deepcopy__(None)
            else:
                self.array[:] = rhs.array

    @property
    def scalar(self) -> bool:
        if self.deferred is not None:
            return self.deferred.scalar
        return self.array.size == 1

    def _create_indexing_key(self, key: IndexKey) -> Any:
        if key is None or key is Ellipsis:
            return key
        if isinstance(key, int):
            return key
        if isinstance(key, slice):
            return key
        if isinstance(key, tuple):
            result: tuple[Any, ...] = ()
            for k in key:
                result += (self._create_indexing_key(k),)
            return result
        assert isinstance(key, NumPyThunk)
        return runtime.to_eager_array(key).array

    def get_item(self, key: IndexKey) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.get_item(key)
        if is_advanced_indexing(key):
            index_key = self._create_indexing_key(key)
            out = self.array[index_key]
            result = EagerArray(out)
        else:
            child = self.array[key]  # type: ignore[index]
            result = EagerArray(child, parent=self, key=("get_item", key))
            self.children.append(result)
        return result

    def set_item(self, key: IndexKey, value: Any) -> None:
        self.check_eager_args(value)
        if self.deferred is not None:
            self.deferred.set_item(key, value)
        else:
            if is_advanced_indexing(key):
                index_key = self._create_indexing_key(key)
                if isinstance(value, EagerArray):
                    self.array[index_key] = value.array
                else:
                    self.array[index_key] = value
            else:
                if isinstance(value, EagerArray):
                    self.array[key] = value.array  # type: ignore[index]
                else:
                    self.array[key] = value  # type: ignore[index]

    def reshape(self, newshape: NdShape, order: OrderType) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.reshape(newshape, order)
        child = self.array.reshape(newshape, order=order)
        # See if we are aliased or not
        if child.base is None:
            result = EagerArray(child)
        else:
            result = EagerArray(
                child, parent=self, key=("reshape", newshape, order)
            )
            self.children.append(result)
        return result

    def squeeze(self, axis: int | tuple[int, ...] | None) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.squeeze(axis)
        # See https://github.com/numpy/numpy/issues/22019
        child = self.array.squeeze(cast(Any, axis))
        # Early exit if there's no dimension to squeeze
        if child is self.array:
            return self
        # Should be aliased with parent region
        assert child.base is not None
        result = EagerArray(child, parent=self, key=("squeeze", axis))
        self.children.append(result)
        return result

    def swapaxes(self, axis1: int, axis2: int) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.swapaxes(axis1, axis2)
        child = self.array.swapaxes(axis1, axis2)
        # Should be aliased with parent region
        assert child.base is not None
        result = EagerArray(child, parent=self, key=("swapaxes", axis1, axis2))
        self.children.append(result)
        return result

    def convert(
        self,
        rhs: Any,
        warn: bool = True,
        nan_op: ConvertCode = ConvertCode.NOOP,
        temporary: bool = False,
    ) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            return self.deferred.convert(rhs, warn=warn)
        else:
            if self.array.size == 1:
                if nan_op is ConvertCode.SUM and np.isnan(rhs.array.item()):
                    self.array.fill(0)
                elif nan_op is ConvertCode.PROD and np.isnan(rhs.array.item()):
                    self.array.fill(1)
                else:
                    self.array.fill(rhs.array.astype(self.array.dtype).item())
            else:
                if nan_op is ConvertCode.SUM:
                    self.array[:] = np.where(np.isnan(rhs.array), 0, rhs.array)
                elif nan_op is ConvertCode.PROD:
                    self.array[:] = np.where(np.isnan(rhs.array), 1, rhs.array)
                else:
                    self.array[:] = rhs.array

    def fill(self, value: Any) -> None:
        if self.deferred is not None:
            self.deferred.fill(value)
        else:
            self.array.fill(value)

    def transpose(self, axes: tuple[int, ...] | list[int]) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.transpose(axes)
        # See https://github.com/numpy/numpy/issues/22019
        child = self.array.transpose(cast(Any, axes))
        # Should be aliased with parent region
        assert child.base is not None
        result = EagerArray(child, parent=self, key=("transpose", axes))
        self.children.append(result)
        return result

    def repeat(
        self, repeats: Any, axis: int, scalar_repeats: bool
    ) -> NumPyThunk:
        if not scalar_repeats:
            self.check_eager_args(repeats)
        if self.deferred is not None:
            return self.deferred.repeat(repeats, axis, scalar_repeats)
        else:
            if not scalar_repeats:
                array = np.repeat(self.array, repeats.array, axis)
            else:
                array = np.repeat(self.array, repeats, axis)
            return EagerArray(array)

    def pad(
        self,
        pad_width: tuple[tuple[int, int], ...],
        mode: str,
        constant_value_thunk: Any = None,
        constant_rows: int = 0,
        constant_cols: int = 0,
    ) -> None:
        if self.deferred is not None:
            self.deferred.pad(
                pad_width,
                mode,
                constant_value_thunk,
                constant_rows,
                constant_cols,
            )
        else:
            # Use numpy for eager execution
            kwargs: dict[str, Any] = {}
            if mode == "constant" and constant_value_thunk is not None:
                const_array = constant_value_thunk.__numpy_array__()
                if const_array.ndim == 0:
                    kwargs["constant_values"] = const_array.item()
                elif constant_rows > 0 and constant_cols > 0:
                    reshaped = const_array.reshape(
                        (constant_rows, constant_cols)
                    )
                    kwargs["constant_values"] = reshaped.tolist()
                else:
                    kwargs["constant_values"] = cast(Any, const_array)

            center_slice = tuple(
                slice(
                    pad_width[dim][0],
                    self.array.shape[dim] - pad_width[dim][1],
                )
                for dim in range(self.array.ndim)
            )
            core = self.array[center_slice]
            padded = np.pad(core, pad_width, mode=cast(Any, mode), **kwargs)
            np.copyto(self.array, padded)

    def flip(self, rhs: Any, axes: int | tuple[int, ...] | None) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.flip(rhs, axes)
        else:
            self.array = np.flip(rhs.array, axes)

    def broadcast_to(self, shape: NdShape) -> NumPyThunk:
        # When Eager and Deferred broadcasted arrays are used for computation,
        # eager arrays are converted by 'to_deferred()'
        # this method uses array.base to create a deferred array,
        # which is different from the shape of the broadcasted arrays
        if self.deferred is not None:
            return self.deferred.broadcast_to(shape)
        child = np.broadcast_to(self.array, shape)
        # Should be aliased with parent region
        assert child.base is not None
        result = EagerArray(child, parent=self, key=("broadcast_to", shape))
        self.children.append(result)
        return result

    def take(
        self,
        indices: Any,
        axis: int | None = None,
        out: Any | None = None,
        mode: BoundsMode = "raise",
    ) -> Any:
        is_scalar = np.isscalar(indices)
        if not is_scalar:
            self.check_eager_args(indices)
        if self.deferred is not None:
            return self.deferred.take(indices, axis, out=out, mode=mode)
        out_array = None
        if out is not None:
            out_array = out.__numpy_array__()
        result = self.array.take(
            indices if is_scalar else indices.array,
            axis,
            out=out_array,
            mode=mode,
        )
        if out is not None:
            assert result is out_array
            return out
        return EagerArray(result)

    def take_along_axis(
        self,
        indices: Any,
        axis: int,
        out: Any | None = None,
        mode: BoundsMode = "raise",
    ) -> Any:
        self.check_eager_args(indices)
        if self.deferred is not None:
            return self.deferred.take_along_axis(
                indices, axis, out=out, mode=mode
            )
        out_array = None
        if out is not None:
            out_array = out.__numpy_array__()
        result = np.take_along_axis(self.array, indices.array, axis)
        if out is not None:
            assert out_array is not None
            out_array[:] = result
            return out
        return EagerArray(result)

    def contract(
        self,
        lhs_modes: list[str],
        rhs1_thunk: Any,
        rhs1_modes: list[str],
        rhs2_thunk: Any,
        rhs2_modes: list[str],
        mode2extent: dict[str, int],
    ) -> None:
        self.check_eager_args(rhs1_thunk, rhs2_thunk)
        if self.deferred is not None:
            self.deferred.contract(
                lhs_modes,
                rhs1_thunk,
                rhs1_modes,
                rhs2_thunk,
                rhs2_modes,
                mode2extent,
            )
        else:
            np.einsum(
                f"{''.join(rhs1_modes)},{''.join(rhs2_modes)}->{''.join(lhs_modes)}",
                rhs1_thunk.array,
                rhs2_thunk.array,
                out=self.array,
            )

    def choose(self, rhs: Any, *args: Any) -> None:
        self.check_eager_args(*args, rhs)
        if self.deferred is not None:
            self.deferred.choose(rhs, *args)
        else:
            choices = tuple(c.array for c in args)
            self.array[:] = np.choose(rhs.array, choices, mode="raise")

    def select(
        self,
        condlist: Iterable[Any],
        choicelist: Iterable[Any],
        default: npt.NDArray[Any],
    ) -> None:
        self.check_eager_args(*condlist, *choicelist)
        if self.deferred is not None:
            self.deferred.select(condlist, choicelist, default)
        else:
            self.array[...] = np.select(
                tuple(c.array for c in condlist),
                tuple(c.array for c in choicelist),
                default,
            )

    def _diag_helper(
        self, rhs: Any, offset: int, naxes: int, extract: bool, trace: bool
    ) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred._diag_helper(rhs, offset, naxes, extract, trace)
        else:
            if (naxes == 2) and extract and not trace:
                ndims = rhs.array.ndim
                self.array[:] = np.diagonal(
                    rhs.array, offset=offset, axis1=ndims - 2, axis2=ndims - 1
                )
            elif (naxes < 2) and not extract:
                self.array[:] = np.diag(rhs.array, offset)
            elif (naxes >= 2) and trace:
                ndim = rhs.array.ndim
                self.array[:] = np.trace(
                    rhs.array, offset=offset, axis1=ndim - 2, axis2=ndim - 1
                )
            else:  # naxes>2
                ndims = rhs.array.ndim
                axes = tuple(range(ndims - naxes, ndims))
                self.array = diagonal_reference(rhs.array, axes)

    def put(self, indices: Any, values: Any, check_bounds: bool) -> None:
        self.check_eager_args(indices, values)
        if self.deferred is not None:
            self.deferred.put(indices, values, check_bounds)
        else:
            np.put(self.array, indices.array, values.array)

    def putmask(self, mask: Any, values: Any) -> None:
        self.check_eager_args(mask, values)
        if self.deferred is not None:
            self.deferred.putmask(mask, values)
        else:
            np.putmask(self.array, mask.array, values.array)

    def eye(self, k: int) -> None:
        if self.deferred is not None:
            self.deferred.eye(k)
        else:
            if self.array.size == 1:
                self.array.fill(1)
            else:
                self.array[:] = np.eye(
                    self.shape[0], self.shape[1], k, dtype=self.dtype
                )

    def arange(self, start: float, stop: float, step: float) -> None:
        if self.deferred is not None:
            self.deferred.arange(start, stop, step)
        else:
            self.array = np.arange(start, stop, step, self.dtype)

    def tile(self, rhs: Any, reps: int | Sequence[int]) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.tile(rhs, reps)
        else:
            self.array[:] = np.tile(rhs.array, reps)

    def bincount(self, rhs: Any, weights: NumPyThunk | None = None) -> None:
        self.check_eager_args(rhs, weights)
        if self.deferred is not None:
            self.deferred.bincount(rhs, weights=weights)
        else:
            self.array[:] = np.bincount(
                rhs.array,
                cast(EagerArray, weights).array if weights else None,
                minlength=self.array.size,
            )

    def nonzero(self) -> tuple[NumPyThunk, ...]:
        if self.deferred is not None:
            return self.deferred.nonzero()
        else:
            arrays = self.array.nonzero()
            result: tuple[NumPyThunk, ...] = ()
            for array in arrays:
                result += (EagerArray(array),)
            return result

    def searchsorted(self, rhs: Any, v: Any, side: SortSide = "left") -> None:
        self.check_eager_args(rhs, v)
        if self.deferred is not None:
            self.deferred.searchsorted(rhs, v, side)
        else:
            self.array = np.searchsorted(rhs.array, v.array, side=side)

    def sort(
        self,
        rhs: Any,
        argsort: bool = False,
        axis: int | None = -1,
        kind: SortType = "quicksort",
        order: str | list[str] | None = None,
    ) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.sort(rhs, argsort, axis, kind, order)
        else:
            if argsort:
                self.array = np.argsort(rhs.array, axis, kind, order)
            else:
                self.array = np.sort(rhs.array, axis, kind, order)

    def bitgenerator_random_raw(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_random_raw(
                handle, generatorType, seed, flags
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randint(0, 2**32 - 1))
            else:
                a = np.random.randint(
                    low=0,
                    high=2**32 - 1,
                    size=self.array.shape,
                    dtype=self.array.dtype,
                )
                self.array[:] = a[:]

    def bitgenerator_integers(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        low: int,
        high: int,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_integers(
                handle, generatorType, seed, flags, low, high
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randint(low, high))
            else:
                a = np.random.randint(low, high, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_lognormal(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mean: float,
        sigma: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_lognormal(
                handle, generatorType, seed, flags, mean, sigma
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.lognormal(mean, sigma))
            else:
                a = np.random.lognormal(mean, sigma, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_normal(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mean: float,
        sigma: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_normal(
                handle, generatorType, seed, flags, mean, sigma
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.normal(mean, sigma))
            else:
                a = np.random.normal(mean, sigma, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_uniform(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        low: float,
        high: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_uniform(
                handle, generatorType, seed, flags, low, high
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.uniform(low, high))
            else:
                a = np.random.uniform(low, high, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_poisson(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        lam: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_poisson(
                handle, generatorType, seed, flags, lam
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.poisson(lam))
            else:
                a = np.random.poisson(lam, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_exponential(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        scale: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_exponential(
                handle, generatorType, seed, flags, scale
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.exponential(scale))
            else:
                a = np.random.exponential(scale, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_gumbel(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_gumbel(
                handle, generatorType, seed, flags, mu, beta
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.gumbel(mu, beta))
            else:
                a = np.random.gumbel(mu, beta, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_laplace(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_laplace(
                handle, generatorType, seed, flags, mu, beta
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.laplace(mu, beta))
            else:
                a = np.random.laplace(mu, beta, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_logistic(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mu: float,
        beta: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_logistic(
                handle, generatorType, seed, flags, mu, beta
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.logistic(mu, beta))
            else:
                a = np.random.logistic(mu, beta, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_pareto(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        alpha: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_pareto(
                handle, generatorType, seed, flags, alpha
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.pareto(alpha))
            else:
                a = np.random.pareto(alpha, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_power(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        alpha: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_power(
                handle, generatorType, seed, flags, alpha
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.power(alpha))
            else:
                a = np.random.power(alpha, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_rayleigh(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        sigma: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_rayleigh(
                handle, generatorType, seed, flags, sigma
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.rayleigh(sigma))
            else:
                a = np.random.rayleigh(sigma, size=self.array.shape)
                self.array[:] = a

    def bitgenerator_cauchy(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        x0: float,
        gamma: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_cauchy(
                handle, generatorType, seed, flags, x0, gamma
            )
        else:
            if self.array.size == 1:
                self.array.fill(x0 + gamma * np.random.standard_cauchy())
            else:
                a = np.random.standard_cauchy(size=self.array.shape)
                self.array[:] = x0 + gamma * a

    def bitgenerator_triangular(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        a: float,
        b: float,
        c: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_triangular(
                handle, generatorType, seed, flags, a, b, c
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.triangular(a, c, b))
            else:
                aa = np.random.triangular(a, c, b, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_weibull(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        lam: float,
        k: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_weibull(
                handle, generatorType, seed, flags, lam, k
            )
        else:
            if self.array.size == 1:
                self.array.fill(lam * np.random.weibull(k))
            else:
                aa = np.random.weibull(k, size=self.array.shape)
                self.array[:] = lam * aa

    def bitgenerator_bytes(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_bytes(
                handle, generatorType, seed, flags
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.bytes(1))
            else:
                aa = np.random.bytes(self.array.size)
                b = bytearray()
                b.extend(aa)
                self.array[:] = b

    def bitgenerator_beta(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        a: float,
        b: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_beta(
                handle, generatorType, seed, flags, a, b
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.beta(a, b))
            else:
                aa = np.random.beta(a, b, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_f(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        dfnum: float,
        dfden: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_f(
                handle, generatorType, seed, flags, dfnum, dfden
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.f(dfnum, dfden))
            else:
                aa = np.random.f(dfnum, dfden, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_logseries(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        p: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_logseries(
                handle, generatorType, seed, flags, p
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.logseries(p))
            else:
                aa = np.random.logseries(p, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_noncentral_f(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        dfnum: float,
        dfden: float,
        nonc: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_noncentral_f(
                handle, generatorType, seed, flags, dfnum, dfden, nonc
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.noncentral_f(dfnum, dfden, nonc))
            else:
                aa = np.random.noncentral_f(
                    dfnum, dfden, nonc, size=self.array.shape
                )
                self.array[:] = aa

    def bitgenerator_chisquare(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        df: float,
        nonc: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_chisquare(
                handle, generatorType, seed, flags, df, nonc
            )
        else:
            if self.array.size == 1:
                if nonc == 0.0:
                    self.array.fill(np.random.chisquare(df))
                else:
                    self.array.fill(np.random.noncentral_chisquare(df, nonc))
            else:
                if nonc == 0.0:
                    aa = np.random.chisquare(df, size=self.array.shape)
                else:
                    aa = np.random.noncentral_chisquare(
                        df, nonc, size=self.array.shape
                    )
                self.array[:] = aa

    def bitgenerator_gamma(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        k: float,
        theta: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_gamma(
                handle, generatorType, seed, flags, k, theta
            )
        else:
            if self.array.size == 1:
                if theta == 1.0:
                    self.array.fill(np.random.standard_gamma(k))
                else:
                    self.array.fill(np.random.gamma(k, theta))
            else:
                if theta == 1.0:
                    aa = np.random.standard_gamma(k, size=self.array.shape)
                else:
                    aa = np.random.gamma(k, theta, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_standard_t(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        df: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_standard_t(
                handle, generatorType, seed, flags, df
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.standard_t(df))
            else:
                aa = np.random.standard_t(df, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_hypergeometric(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        ngood: int,
        nbad: int,
        nsample: int,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_hypergeometric(
                handle, generatorType, seed, flags, ngood, nbad, nsample
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.hypergeometric(ngood, nbad, nsample))
            else:
                aa = np.random.hypergeometric(
                    ngood, nbad, nsample, size=self.array.shape
                )
                self.array[:] = aa

    def bitgenerator_vonmises(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mu: float,
        kappa: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_vonmises(
                handle, generatorType, seed, flags, mu, kappa
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.vonmises(mu, kappa))
            else:
                aa = np.random.vonmises(mu, kappa, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_zipf(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        alpha: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_zipf(
                handle, generatorType, seed, flags, alpha
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.zipf(alpha))
            else:
                aa = np.random.zipf(alpha, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_geometric(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        p: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_geometric(
                handle, generatorType, seed, flags, p
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.geometric(p))
            else:
                aa = np.random.geometric(p, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_wald(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        mean: float,
        scale: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_wald(
                handle, generatorType, seed, flags, mean, scale
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.wald(mean, scale))
            else:
                aa = np.random.wald(mean, scale, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_binomial(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        ntrials: int,
        p: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_binomial(
                handle, generatorType, seed, flags, ntrials, p
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.binomial(ntrials, p))
            else:
                aa = np.random.binomial(ntrials, p, size=self.array.shape)
                self.array[:] = aa

    def bitgenerator_negative_binomial(
        self,
        handle: int,
        generatorType: BitGeneratorType,
        seed: int | None,
        flags: int,
        ntrials: int,
        p: float,
    ) -> None:
        if self.deferred is not None:
            self.deferred.bitgenerator_negative_binomial(
                handle, generatorType, seed, flags, ntrials, p
            )
        else:
            if self.array.size == 1:
                self.array.fill(np.random.negative_binomial(ntrials, p))
            else:
                aa = np.random.negative_binomial(
                    ntrials, p, size=self.array.shape
                )
                self.array[:] = aa

    def partition(
        self,
        rhs: Any,
        kth: int | Sequence[int],
        argpartition: bool = False,
        axis: int | None = -1,
        kind: SelectKind = "introselect",
        order: str | list[str] | None = None,
    ) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.partition(rhs, kth, argpartition, axis, kind, order)
        else:
            if argpartition:
                self.array = np.argpartition(rhs.array, kth, axis, kind, order)
            else:
                self.array = np.partition(rhs.array, kth, axis, kind, order)

    def random_uniform(self) -> None:
        if self.deferred is not None:
            self.deferred.random_uniform()
        else:
            if self.array.size == 1:
                self.array.fill(np.random.rand())
            else:
                self.array[:] = np.random.rand(*(self.array.shape))

    def random_normal(self) -> None:
        if self.deferred is not None:
            self.deferred.random_normal()
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randn())
            else:
                self.array[:] = np.random.randn(*(self.array.shape))

    def random_integer(
        self, low: int | npt.NDArray[Any], high: int | npt.NDArray[Any]
    ) -> None:
        if self.deferred is not None:
            self.deferred.random_integer(low, high)
        else:
            if self.array.size == 1:
                self.array.fill(np.random.randint(low, high))
            else:
                self.array[:] = np.random.randint(
                    low, high, size=self.array.shape, dtype=self.array.dtype
                )

    # binary
    def _matmul(
        self,
        rhs: Any,
        out: Any | None = None,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Helper method to apply binary ufunc operations.
        """
        if self.deferred is not None:
            from .._array.array import ndarray
            from .._module.linalg_mvp import matmul

            if kwargs:
                keys = ", ".join(str(k) for k in kwargs.keys())
                raise NotImplementedError(
                    f"matmul doesn't support kwargs: {keys}"
                )

            a = ndarray._from_thunk(self.deferred)
            return matmul(a, rhs, out=out, casting=casting, dtype=dtype)
        else:
            rhs_array = (
                rhs._thunk.__numpy_array__()
                if (rhs is not None and hasattr(rhs, "_thunk"))
                else rhs
            )

            out_array = (
                out._thunk.__numpy_array__()
                if (out is not None and hasattr(out, "_thunk"))
                else out
            )
            return np.matmul(self.array, rhs_array, out=out_array, **kwargs)

    _add = _make_eager_binary_ufunc(np.add, _ufunc.add)
    _multiply = _make_eager_binary_ufunc(np.multiply, _ufunc.multiply)
    _subtract = _make_eager_binary_ufunc(np.subtract, _ufunc.subtract)
    _true_divide = _make_eager_binary_ufunc(np.true_divide, _ufunc.true_divide)
    _floor_divide = _make_eager_binary_ufunc(
        np.floor_divide, _ufunc.floor_divide
    )
    _logaddexp = _make_eager_binary_ufunc(np.logaddexp, _ufunc.logaddexp)
    _logaddexp2 = _make_eager_binary_ufunc(np.logaddexp2, _ufunc.logaddexp2)
    _power = _make_eager_binary_ufunc(np.power, _ufunc.power)
    _float_power = _make_eager_binary_ufunc(np.float_power, _ufunc.float_power)
    _remainder = _make_eager_binary_ufunc(np.remainder, _ufunc.remainder)
    _gcd = _make_eager_binary_ufunc(np.gcd, _ufunc.gcd)
    _lcm = _make_eager_binary_ufunc(np.lcm, _ufunc.lcm)

    # unary
    _negative = _make_eager_unary_ufunc(np.negative, _ufunc.negative)
    _positive = _make_eager_unary_ufunc(np.positive, _ufunc.positive)
    _absolute = _make_eager_unary_ufunc(np.absolute, _ufunc.absolute)
    _rint = _make_eager_unary_ufunc(np.rint, _ufunc.rint)
    _sign = _make_eager_unary_ufunc(np.sign, _ufunc.sign)
    _conjugate = _make_eager_unary_ufunc(np.conjugate, _ufunc.conjugate)
    _exp = _make_eager_unary_ufunc(np.exp, _ufunc.exp)
    _exp2 = _make_eager_unary_ufunc(np.exp2, _ufunc.exp2)
    _log = _make_eager_unary_ufunc(np.log, _ufunc.log)
    _log2 = _make_eager_unary_ufunc(np.log2, _ufunc.log2)
    _log10 = _make_eager_unary_ufunc(np.log10, _ufunc.log10)
    _expm1 = _make_eager_unary_ufunc(np.expm1, _ufunc.expm1)
    _log1p = _make_eager_unary_ufunc(np.log1p, _ufunc.log1p)
    _square = _make_eager_unary_ufunc(np.square, _ufunc.square)
    _sqrt = _make_eager_unary_ufunc(np.sqrt, _ufunc.sqrt)
    _cbrt = _make_eager_unary_ufunc(np.cbrt, _ufunc.cbrt)
    _reciprocal = _make_eager_unary_ufunc(np.reciprocal, _ufunc.reciprocal)

    # logical
    _greater_equal = _make_eager_binary_ufunc(
        np.greater_equal, _ufunc.greater_equal
    )
    _equal = _make_eager_binary_ufunc(np.equal, _ufunc.equal)
    _greater = _make_eager_binary_ufunc(np.greater, _ufunc.greater)
    _less = _make_eager_binary_ufunc(np.less, _ufunc.less)
    _less_equal = _make_eager_binary_ufunc(np.less_equal, _ufunc.less_equal)
    _not_equal = _make_eager_binary_ufunc(np.not_equal, _ufunc.not_equal)
    _logical_and = _make_eager_binary_ufunc(np.logical_and, _ufunc.logical_and)
    _logical_or = _make_eager_binary_ufunc(np.logical_or, _ufunc.logical_or)
    _logical_xor = _make_eager_binary_ufunc(np.logical_xor, _ufunc.logical_xor)
    _logical_not = _make_eager_unary_ufunc(np.logical_not, _ufunc.logical_not)
    _maximum = _make_eager_binary_ufunc(np.maximum, _ufunc.maximum)
    _minimum = _make_eager_binary_ufunc(np.minimum, _ufunc.minimum)

    # bit_twiddling
    _bitwise_and = _make_eager_binary_ufunc(np.bitwise_and, _ufunc.bitwise_and)
    _bitwise_or = _make_eager_binary_ufunc(np.bitwise_or, _ufunc.bitwise_or)
    _bitwise_xor = _make_eager_binary_ufunc(np.bitwise_xor, _ufunc.bitwise_xor)
    _invert = _make_eager_unary_ufunc(np.invert, _ufunc.invert)
    _left_shift = _make_eager_binary_ufunc(np.left_shift, _ufunc.left_shift)
    _right_shift = _make_eager_binary_ufunc(np.right_shift, _ufunc.right_shift)

    # floating:
    _isfinite = _make_eager_unary_ufunc(np.isfinite, _ufunc.isfinite)
    _isinf = _make_eager_unary_ufunc(np.isinf, _ufunc.isinf)
    _isnan = _make_eager_unary_ufunc(np.isnan, _ufunc.isnan)
    _fabs = _make_eager_unary_ufunc(np.fabs, _ufunc.fabs)
    _signbit = _make_eager_unary_ufunc(np.signbit, _ufunc.signbit)
    _copysign = _make_eager_binary_ufunc(np.copysign, _ufunc.copysign)
    _nextafter = _make_eager_binary_ufunc(np.nextafter, _ufunc.nextafter)
    _ldexp = _make_eager_binary_ufunc(np.ldexp, _ufunc.ldexp)
    _fmod = _make_eager_binary_ufunc(np.fmod, _ufunc.fmod)
    _floor = _make_eager_unary_ufunc(np.floor, _ufunc.floor)
    _ceil = _make_eager_unary_ufunc(np.ceil, _ufunc.ceil)
    _trunc = _make_eager_unary_ufunc(np.trunc, _ufunc.trunc)

    # trigonometric:
    _sin = _make_eager_unary_ufunc(np.sin, _ufunc.sin)
    _cos = _make_eager_unary_ufunc(np.cos, _ufunc.cos)
    _tan = _make_eager_unary_ufunc(np.tan, _ufunc.tan)
    _arcsin = _make_eager_unary_ufunc(np.arcsin, _ufunc.arcsin)
    _arccos = _make_eager_unary_ufunc(np.arccos, _ufunc.arccos)
    _arctan = _make_eager_unary_ufunc(np.arctan, _ufunc.arctan)
    _arctan2 = _make_eager_binary_ufunc(np.arctan2, _ufunc.arctan2)
    _hypot = _make_eager_binary_ufunc(np.hypot, _ufunc.hypot)
    _sinh = _make_eager_unary_ufunc(np.sinh, _ufunc.sinh)
    _cosh = _make_eager_unary_ufunc(np.cosh, _ufunc.cosh)
    _tanh = _make_eager_unary_ufunc(np.tanh, _ufunc.tanh)
    _arcsinh = _make_eager_unary_ufunc(np.arcsinh, _ufunc.arcsinh)
    _arccosh = _make_eager_unary_ufunc(np.arccosh, _ufunc.arccosh)
    _arctanh = _make_eager_unary_ufunc(np.arctanh, _ufunc.arctanh)
    _deg2rad = _make_eager_unary_ufunc(np.deg2rad, _ufunc.deg2rad)
    _rad2deg = _make_eager_unary_ufunc(np.rad2deg, _ufunc.rad2deg)

    def unary_op(
        self,
        op: UnaryOpCode,
        rhs: Any,
        where: Any,
        args: tuple[Scalar, ...] = (),
        multiout: Any | None = None,
    ) -> None:
        if multiout is None:
            self.check_eager_args(rhs, where)
        else:
            self.check_eager_args(rhs, where, *multiout)

        if self.deferred is not None:
            self.deferred.unary_op(op, rhs, where, args, multiout=multiout)
            return

        if op in _UNARY_OPS:
            func = _UNARY_OPS[op]
            if multiout is None:
                func(
                    rhs.array,
                    out=self.array,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
            else:
                func(
                    rhs.array,
                    out=(self.array, *(out.array for out in multiout)),
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                )
        elif op == UnaryOpCode.CLIP:
            np.clip(
                rhs.array,
                out=self.array,
                a_min=args[0].value(),
                a_max=args[1].value(),
            )
        elif op == UnaryOpCode.COPY:
            self.array[:] = rhs.array[:]
        elif op == UnaryOpCode.IMAG:
            self.array = np.imag(rhs.array)
        elif op == UnaryOpCode.REAL:
            self.array = np.real(rhs.array)
        elif op == UnaryOpCode.ROUND:
            np.round(rhs.array, out=self.array, decimals=args[0].value())
        elif op == UnaryOpCode.ANGLE:
            self.array = np.angle(rhs.array, args[0].value())
        else:
            raise RuntimeError("unsupported unary op " + str(op))

    def unary_reduction(
        self,
        op: UnaryRedCode,
        rhs: Any,
        where: Any,
        orig_axis: int | None,
        axes: tuple[int, ...],
        keepdims: bool,
        args: tuple[Scalar, ...],
        initial: Any,
    ) -> None:
        self.check_eager_args(rhs, where)
        if self.deferred is not None:
            self.deferred.unary_reduction(
                op, rhs, where, orig_axis, axes, keepdims, args, initial
            )
            return
        if where is None:
            where = True

        match op:
            case op if op in _UNARY_RED_OPS_WITH_ARG:
                fn = _UNARY_RED_OPS_WITH_ARG[op]
                # arg based APIs don't have the following arguments: where, initial
                if op in _UNARY_RED_OPS_WITH_ARG:
                    fn(
                        rhs.array,
                        out=self.array,
                        axis=orig_axis,
                        keepdims=keepdims,
                    )
            case op if op in _UNARY_RED_OPS_WITHOUT_ARG:
                fn = _UNARY_RED_OPS_WITHOUT_ARG[op]
                # Need to be more careful here, Numpy does not use None to mean
                # "was not passed in" in this instance
                kws = {"initial": initial} if initial is not None else {}
                fn(
                    rhs.array,
                    out=self.array,
                    axis=orig_axis,
                    keepdims=keepdims,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                    **kws,
                )
            case UnaryRedCode.SUM_SQUARES:
                squared = np.square(rhs.array)
                np.sum(
                    squared,
                    out=self.array,
                    axis=orig_axis,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                    keepdims=keepdims,
                )
            case UnaryRedCode.VARIANCE:
                (mu,) = args
                centered = np.subtract(rhs.array, np.asarray(mu))
                squares = np.square(centered)
                np.sum(
                    squares,
                    axis=orig_axis,
                    where=where
                    if not isinstance(where, EagerArray)
                    else where.array,
                    keepdims=keepdims,
                    out=self.array,
                )
            case UnaryRedCode.CONTAINS:
                self.array.fill(args[0].value() in rhs.array)
            case UnaryRedCode.COUNT_NONZERO:
                self.array[()] = np.count_nonzero(
                    rhs.array, axis=orig_axis, keepdims=keepdims
                )
            case _:
                raise RuntimeError("unsupported unary reduction op " + str(op))

    def isclose(
        self, rhs1: Any, rhs2: Any, rtol: float, atol: float, equal_nan: bool
    ) -> None:
        self.check_eager_args(rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.isclose(rhs1, rhs2, rtol, atol, equal_nan)
        else:
            self.array[...] = np.isclose(
                rhs1.array,
                rhs2.array,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )

    def binary_op(
        self,
        op: BinaryOpCode,
        rhs1: Any,
        rhs2: Any,
        where: Any,
        args: tuple[Scalar, ...],
    ) -> None:
        self.check_eager_args(rhs1, rhs2, where)
        if self.deferred is not None:
            self.deferred.binary_op(op, rhs1, rhs2, where, args)
        else:
            func = _BINARY_OPS.get(op, None)
            if func is None:
                raise RuntimeError("unsupported binary op " + str(op))
            func(
                rhs1.array,
                rhs2.array,
                out=self.array,
                where=where
                if not isinstance(where, EagerArray)
                else where.array,
            )

    def binary_reduction(
        self,
        op: BinaryOpCode,
        rhs1: Any,
        rhs2: Any,
        broadcast: NdShape | None,
        args: tuple[Scalar, ...],
    ) -> None:
        self.check_eager_args(rhs1, rhs2)
        if self.deferred is not None:
            self.deferred.binary_reduction(op, rhs1, rhs2, broadcast, args)
        else:
            if op == BinaryOpCode.ISCLOSE:
                self.array = np.array(
                    np.allclose(
                        rhs1.array,
                        rhs2.array,
                        rtol=args[0].value(),
                        atol=args[1].value(),
                    )
                )
            elif op == BinaryOpCode.EQUAL:
                self.array = np.array(np.array_equal(rhs1.array, rhs2.array))
            else:
                raise RuntimeError(
                    "unsupported binary reduction op " + str(op)
                )

    def where(self, rhs1: Any, rhs2: Any, rhs3: Any) -> None:
        self.check_eager_args(rhs1, rhs2, rhs3)
        if self.deferred is not None:
            self.deferred.where(rhs1, rhs2, rhs3)
        else:
            self.array[...] = np.where(rhs1.array, rhs2.array, rhs3.array)

    def argwhere(self) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.argwhere()
        else:
            return EagerArray(np.argwhere(self.array))

    def trilu(self, rhs: Any, k: int, lower: bool) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.trilu(rhs, k, lower)
        else:
            if lower:
                self.array[:] = np.tril(rhs.array, k)
            else:
                self.array[:] = np.triu(rhs.array, k)

    def cholesky(self, src: Any) -> None:
        self.check_eager_args(src)
        if self.deferred is not None:
            self.deferred.cholesky(src)
        else:
            try:
                result = np.linalg.cholesky(src.array)
            except np.linalg.LinAlgError as e:
                from ..linalg import LinAlgError

                raise LinAlgError(e) from e

            self.array[:] = result

    def eig(self, ew: Any, ev: Any) -> None:
        self.check_eager_args(ew, ev)
        if self.deferred is not None and (
            runtime.num_gpus == 0 or runtime.cusolver_has_geev()
        ):
            self.deferred.eig(ew, ev)
        else:
            try:
                result_ew, result_ev = np.linalg.eig(self.array)
            except np.linalg.LinAlgError as e:
                from ..linalg import LinAlgError

                raise LinAlgError(e) from e
            ew.array[:] = result_ew
            ev.array[:] = result_ev

    def eigvals(self, ew: Any) -> None:
        self.check_eager_args(ew)
        if self.deferred is not None and (
            runtime.num_gpus == 0 or runtime.cusolver_has_geev()
        ):
            self.deferred.eigvals(ew)
        else:
            try:
                result_ew = np.linalg.eigvals(self.array)
            except np.linalg.LinAlgError as e:
                from ..linalg import LinAlgError

                raise LinAlgError(e) from e
            ew.array[:] = result_ew

    def eigh(self, ew: Any, ev: Any, uplo_l: bool) -> None:
        self.check_eager_args(ew, ev)
        if self.deferred is not None:
            self.deferred.eigh(ew, ev, uplo_l)
        else:
            try:
                result_ew, result_ev = np.linalg.eigh(
                    self.array, "L" if uplo_l else "U"
                )
            except np.linalg.LinAlgError as e:
                from ..linalg import LinAlgError

                raise LinAlgError(e) from e
            ew.array[:] = result_ew
            ev.array[:] = result_ev

    def eigvalsh(self, ew: Any, uplo_l: bool) -> None:
        self.check_eager_args(ew)
        if self.deferred is not None:
            self.deferred.eigvalsh(ew, uplo_l)
        else:
            try:
                result_ew = np.linalg.eigvalsh(
                    self.array, "L" if uplo_l else "U"
                )
            except np.linalg.LinAlgError as e:
                from ..linalg import LinAlgError

                raise LinAlgError(e) from e
            ew.array[:] = result_ew

    def qr(self, q: Any, r: Any) -> None:
        self.check_eager_args(q, r)
        if self.deferred is not None:
            self.deferred.qr(q, r)
        else:
            try:
                result_q, result_r = np.linalg.qr(self.array)
            except np.linalg.LinAlgError as e:
                from ..linalg import LinAlgError

                raise LinAlgError(e) from e
            q.array[:] = result_q
            r.array[:] = result_r

    def solve(self, a: Any, b: Any) -> None:
        self.check_eager_args(a, b)
        if self.deferred is not None:
            self.deferred.solve(a, b)
        else:
            try:
                result = np.linalg.solve(a.array, b.array)
            except np.linalg.LinAlgError as e:
                from ..linalg import LinAlgError

                raise LinAlgError(e) from e
            self.array[:] = result

    def svd(self, u: Any, s: Any, vh: Any) -> None:
        self.check_eager_args(u, s, vh)
        if self.deferred is not None:
            self.deferred.svd(u, s, vh)
        else:
            try:
                full_matrices = (
                    u.shape[0] == u.shape[1] and vh.shape[0] == vh.shape[1]
                )
                result_u, result_s, result_vh = np.linalg.svd(
                    self.array, full_matrices
                )
            except np.linalg.LinAlgError as e:
                from ..linalg import LinAlgError

                raise LinAlgError(e) from e
            u.array[:] = result_u
            s.array[:] = result_s
            vh.array[:] = result_vh

    def scan(
        self,
        op: int,
        rhs: Any,
        axis: int,
        dtype: npt.DTypeLike | None,
        nan_to_identity: bool,
    ) -> None:
        self.check_eager_args(rhs)
        if self.deferred is not None:
            self.deferred.scan(op, rhs, axis, dtype, nan_to_identity)
            return
        if op is ScanCode.SUM:
            if nan_to_identity is False:
                np.cumsum(rhs.array, axis, dtype, self.array)
            else:
                np.nancumsum(rhs.array, axis, dtype, self.array)
        elif op is ScanCode.PROD:
            if nan_to_identity is False:
                np.cumprod(rhs.array, axis, dtype, self.array)
            else:
                np.nancumprod(rhs.array, axis, dtype, self.array)
        else:
            raise RuntimeError(f"unsupported scan op {op}")

    def unique(self) -> NumPyThunk:
        if self.deferred is not None:
            return self.deferred.unique()
        else:
            return EagerArray(np.unique(self.array))

    def create_window(self, op_code: WindowOpCode, M: int, *args: Any) -> None:
        if self.deferred is not None:
            return self.deferred.create_window(op_code, M, *args)
        else:
            fn = _WINDOW_OPS[op_code]
            self.array[:] = fn(M, *args)

    def packbits(self, src: Any, axis: int | None, bitorder: BitOrder) -> None:
        self.check_eager_args(src)
        if self.deferred is not None:
            self.deferred.packbits(src, axis, bitorder)
        else:
            self.array[:] = np.packbits(
                src.array, axis=axis, bitorder=bitorder
            )

    def unpackbits(
        self, src: Any, axis: int | None, bitorder: BitOrder
    ) -> None:
        self.check_eager_args(src)
        if self.deferred is not None:
            self.deferred.unpackbits(src, axis, bitorder)
        else:
            self.array[:] = np.unpackbits(
                src.array, axis=axis, bitorder=bitorder
            )

    def _wrap(self, src: Any, new_len: int) -> None:
        self.check_eager_args(src)
        if self.deferred is not None:
            self.deferred._wrap(src, new_len)
        else:
            src_flat = np.ravel(src.array)
            if src_flat.size == new_len:
                self.array[:] = src_flat[:]
            elif src_flat.size > new_len:
                self.array[:] = src_flat[:new_len]
            else:
                reps = (new_len + src_flat.size - 1) // src_flat.size
                if reps > 1:
                    src_flat = np.tile(src_flat, reps)
                self.array[:] = src_flat[:new_len]

    def histogram(self, src: Any, bins: Any, weights: Any) -> None:
        self.check_eager_args(src, bins, weights)
        if self.deferred is not None:
            self.deferred.histogram(src, bins, weights)
        else:
            self.array[:], _ = np.histogram(
                src.array,
                cast(EagerArray, bins).array,
                weights=cast(EagerArray, weights).array,
            )

    def histogramdd(self, coords: Any, weights: Any, bins_set: Any) -> None:
        self.check_eager_args(coords, weights, bins_set)
        if self.deferred is not None:
            self.deferred.histogramdd(coords, weights, bins_set)
        else:
            weights_array = (
                cast(EagerArray, weights).array if weights else None
            )

            # numpy wants the output to be a D-dimensional array
            output_shape = tuple([len(b.array) - 1 for b in bins_set])
            try:
                output_np = np.reshape(
                    self.array, shape=output_shape, copy=False
                )
                inplace = True
            except ValueError:
                output_np = np.ndarray(output_shape, dtype=self.array.dtype)
                inplace = False

            output_np[...], _ = np.histogramdd(
                coords.array,
                bins=[cast(EagerArray, bins).array for bins in bins_set],
                weights=weights_array,
            )
            if not inplace:
                self.array[:] = output_np.reshape(self.array.shape)

    def stencil_hint(
        self, low_offsets: tuple[int, ...], high_offsets: tuple[int, ...]
    ) -> None:
        if self.deferred is not None:
            self.deferred.stencil_hint(low_offsets, high_offsets)

    def in1d(
        self,
        ar2: Any,
        assume_unique: bool = False,
        invert: bool = False,
        kind: Literal["sort", "table"] | None = None,
        ar2_min: int = 0,
        ar2_max: int = 0,
    ) -> NumPyThunk:
        self.check_eager_args(ar2)
        if self.deferred is not None:
            result = self.deferred.in1d(
                ar2, assume_unique, invert, kind, ar2_min, ar2_max
            )
            return result
        else:
            result = EagerArray(
                np.in1d(
                    self.array,
                    ar2.array,
                    assume_unique=assume_unique,
                    invert=invert,
                    kind=kind,
                )
            )
            return result
