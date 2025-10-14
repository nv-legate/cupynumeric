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

from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from legate.core.utils import OrderedSet

from cupynumeric._utils import is_np2_1

from .._array.thunk import perform_unary_reduction
from .._array.util import (
    add_boilerplate,
    check_writeable,
    convert_to_cupynumeric_ndarray,
)
from ..config import BinaryOpCode, UnaryOpCode, UnaryRedCode
from ..types import NdShape

if TYPE_CHECKING:
    from typing import Any, Callable, Sequence, TypeAlias
    import numpy.typing as npt

    from .._array.array import ndarray
    from ..types import CastingKind

    PostResolutionCheckFunc: TypeAlias = Callable[
        [ndarray, ndarray, Any, Any, BinaryOpCode],
        tuple[ndarray, ndarray, BinaryOpCode],
    ]


_UNARY_DOCSTRING_TEMPLATE = """{}

Parameters
----------
x : array_like
    Input array.
out : ndarray, or None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
out : ndarray or scalar
    Result.
    This is a scalar if `x` is a scalar.

See Also
--------
numpy.{}

Availability
------------
Multiple GPUs, Multiple CPUs
"""

_MULTIOUT_UNARY_DOCSTRING_TEMPLATE = """{}

Parameters
----------
x : array_like
    Input array.
out : tuple[ndarray or None], or None, optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y1 : ndarray
    This is a scalar if `x` is a scalar.
y2 : ndarray
    This is a scalar if `x` is a scalar.

See Also
--------
numpy.{}

Availability
------------
Multiple GPUs, Multiple CPUs
"""

_BINARY_DOCSTRING_TEMPLATE = """{}

Parameters
----------
x1, x2 : array_like
    Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable
    to a common shape (which becomes the shape of the output).
out : ndarray, None, or tuple[ndarray or None], optional
    A location into which the result is stored. If provided, it must have
    a shape that the inputs broadcast to. If not provided or None,
    a freshly-allocated array is returned. A tuple (possible only as a
    keyword argument) must have length equal to the number of outputs.
where : array_like, optional
    This condition is broadcast over the input. At locations where the
    condition is True, the `out` array will be set to the ufunc result.
    Elsewhere, the `out` array will retain its original value.
    Note that if an uninitialized `out` array is created via the default
    ``out=None``, locations within it where the condition is False will
    remain uninitialized.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y : ndarray or scalar
    Result.
    This is a scalar if both `x1` and `x2` are scalars.

See Also
--------
numpy.{}

Availability
------------
Multiple GPUs, Multiple CPUs
"""

float_dtypes = ["e", "f", "d"]

complex_dtypes = ["F", "D"]

float_and_complex = float_dtypes + complex_dtypes

integer_dtypes = ["b", "B", "h", "H", "i", "I", "l", "L", "q", "Q"]

all_but_boolean = integer_dtypes + float_and_complex

all_dtypes = ["?"] + all_but_boolean


def predicate_types_of(dtypes: Sequence[str]) -> list[str]:
    return [ty + "?" for ty in dtypes]


def relation_types_of(dtypes: Sequence[str]) -> list[str]:
    return [ty * 2 + "?" for ty in dtypes]


def to_dtypes(chars: str) -> tuple[np.dtype[Any], ...]:
    return tuple(np.dtype(char) for char in chars)


def _get_kind_score(kind: type) -> int:
    if issubclass(kind, np.bool_):
        return 0
    if issubclass(kind, (np.integer, int)):
        return 1
    if issubclass(kind, (np.inexact, float, complex)):
        return 2
    # unknown type, assume higher score
    return 3


def _check_should_use_weak_scalar(key: tuple[str | type, ...]) -> bool:
    """Helper function for promotion, where we need to check whether we
    should use weak promotion for python floats/integers (NEP 50/NumPy 2).
    """
    max_scalar_kind = -1
    max_array_kind = -1

    for in_t in key:
        if isinstance(in_t, str):
            kind = _get_kind_score(np.dtype(in_t).type)
            max_array_kind = max(max_array_kind, kind)
        else:
            kind = _get_kind_score(in_t)
            max_scalar_kind = max(max_scalar_kind, kind)

    all_scalars_or_arrays = max_scalar_kind == -1 or max_array_kind == -1
    return not all_scalars_or_arrays and max_array_kind >= max_scalar_kind


def _check_where(where: Any) -> None:
    if not isinstance(where, bool) or not where:
        raise NotImplementedError("the 'where' keyword is not yet supported")


def _default_post_resolution_check(
    arr_x: ndarray,
    arr_y: ndarray,
    obj_x: Any,
    obj_y: Any,
    op_code: BinaryOpCode,
) -> tuple[ndarray, ndarray, BinaryOpCode]:
    """Check whether Python integers fit into integer operand dtypes.
    This check is overloaded by comparisons to always succeed.
    """
    if type(obj_x) is int and arr_x.dtype.kind in "iu":
        # Check if original Python integer fits first operand.
        arr_x.dtype.type(obj_x)
    if type(obj_y) is int and arr_y.dtype.kind in "iu":
        # Check if original Python integer fits second operand.
        arr_y.dtype.type(obj_y)

    return arr_x, arr_y, op_code


T = TypeVar("T")


class ufunc(Generic[T]):
    _types: dict[Any, str]
    _nin: int
    _nout: int
    _op_code: T

    def __init__(self, name: str, doc: str, op_code: T) -> None:
        self._name = name
        self._op_code = op_code
        self.__doc__ = doc

    @property
    def __name__(self) -> str:
        return self._name

    @property
    def nin(self) -> int:
        return self._nin

    @property
    def nout(self) -> int:
        return self._nout

    @property
    def types(self) -> list[str]:
        return [
            f"{''.join(in_tys)}->{''.join(out_tys)}"
            for in_tys, out_tys in self._types.items()
        ]

    @property
    def ntypes(self) -> int:
        return len(self._types)

    def _maybe_cast_input(
        self, arr: ndarray, to_dtype: np.dtype[Any], casting: CastingKind
    ) -> ndarray:
        if arr.dtype == to_dtype:
            return arr

        if not np.can_cast(arr.dtype, to_dtype, casting=casting):
            raise TypeError(
                f"Cannot cast ufunc '{self._name}' input from "
                f"{arr.dtype} to {to_dtype} with casting rule '{casting}'"
            )

        return arr._astype(to_dtype, temporary=True)

    def _maybe_create_result(
        self,
        out: ndarray | None,
        out_shape: NdShape,
        res_dtype: np.dtype[Any],
        casting: CastingKind,
        inputs: tuple[ndarray, ...],
    ) -> ndarray:
        from .._array.array import ndarray

        if out is None:
            return ndarray._from_inputs(
                shape=out_shape, dtype=res_dtype, inputs=inputs
            )
        elif out.dtype != res_dtype:
            if not np.can_cast(res_dtype, out.dtype, casting=casting):
                raise TypeError(
                    f"Cannot cast ufunc '{self._name}' output from "
                    f"{res_dtype} to {out.dtype} with casting rule "
                    f"'{casting}'"
                )
            return ndarray._from_inputs(
                shape=out.shape, dtype=res_dtype, inputs=inputs
            )
        else:
            return out

    @staticmethod
    def _maybe_cast_output(out: ndarray | None, result: ndarray) -> ndarray:
        if out is None or out is result:
            return result
        out._thunk.convert(result._thunk, warn=False)
        return out

    @staticmethod
    def _maybe_convert_output_to_cupynumeric_ndarray(
        out: ndarray | npt.NDArray[Any] | None,
    ) -> ndarray | None:
        from .._array.array import ndarray

        if out is None:
            return None
        if isinstance(out, ndarray):
            return out
        if isinstance(out, np.ndarray):
            return convert_to_cupynumeric_ndarray(out, share=True)
        raise TypeError("return arrays must be of ArrayType")

    def _prepare_operands(
        self, *args: Any, out: ndarray | tuple[ndarray, ...] | None
    ) -> tuple[Sequence[ndarray], Sequence[ndarray | None], tuple[int, ...]]:
        max_nargs = self.nin + self.nout
        if len(args) < self.nin or len(args) > max_nargs:
            raise TypeError(
                f"{self._name}() takes from {self.nin} to {max_nargs} "
                f"positional arguments but {len(args)} were given"
            )

        inputs = tuple(
            convert_to_cupynumeric_ndarray(arr) for arr in args[: self.nin]
        )

        if len(args) > self.nin:
            if out is not None:
                raise TypeError(
                    "cannot specify 'out' as both a positional and keyword argument"
                )
            computed_out = args[self.nin :]
            # Missing outputs are treated as Nones
            computed_out += (None,) * (self.nout - len(computed_out))
        elif out is None:
            computed_out = (None,) * self.nout
        elif not isinstance(out, tuple):
            computed_out = (out,)
        else:
            computed_out = out

        outputs = tuple(
            self._maybe_convert_output_to_cupynumeric_ndarray(arr)
            for arr in computed_out
        )

        if self.nout != len(outputs):
            raise ValueError(
                "The 'out' tuple must have exactly one entry per ufunc output"
            )

        shapes = [arr.shape for arr in inputs]
        shapes.extend(arr.shape for arr in outputs if arr is not None)

        # Check if the broadcasting is possible
        out_shape = np.broadcast_shapes(*shapes)

        for out in outputs:
            if out is not None and out.shape != out_shape:
                raise ValueError(
                    f"non-broadcastable output operand with shape "
                    f"{out.shape} doesn't match the broadcast shape "
                    f"{out_shape}"
                )
            check_writeable(out)

        return inputs, outputs, out_shape

    def __repr__(self) -> str:
        return f"<ufunc {self._name}>"


class unary_ufunc(ufunc[UnaryOpCode]):
    _nin = 1
    _nout = 1

    def __init__(
        self,
        name: str,
        doc: str,
        op_code: UnaryOpCode,
        types: dict[str, str],
        overrides: dict[str, UnaryOpCode],
    ) -> None:
        super().__init__(name, doc, op_code)

        self._types = types
        assert len(self._types)

        in_ty, out_ty = next(iter(self._types.items()))
        assert len(in_ty) == self.nin
        assert len(out_ty) == self.nout

        self._resolution_cache: dict[np.dtype[Any], np.dtype[Any]] = {}
        self._overrides = overrides

    def _resolve_dtype(
        self, arr: ndarray, precision_fixed: bool
    ) -> tuple[ndarray, np.dtype[Any]]:
        if arr.dtype.char in self._types:
            return arr, np.dtype(self._types[arr.dtype.char])

        if not precision_fixed:
            if arr.dtype in self._resolution_cache:
                to_dtype = self._resolution_cache[arr.dtype]
                arr = arr._astype(to_dtype, temporary=True)
                return arr, np.dtype(self._types[to_dtype.char])

        chosen = None
        if not precision_fixed:
            for in_ty in self._types.keys():
                if np.can_cast(arr.dtype, in_ty):
                    chosen = in_ty
                    break

        if chosen is None:
            raise TypeError(
                f"No matching signature of ufunc {self._name} is found "
                "for the given casting"
            )

        to_dtype = np.dtype(chosen)
        self._resolution_cache[arr.dtype] = to_dtype

        return arr._astype(to_dtype, temporary=True), np.dtype(
            self._types[to_dtype.char]
        )

    def __call__(
        self,
        *args: Any,
        out: ndarray | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> ndarray:
        x = convert_to_cupynumeric_ndarray(args[0])
        if len(args) > self.nin:
            if out is not None:
                raise TypeError(
                    "cannot specify 'out' as both a positional and keyword argument"
                )
            out = args[self.nin]
        result = getattr(x._thunk, f"_{self._name}")(
            out=out, where=where, casting=casting, order=order, dtype=dtype
        )
        return convert_to_cupynumeric_ndarray(result)

    def _call_full(
        self,
        *args: Any,
        out: ndarray | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> ndarray:
        _check_where(where)

        (x,), (out,), out_shape = self._prepare_operands(*args, out=out)

        # If no dtype is given to prescribe the accuracy, we use the dtype
        # of the input
        precision_fixed = False
        if dtype is not None:
            # If a dtype is given, that determines the precision
            # of the computation.
            precision_fixed = True
            x = self._maybe_cast_input(x, dtype, casting)

        if (
            self._name in {"ceil", "floor", "trunc"}
            and is_np2_1
            and np.issubdtype(x.dtype, np.integer)
        ):
            result = x
            return self._maybe_cast_output(out, result)

        # Resolve the dtype to use for the computation and cast the input
        # if necessary. If the dtype is already fixed by the caller,
        # the dtype must be one of the dtypes supported by this operation.
        x, res_dtype = self._resolve_dtype(x, precision_fixed)

        result = self._maybe_create_result(
            out, out_shape, res_dtype, casting, (x,)
        )

        op_code = self._overrides.get(x.dtype.char, self._op_code)
        result._thunk.unary_op(op_code, x._thunk, where)

        return self._maybe_cast_output(out, result)


class multiout_unary_ufunc(ufunc[UnaryOpCode]):
    _nin = 1

    def __init__(
        self, name: str, doc: str, op_code: UnaryOpCode, types: dict[Any, Any]
    ) -> None:
        super().__init__(name, doc, op_code)

        self._types = types
        assert len(self._types)

        in_ty, out_ty = next(iter(self._types.items()))
        self._nout = len(out_ty)
        assert len(in_ty) == self.nin

        self._resolution_cache: dict[np.dtype[Any], np.dtype[Any]] = {}

    def _resolve_dtype(
        self, arr: ndarray, precision_fixed: bool
    ) -> tuple[ndarray, tuple[np.dtype[Any], ...]]:
        if arr.dtype.char in self._types:
            return arr, to_dtypes(self._types[arr.dtype.char])

        if not precision_fixed:
            if arr.dtype in self._resolution_cache:
                to_dtype = self._resolution_cache[arr.dtype]
                arr = arr._astype(to_dtype, temporary=True)
                return arr, to_dtypes(self._types[to_dtype.char])

        chosen = None
        if not precision_fixed:
            for in_ty in self._types.keys():
                if np.can_cast(arr.dtype, in_ty):
                    chosen = in_ty
                    break

        if chosen is None:
            raise TypeError(
                f"No matching signature of ufunc {self._name} is found "
                "for the given casting"
            )

        to_dtype = np.dtype(chosen)
        self._resolution_cache[arr.dtype] = to_dtype

        return arr._astype(to_dtype, temporary=True), to_dtypes(
            self._types[to_dtype.char]
        )

    def __call__(
        self,
        *args: Any,
        out: ndarray | tuple[ndarray, ...] | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[ndarray, ...]:
        _check_where(where)

        (x,), outs, out_shape = self._prepare_operands(*args, out=out)

        # If no dtype is given to prescribe the accuracy, we use the dtype
        # of the input
        precision_fixed = False
        if dtype is not None:
            # If a dtype is given, that determines the precision
            # of the computation.
            precision_fixed = True
            x = self._maybe_cast_input(x, dtype, casting)

        # Resolve the dtype to use for the computation and cast the input
        # if necessary. If the dtype is already fixed by the caller,
        # the dtype must be one of the dtypes supported by this operation.
        x, res_dtypes = self._resolve_dtype(x, precision_fixed)

        results = tuple(
            self._maybe_create_result(out, out_shape, res_dtype, casting, (x,))
            for out, res_dtype in zip(outs, res_dtypes)
        )

        result_thunks = tuple(result._thunk for result in results)
        result_thunks[0].unary_op(
            self._op_code, x._thunk, where, (), multiout=result_thunks[1:]
        )

        return tuple(
            self._maybe_cast_output(out, result)
            for out, result in zip(outs, results)
        )


class binary_ufunc(ufunc[BinaryOpCode]):
    _nin = 2
    _nout = 1
    _post_resolution_check: PostResolutionCheckFunc

    def __init__(
        self,
        name: str,
        doc: str,
        op_code: BinaryOpCode,
        types: dict[tuple[str, str], str],
        red_code: UnaryRedCode | None = None,
        use_common_type: bool = True,
        post_resolution_check: PostResolutionCheckFunc | None = None,
    ) -> None:
        super().__init__(name, doc, op_code)

        self._types = types
        assert len(self._types)

        in_ty, out_ty = next(iter(self._types.items()))
        assert len(in_ty) == self.nin
        assert len(out_ty) == self.nout

        self._resolution_cache: dict[
            tuple[str | type, ...], tuple[np.dtype[Any], ...]
        ] = {}
        self._red_code = red_code
        self._use_common_type = use_common_type
        if post_resolution_check is None:
            self._post_resolution_check = _default_post_resolution_check
        else:
            self._post_resolution_check = post_resolution_check

    @staticmethod
    def _find_common_type(
        arrs: Sequence[ndarray], orig_args: Sequence[Any]
    ) -> np.dtype[Any]:
        from .._array.array import ndarray

        all_ndarray = all(isinstance(arg, ndarray) for arg in orig_args)
        unique_dtypes = OrderedSet(arr.dtype for arr in arrs)
        # If all operands are ndarrays and they all have the same dtype,
        # we already know the common dtype
        if len(unique_dtypes) == 1 and all_ndarray:
            return arrs[0].dtype

        scalar_types = []
        array_types = []
        for arr, orig_arg in zip(arrs, orig_args):
            if type(orig_arg) in (int, float, complex):
                scalar_types.append(orig_arg)
            else:
                array_types.append(arr.dtype)

        return np.result_type(*array_types, *scalar_types)

    def _resolve_dtype(
        self,
        arrs: Sequence[ndarray],
        orig_args: Sequence[Any],
        casting: CastingKind,
        precision_fixed: bool,
    ) -> tuple[Sequence[ndarray], np.dtype[Any]]:
        to_dtypes: tuple[np.dtype[Any], ...]
        key: tuple[str | type, ...]
        if self._use_common_type:
            common_dtype = self._find_common_type(arrs, orig_args)
            to_dtypes = (common_dtype, common_dtype)
            key = (common_dtype.char, common_dtype.char)
        else:
            to_dtypes = tuple(arr.dtype for arr in arrs)
            key = tuple(
                arr.dtype.char
                if type(orig) not in (int, float, complex)
                else type(orig)
                for orig, arr in zip(orig_args, arrs)
            )
            # When all inputs are scalars, cannot use weak logic below.
            # (Using arr.dtype.char may be off for huge integers that map to
            # an unsigned int.  But NumPy should mostly do the same currently.)
            if not _check_should_use_weak_scalar(key):
                key = tuple(arr.dtype.char for arr in arrs)

        if key in self._types:
            arrs = [
                arr._astype(to_dtype, temporary=True)
                for arr, to_dtype in zip(arrs, to_dtypes)
            ]
            return arrs, np.dtype(self._types[key])

        if not precision_fixed:
            if key in self._resolution_cache:
                to_dtypes = self._resolution_cache[key]
                arrs = [
                    arr._astype(to_dtype, temporary=True)
                    for arr, to_dtype in zip(arrs, to_dtypes)
                ]
                return arrs, np.dtype(self._types[to_dtypes])

        chosen = None
        if not precision_fixed:
            for in_dtypes in self._types.keys():
                for in_t, to_dtype in zip(key, in_dtypes):
                    # Break if `to_dtype` doesn't work.
                    if isinstance(in_t, str):
                        if not np.can_cast(in_t, to_dtype):
                            break
                    else:
                        # In NumPy 2, the value doesn't matter.  In NumPy 1.x
                        # it could matter (but caching wouldn't work anyway).
                        if np.result_type(in_t(0), to_dtype) != to_dtype:
                            break
                else:
                    # dtypes OK (no break), choose them and break outer
                    chosen = in_dtypes
                    break

            # If there's no safe match and the operands have different types,
            # try to find a match based on the leading operand
            if chosen is None and not self._use_common_type:
                for in_dtypes in self._types.keys():
                    if not np.can_cast(arrs[0].dtype, in_dtypes[0]):
                        # Check next in_dtypes
                        continue

                    for in_t, to_dtype in zip(key[1:], in_dtypes[1:]):
                        # Break if `to_dtype` doesn't work.
                        if isinstance(in_t, str):
                            if not np.can_cast(
                                in_t, to_dtype, casting=casting
                            ):
                                break
                        elif casting != "unsafe":
                            # Same-kind/safe can use result_type (see above)
                            if np.result_type(in_t(0), to_dtype) != to_dtype:
                                break
                    else:
                        # dtypes OK (no break), choose them and break outer
                        chosen = in_dtypes
                        break

        if chosen is None:
            raise TypeError(
                f"No matching signature of ufunc {self._name} is found "
                "for the given casting"
            )

        self._resolution_cache[key] = chosen
        arrs = [
            arr._astype(to_dtype, temporary=True)
            for arr, to_dtype in zip(arrs, chosen)
        ]

        return arrs, np.dtype(self._types[chosen])

    def __call__(
        self,
        *args: Any,
        out: ndarray | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> ndarray:
        return self._call_full(
            *args,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
        )

    def _call_full(
        self,
        *args: Any,
        out: ndarray | None = None,
        where: bool = True,
        casting: CastingKind = "same_kind",
        order: str = "K",
        dtype: np.dtype[Any] | None = None,
    ) -> ndarray:
        _check_where(where)

        arrs, (out,), out_shape = self._prepare_operands(*args, out=out)

        orig_args = args[: self.nin]

        # If no dtype is given to prescribe the accuracy, we use the dtype
        # of the input
        precision_fixed = False
        if dtype is not None:
            # If a dtype is given, that determines the precision
            # of the computation.
            precision_fixed = True
            arrs = [
                self._maybe_cast_input(arr, dtype, casting) for arr in arrs
            ]

        # Resolve the dtype to use for the computation and cast the input
        # if necessary. If the dtype is already fixed by the caller,
        # the dtype must be one of the dtypes supported by this operation.
        arrs, res_dtype = self._resolve_dtype(
            arrs, orig_args, casting, precision_fixed
        )

        # Check python integers operands.  For comparisons, this may return
        # new values and op_code when the integer is out-of-bounds.
        x1, x2 = arrs
        x1, x2, op_code = self._post_resolution_check(
            x1, x2, orig_args[0], orig_args[1], self._op_code
        )

        result = self._maybe_create_result(
            out, out_shape, res_dtype, casting, (x1, x2)
        )
        result._thunk.binary_op(op_code, x1._thunk, x2._thunk, where, ())

        return self._maybe_cast_output(out, result)

    @add_boilerplate("array")
    def reduce(
        self,
        array: ndarray,
        axis: int | tuple[int, ...] | None = 0,
        dtype: np.dtype[Any] | None = None,
        out: ndarray | None = None,
        keepdims: bool = False,
        initial: Any | None = None,
        where: ndarray | None = None,
    ) -> ndarray:
        """
        reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial=<no
        value>, where=True)

        Reduces `array`'s dimension by one, by applying ufunc along one axis.

        For example, add.reduce() is equivalent to sum().

        Parameters
        ----------
        array : array_like
            The array to act on.
        axis : None or int or tuple of ints, optional
            Axis or axes along which a reduction is performed.  The default
            (`axis` = 0) is perform a reduction over the first dimension of the
            input array. `axis` may be negative, in which case it counts from
            the last to the first axis.
        dtype : data-type code, optional
            The type used to represent the intermediate results. Defaults to
            the data-type of the output array if this is provided, or the
            data-type
            of the input array if no output array is provided.
        out : ndarray, None, or tuple of ndarray and None, optional
            A location into which the result is stored. If not provided or
            None, a freshly-allocated array is returned. For consistency with
            ``ufunc.__call__``, if given as a keyword, this may be wrapped in a
            1-element tuple.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result
            will broadcast correctly against the original `array`.
        initial : scalar, optional
            The value with which to start the reduction.  If the ufunc has no
            identity or the dtype is object, this defaults to None - otherwise
            it defaults to ufunc.identity.  If ``None`` is given, the first
            element of the reduction is used, and an error is thrown if the
            reduction is empty.
        where : array_like of bool, optional
            A boolean array which is broadcasted to match the dimensions of
            `array`, and selects elements to include in the reduction. Note
            that for ufuncs like ``minimum`` that do not have an identity
            defined, one has to pass in also ``initial``.

        Returns
        -------
        r : ndarray
            The reduced array. If `out` was supplied, `r` is a reference to it.

        See Also
        --------
        numpy.ufunc.reduce
        """
        if self._red_code is None:
            raise NotImplementedError(
                f"reduction for {self} is not yet implemented"
            )

        # NumPy seems to be using None as the default axis value for scalars
        if array.ndim == 0 and axis == 0:
            axis = None

        # TODO: Unary reductions still need to be refactored
        return perform_unary_reduction(
            self._red_code,
            array,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )


def _parse_unary_ufunc_type(ty: str) -> tuple[str, str]:
    if len(ty) == 1:
        return (ty, ty)
    return (ty[0], ty[1:])


def create_unary_ufunc(
    summary: str,
    name: str,
    op_code: UnaryOpCode,
    types: Sequence[str],
    overrides: dict[str, UnaryOpCode] = {},
) -> unary_ufunc:
    doc = _UNARY_DOCSTRING_TEMPLATE.format(summary, name)
    types_dict = dict(_parse_unary_ufunc_type(ty) for ty in types)
    return unary_ufunc(name, doc, op_code, types_dict, overrides)


def create_multiout_unary_ufunc(
    summary: str, name: str, op_code: UnaryOpCode, types: Sequence[str]
) -> multiout_unary_ufunc:
    doc = _MULTIOUT_UNARY_DOCSTRING_TEMPLATE.format(summary, name)
    types_dict = dict(_parse_unary_ufunc_type(ty) for ty in types)
    return multiout_unary_ufunc(name, doc, op_code, types_dict)


def _parse_binary_ufunc_type(ty: str) -> tuple[tuple[str, str], str]:
    if len(ty) == 1:
        return ((ty, ty), ty)
    if len(ty) == 3:
        return ((ty[0], ty[1]), ty[2])
    raise NotImplementedError(
        "Binary ufunc must have two inputs and one output"
    )


def create_binary_ufunc(
    summary: str,
    name: str,
    op_code: BinaryOpCode,
    types: Sequence[str],
    red_code: UnaryRedCode | None = None,
    use_common_type: bool = True,
    post_resolution_check: PostResolutionCheckFunc | None = None,
) -> binary_ufunc:
    doc = _BINARY_DOCSTRING_TEMPLATE.format(summary, name)
    types_dict = dict(_parse_binary_ufunc_type(ty) for ty in types)
    return binary_ufunc(
        name,
        doc,
        op_code,
        types_dict,
        red_code=red_code,
        use_common_type=use_common_type,
        post_resolution_check=post_resolution_check,
    )
