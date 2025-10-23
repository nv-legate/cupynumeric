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
from functools import wraps
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
)

import numpy as np

from ..runtime import runtime
from ..settings import settings
from ..types import NdShape
from .doctor import doctor

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShapeLike
    from .array import ndarray


R = TypeVar("R")
P = ParamSpec("P")


def _compute_param_indices(
    func: Callable[P, R], to_convert: set[str]
) -> tuple[set[int], int]:
    # compute the positional index for all of the user-provided argument
    # names, specifically noting the index of an "out" param, if present
    params = signature(func).parameters
    extra = to_convert - set(params) - {"out", "where"}
    assert len(extra) == 0, f"unknown parameter(s): {extra}"

    out_index = -1
    indices = set()
    for idx, param in enumerate(params):
        if param == "out":
            out_index = idx
        if param in to_convert:
            indices.add(idx)

    return indices, out_index


def _convert_args(
    args: tuple[Any, ...], indices: set[int], out_idx: int
) -> tuple[Any, ...]:
    # convert specified non-None positional arguments, making sure
    # that any out-parameters are appropriately writeable
    converted = []
    for idx, arg in enumerate(args):
        if idx in indices and arg is not None:
            if idx == out_idx:
                arg = convert_to_cupynumeric_ndarray(arg, share=True)
                if not arg.flags.writeable:
                    raise ValueError("out is not writeable")
            else:
                arg = convert_to_cupynumeric_ndarray(arg)
        converted.append(arg)
    return tuple(converted)


def _convert_kwargs(
    kwargs: dict[str, Any], to_convert: set[str]
) -> dict[str, Any]:
    # convert specified non-None keyword arguments, making sure
    # that any out-parameters are appropriately writeable
    converted = dict(kwargs)
    for k, v in kwargs.items():
        if k in to_convert and v is not None:
            if k == "out":
                converted[k] = convert_to_cupynumeric_ndarray(v, share=True)
                if not converted[k].flags.writeable:
                    raise ValueError("out is not writeable")
            else:
                converted[k] = convert_to_cupynumeric_ndarray(v)
    return converted


def _convert_cupynumeric_to_numpy(obj: Any) -> Any:
    """
    Convert cuPyNumeric arrays to NumPy arrays, handling one level of nesting.
    Used in fallback to prevent __array_function__ recursion.
    """
    if isinstance(obj, (list, tuple)):
        return type(obj)(maybe_convert_to_np_ndarray(item) for item in obj)
    return maybe_convert_to_np_ndarray(obj)


def add_boilerplate(
    *array_params: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Adds required boilerplate to the wrapped cupynumeric.ndarray or
    module-level function.

    Every time the wrapped function is called, this wrapper will convert all
    specified array-like parameters to cuPyNumeric ndarrays. Additionally, any
    "out" or "where" arguments will also always be automatically converted.

    If conversion fails (e.g., for unsupported dtypes like strings or objects),
    the function automatically falls back to NumPy's implementation and emits
    a RuntimeWarning to inform the user.

    Parameters
    ----------
    *array_params : str
        Names of parameters to convert to cuPyNumeric ndarrays
    """
    to_convert = set(array_params)
    assert len(to_convert) == len(array_params)

    # we also always want to convert "out" and "where"
    # even if they are not explicitly specified by the user
    to_convert.update(("out", "where"))

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        assert not hasattr(func, "__wrapped__"), "apply add_boilerplate first"

        indices, out_index = _compute_param_indices(func, to_convert)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            try:
                args = _convert_args(args, indices, out_index)
                kwargs = _convert_kwargs(kwargs, to_convert)
            except (TypeError, ValueError, NotImplementedError) as e:
                import warnings

                warnings.warn(
                    f"cuPyNumeric does not support the provided input type for "
                    f"{func.__name__} and is falling back to NumPy. "
                    f"({type(e).__name__}: {e})",
                    RuntimeWarning,
                    stacklevel=2,
                )

                # Convert all arguments to NumPy
                numpy_args = tuple(
                    _convert_cupynumeric_to_numpy(arg) for arg in args
                )
                numpy_kwargs = {
                    k: _convert_cupynumeric_to_numpy(v)
                    for k, v in kwargs.items()
                }

                # Fallback only works for top-level numpy module functions (e.g., numpy.lexsort).
                # Does NOT work for:
                # - ndarray methods (e.g., ndarray.sort)
                # - submodule functions (e.g., numpy.linalg.svd, numpy.fft.fft)
                numpy_func = getattr(np, func.__name__, None)
                if numpy_func is None:
                    raise
                # Use __wrapped__ to bypass __array_function__ protocol
                result = (
                    numpy_func.__wrapped__(*numpy_args, **numpy_kwargs)
                    if hasattr(numpy_func, "__wrapped__")
                    else numpy_func(*numpy_args, **numpy_kwargs)
                )

                # Try to convert result back to cupynumeric if possible
                if isinstance(result, np.ndarray):
                    try:
                        return convert_to_cupynumeric_ndarray(result)  # type: ignore[return-value]
                    except TypeError:
                        # Result has unsupported dtype, return NumPy array as-is
                        return result  # type: ignore[return-value]
                return result  # type: ignore[no-any-return]

            if settings.doctor():
                doctor.diagnose(func.__name__, args, kwargs)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def broadcast_where(where: ndarray | None, shape: NdShape) -> ndarray | None:
    if where is not None and where.shape != shape:
        from .._module import broadcast_to

        where = broadcast_to(where, shape)
    return where


def convert_to_cupynumeric_ndarray(obj: Any, share: bool = False) -> ndarray:
    from .array import ndarray
    from .._thunk.thunk import NumPyThunk

    # If this is an instance of one of our ndarrays then we're done
    if isinstance(obj, ndarray):
        return obj
    if isinstance(obj, NumPyThunk):
        thunk = obj
    else:
        # Ask the runtime to make a numpy thunk for this object
        thunk = runtime.get_numpy_thunk(obj, share=share)
    writeable = (
        obj.flags.writeable if isinstance(obj, np.ndarray) and share else True
    )
    return ndarray._from_thunk(thunk, writeable=writeable)


def maybe_convert_to_np_ndarray(obj: Any) -> Any:
    """
    Converts cuPyNumeric arrays into NumPy arrays, otherwise has no effect.
    """
    from ..ma import MaskedArray
    from .array import ndarray

    if isinstance(obj, (ndarray, MaskedArray)):
        return obj.__array__()
    return obj


def check_writeable(arr: ndarray | tuple[ndarray, ...] | None) -> None:
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


def sanitize_shape(
    shape: NdShapeLike | Sequence[Any] | npt.NDArray[Any] | ndarray,
) -> NdShape:
    from .array import ndarray

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
            f"expected a sequence of integers or a single integer, got {shape!r}"
        )
    return result


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
    array_types = [array.dtype for array in args]
    return np.result_type(*array_types)


T = TypeVar("T")


def tuple_pop(tup: tuple[T, ...], index: int) -> tuple[T, ...]:
    return tup[:index] + tup[index + 1 :]
