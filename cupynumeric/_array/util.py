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


def add_boilerplate(
    *array_params: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Adds required boilerplate to the wrapped cupynumeric.ndarray or
    module-level function.

    Every time the wrapped function is called, this wrapper will convert all
    specified array-like parameters to cuPyNumeric ndarrays. Additionally, any
    "out" or "where" arguments will also always be automatically converted.
    """
    to_convert = set(array_params)
    assert len(to_convert) == len(array_params)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        assert not hasattr(
            func, "__wrapped__"
        ), "this decorator must be the innermost"

        params = signature(func).parameters
        extra = to_convert - set(params)
        assert len(extra) == 0, f"unknown parameter(s): {extra}"

        # we also always want to convert "out" and "where"
        # even if they are not explicitly specified by the user
        to_convert.update(("out", "where"))

        out_idx = -1
        indices = set()
        for idx, param in enumerate(params):
            if param == "out":
                out_idx = idx
            if param in to_convert:
                indices.add(idx)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            # convert specified non-None positional arguments, making sure
            # that any out-parameters are appropriately writeable
            converted_args = []
            for idx, arg in enumerate(args):
                if idx in indices and arg is not None:
                    if idx == out_idx:
                        arg = convert_to_cupynumeric_ndarray(arg, share=True)
                        if not arg.flags.writeable:
                            raise ValueError("out is not writeable")
                    else:
                        arg = convert_to_cupynumeric_ndarray(arg)
                converted_args.append(arg)
            args = tuple(converted_args)

            # convert specified non-None keyword arguments, making sure
            # that any out-parameters are appropriately writeable
            for k, v in kwargs.items():
                if k in to_convert and v is not None:
                    if k == "out":
                        kwargs[k] = convert_to_cupynumeric_ndarray(
                            v, share=True
                        )
                        if not kwargs[k].flags.writeable:
                            raise ValueError("out is not writeable")
                    else:
                        kwargs[k] = convert_to_cupynumeric_ndarray(v)

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
            "expected a sequence of integers or a single integer, "
            f"got {shape!r}"
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
    array_types = list()
    scalars = list()
    for array in args:
        if array.ndim == 0:
            scalars.append(array.dtype.type(0))
        else:
            array_types.append(array.dtype)
    return np.result_type(*array_types, *scalars)


T = TypeVar("T")


def tuple_pop(tup: tuple[T, ...], index: int) -> tuple[T, ...]:
    return tup[:index] + tup[index + 1 :]
