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

from .._utils.profiling import profiling_wrapper
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
) -> tuple[tuple[int, ...], int]:
    # compute the positional index for all of the user-provided argument
    # names, specifically noting the index of an "out" param, if present
    params = signature(func).parameters
    extra = to_convert - set(params) - {"out", "where"}
    assert len(extra) == 0, f"unknown parameter(s): {extra}"

    out_index = -1
    indices = []
    for idx, param in enumerate(params):
        if param == "out":
            out_index = idx
        if param in to_convert:
            indices.append(idx)

    return tuple(indices), out_index


def _convert_args(
    args: tuple[Any, ...], indices: tuple[int, ...], out_idx: int
) -> tuple[Any, ...]:
    # convert specified non-None positional arguments, making sure
    # that any out-parameters are appropriately writeable
    converted: list[Any] | None = None
    for idx in indices:
        if idx >= len(args):
            continue
        arg = args[idx]
        if arg is None:
            continue
        if idx == out_idx:
            arg = convert_to_cupynumeric_ndarray(arg, share=True)
            if not arg.flags.writeable:
                raise ValueError("out is not writeable")
        else:
            arg = convert_to_cupynumeric_ndarray(arg)
        if converted is None:
            converted = list(args)
        converted[idx] = arg
    return args if converted is None else tuple(converted)


def _convert_kwargs(
    kwargs: dict[str, Any], to_convert: set[str]
) -> dict[str, Any]:
    # convert specified non-None keyword arguments, making sure
    # that any out-parameters are appropriately writeable
    converted: dict[str, Any] | None = None
    for k in to_convert:
        v = kwargs.get(k)
        if v is None:
            continue
        if k == "out":
            v = convert_to_cupynumeric_ndarray(v, share=True)
            if not v.flags.writeable:
                raise ValueError("out is not writeable")
        else:
            v = convert_to_cupynumeric_ndarray(v)
        if converted is None:
            converted = dict(kwargs)
        converted[k] = v
    return kwargs if converted is None else converted


def add_boilerplate(
    *array_params: str, name: str | None = None, prefix: str | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Adds required boilerplate to the wrapped cupynumeric.ndarray or
    module-level function.

    Every time the wrapped function is called, this wrapper will convert all
    specified array-like parameters to cuPyNumeric ndarrays. Additionally, any
    "out" or "where" arguments will also always be automatically converted.

    Parameters
    ----------
    *array_params : str
        Names of parameters to convert to cuPyNumeric ndarrays.
    name : str, optional
        Full profile label, e.g. ``"cupynumeric.random.Generator.beta"``.
        Overrides the auto-derived label entirely. Use when the function's
        ``__qualname__`` doesn't capture the user-facing path (e.g. methods
        defined in a private module ``_foo.py`` but exposed as
        ``cupynumeric.foo.Bar.method``).
    prefix : str, optional
        Segment inserted between ``cupynumeric.`` and ``__qualname__`` in
        the auto-derived label. For ``Generator.beta`` defined in
        ``cupynumeric/random/_generator.py``, ``prefix="random"`` produces
        ``cupynumeric.random.Generator.beta``. Ignored if ``name`` is set.
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
            args = _convert_args(args, indices, out_index)
            kwargs = _convert_kwargs(kwargs, to_convert)

            if settings.doctor():
                doctor.diagnose(func.__name__, args, kwargs)

            return func(*args, **kwargs)

        # Build the profile label. __qualname__ gives "Class.method" for
        # methods and just the function name for free functions — that's
        # usually what we want. `prefix` is for cases where the module path
        # carries meaningful context not visible in __qualname__ (private
        # module name vs. public namespace). `name` is a full override.
        qualname = getattr(
            func, "__qualname__", getattr(func, "__name__", "unknown")
        )
        if name is not None:
            label = name
        elif prefix is not None:
            label = f"cupynumeric.{prefix}.{qualname}"
        else:
            label = f"cupynumeric.{qualname}"
        profiled_wrapper = profiling_wrapper(wrapper, label)

        return profiled_wrapper

    return decorator


def broadcast_where(where: ndarray | None, shape: NdShape) -> ndarray | None:
    if where is not None and where.shape != shape:
        from .._module import broadcast_to

        where = broadcast_to(where, shape)
    return where


def convert_to_cupynumeric_ndarray(obj: Any, share: bool = False) -> ndarray:
    from .array import ndarray
    from .._thunk.deferred import DeferredArray

    # If this is an instance of one of our ndarrays then we're done
    if isinstance(obj, ndarray):
        return obj
    if isinstance(obj, DeferredArray):
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
    from .array import ndarray

    if isinstance(obj, ndarray):
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
