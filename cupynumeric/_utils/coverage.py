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

import warnings
from dataclasses import dataclass
from enum import Enum
import numpy
from contextlib import contextmanager
from contextvars import ContextVar
from functools import WRAPPER_ASSIGNMENTS, cache, wraps
from types import BuiltinFunctionType, ModuleType
from typing import (
    Any,
    Callable,
    Container,
    Iterable,
    Mapping,
    Protocol,
    TypeVar,
    cast,
    Iterator,
)

from legate.core import track_provenance, get_legate_runtime
from legate.core.utils import OrderedSet

from ..runtime import runtime
from ..settings import settings
from .stack import (
    find_last_user_line_numbers,
    find_last_user_stacklevel_and_frame,
)
from .structure import deep_apply

__all__ = ("GPUSupport", "clone_module", "clone_class")

FALLBACK_WARNING = (
    "cuPyNumeric has not implemented {what} "
    "and is falling back to canonical NumPy. "
    "You may notice significantly decreased performance "
    "for this function call."
)

MOD_INTERNAL = {"__dir__", "__getattr__"}

UFUNC_METHODS = ("at", "accumulate", "outer", "reduce", "reduceat")

upr_result: ContextVar[object] = ContextVar("upr_result", default=None)


def _infer_mode(result: Any) -> str:
    if result is None:
        return "N/A"
    elif hasattr(result, "_thunk"):
        is_deferred = runtime.is_deferred_array(result._thunk)
        return "deferred" if is_deferred else "eager"
    elif isinstance(result, numpy.ndarray):
        return "eager"
    else:
        return "N/A"


@contextmanager
def ProfileRange(name: str, location: str) -> Iterator[None]:
    get_legate_runtime().start_profiling_range()  # type: ignore
    token = upr_result.set(None)
    try:
        yield
    finally:
        result = upr_result.get()
        mode = _infer_mode(result)
        provenance = (
            f"{name.replace('numpy.', 'cupynumeric.')} "
            f"[mode: {mode}] on {location}"
        )
        upr_result.reset(token)
        get_legate_runtime().stop_profiling_range(provenance)  # type: ignore


@cache
def _profiling_enabled() -> bool:
    return get_legate_runtime().config().profile  # type: ignore


def issue_fallback_warning(what: str) -> None:
    stacklevel, frame = find_last_user_stacklevel_and_frame()
    msg = FALLBACK_WARNING.format(what=what)
    if settings.fallback_stacktrace():
        if frame:
            import traceback

            stacklist = traceback.extract_stack(frame)
            stack = "".join(traceback.format_list(stacklist))
            msg += f"\n\n{stack}"
        else:
            msg += " (could not extract stacktrace)"
    else:
        msg += (
            "\n\nSet CUPYNUMERIC_FALLBACK_STACKTRACE=1 and re-run to include "
            "a full stack trace with this warning."
        )
    warnings.warn(msg, stacklevel=stacklevel, category=RuntimeWarning)


def filter_namespace(
    ns: Mapping[str, Any],
    *,
    omit_names: Container[str] | None = None,
    omit_types: tuple[type, ...] = (),
) -> dict[str, Any]:
    omit_names = omit_names or OrderedSet()
    return {
        attr: value
        for attr, value in ns.items()
        if attr not in omit_names and not isinstance(value, omit_types)
    }


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class GPUSupport(Enum):
    YES = 1
    NO = 2
    PARTIAL = 3


def _scrape_docstring_support(doc: str) -> tuple[GPUSupport, GPUSupport]:
    multi = GPUSupport.NO
    if "Multiple GPUs (partial)" in doc:
        multi = GPUSupport.PARTIAL
    elif "Multiple GPUs" in doc:
        multi = GPUSupport.YES

    if "Single GPU" in doc:
        single = GPUSupport.YES
    else:
        single = multi

    return single, multi


@dataclass(frozen=True)
class CuWrapperMetadata:
    implemented: bool
    single: GPUSupport = GPUSupport.NO
    multi: GPUSupport = GPUSupport.NO


class CuWrapped(AnyCallable, Protocol):
    _cupynumeric_metadata: CuWrapperMetadata
    __wrapped__: AnyCallable
    __name__: str
    __qualname__: str


# This is in order to have less generically named wrapper functions in
# profiles. This approach "was used in NumPy and Guido approved".
def _fixup_co_name(
    func: Callable[[Any], Any], kind: str
) -> Callable[[Any], Any]:
    def decorator(wrapper: Callable[[Any], Any]) -> Callable[[Any], Any]:
        if hasattr(func, "__name__"):
            wrapper.__code__ = wrapper.__code__.replace(
                co_name=f"{func.__name__}_{kind}_wrapper",
                co_filename="<cupynumeric internals>",  # hide from TB
            )
        return wrapper

    return decorator


def implemented(func: AnyCallable, prefix: str, name: str) -> CuWrapped:
    name = f"{prefix}.{name}"
    reporting = settings.report_coverage()
    profiling = _profiling_enabled()
    full_bt = reporting and settings.report_dump_callstack()

    wrapper: CuWrapped

    @wraps(func)
    @track_provenance()
    @_fixup_co_name(func, "implemented")
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        if not reporting and not profiling:
            return func(*args, **kwargs)

        location = find_last_user_line_numbers(top_only=(not full_bt))

        def run_func(args: Any, kwargs: Any) -> Any:
            if reporting:
                runtime.record_api_call(
                    name=name, location=location, implemented=True
                )
            return func(*args, **kwargs)

        if not profiling:
            return run_func(args, kwargs)

        with ProfileRange(name, location):
            result = run_func(args, kwargs)
            upr_result.set(result)
            return result

    wrapper = cast(CuWrapped, _wrapper)

    # This is incredibly ugly and unpleasant, but @wraps(func) doesn't handle
    # ufuncs the way we need it to. The alternative would be to vendor and
    # modify a custom version of @wraps
    if hasattr(wrapper.__wrapped__, "_name"):
        wrapper.__name__ = wrapper.__wrapped__._name
        wrapper.__qualname__ = wrapper.__wrapped__._name

    # TODO (bev) Scraping text to set flags seems a bit fragile. It would be
    # preferable to start with flags, and use those to update docstrings.
    doc = getattr(func, "__doc__", None) or ""
    single, multi = _scrape_docstring_support(doc)

    wrapper._cupynumeric_metadata = CuWrapperMetadata(
        implemented=True, single=single, multi=multi
    )

    return wrapper


_UNIMPLEMENTED_COPIED_ATTRS = tuple(
    attr for attr in WRAPPER_ASSIGNMENTS if attr != "__doc__"
)


def unimplemented(
    func: AnyCallable,
    prefix: str,
    name: str,
    fallback: Callable[[Any], Any] | None = None,
) -> CuWrapped:
    name = f"{prefix}.{name}"
    reporting = settings.report_coverage()
    profiling = _profiling_enabled()
    full_bt = reporting and settings.report_dump_callstack()

    # Previously we were depending on NumPy functions to automatically convert
    # all array-like arguments to `numpy.ndarray` through `__array__()` (taking
    # some care to skip the `__array_function__` dispatch logic, to avoid
    # infinite loops). However, it appears that this behavior is inconsistent
    # in NumPy, so we will instead convert any `cupynumeric.ndarray`s manually
    # before calling into NumPy.

    wrapper: CuWrapped

    @wraps(func, assigned=_UNIMPLEMENTED_COPIED_ATTRS)
    @_fixup_co_name(func, "unimplemented")
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        if reporting or profiling:
            location = find_last_user_line_numbers(top_only=(not full_bt))

        def run_func(args: Any, kwargs: Any) -> Any:
            if reporting:
                runtime.record_api_call(
                    name=name, location=location, implemented=False
                )
            else:
                issue_fallback_warning(what=name)
            if fallback:
                args = deep_apply(args, fallback)
                kwargs = deep_apply(kwargs, fallback)
                result = func(*args, **kwargs)
                if isinstance(result, numpy.ndarray):
                    from .._array.util import convert_to_cupynumeric_ndarray

                    return convert_to_cupynumeric_ndarray(result)
                return result
            else:
                return func(*args, **kwargs)

        if not profiling:
            return run_func(args, kwargs)

        with ProfileRange(name, location):
            result = run_func(args, kwargs)
            upr_result.set(result)
            return result

    wrapper = cast(CuWrapped, _wrapper)

    wrapper.__doc__ = f"""
    cuPyNumeric has not implemented this function, and will fall back to NumPy.

    See Also
    --------
    {name}
    """
    wrapper._cupynumeric_metadata = CuWrapperMetadata(implemented=False)

    return wrapper


def clone_module(
    origin_module: ModuleType,
    new_globals: dict[str, Any],
    fallback: Callable[[Any], Any] | None = None,
    include_builtin_function_type: bool = False,
) -> None:
    """Copy attributes from one module to another, excluding submodules

    Function types are wrapped with a decorator to report API calls. All
    other values are copied as-is.

    Parameters
    ----------
    origin_module : ModuleTpe
        Existing module to clone attributes from

    new_globals : dict
        A globals() dict for the new module to clone into

    fallback :Callable[[Any], Any] | None
        A function that will be applied to each argument before calling into
        the original module, to handle unimplemented functions. The function
        will be called recursively on list/tuple/dict containers, and should
        convert objects of custom types into objects that the corresponding API
        on the original module can handle. Anything else should be passed
        through unchanged.

    include_builtin_function_type: bool
        Whether to wrap the "builtin" (C-extension) functions declared in the
        wrapped module

    Returns
    -------
    None

    """
    mod_name = origin_module.__name__

    missing = filter_namespace(
        origin_module.__dict__,
        omit_names=set(new_globals).union(MOD_INTERNAL),
        omit_types=(ModuleType,),
    )

    from .._ufunc.ufunc import ufunc as lgufunc

    for attr, value in new_globals.items():
        # Only need to wrap things that are in the origin module to begin with
        if attr not in origin_module.__dict__:
            continue
        if should_wrap(value) or (
            include_builtin_function_type
            and isinstance(value, BuiltinFunctionType)
        ):
            wrapped = implemented(cast(AnyCallable, value), mod_name, attr)
            new_globals[attr] = wrapped
            if isinstance(value, lgufunc):
                for method in UFUNC_METHODS:
                    wrapped_method = (
                        implemented(
                            getattr(value, method),
                            f"{mod_name}.{attr}",
                            method,
                        )
                        if hasattr(value, method)
                        else unimplemented(
                            getattr(getattr(origin_module, attr), method),
                            f"{mod_name}.{attr}",
                            method,
                            fallback=fallback,
                        )
                    )
                    setattr(wrapped, method, wrapped_method)

    from numpy import ufunc as npufunc

    for attr, value in missing.items():
        if should_wrap(value) or (
            include_builtin_function_type
            and isinstance(value, BuiltinFunctionType)
        ):
            wrapped = unimplemented(value, mod_name, attr, fallback=fallback)
            new_globals[attr] = wrapped
            if isinstance(value, npufunc):
                for method in UFUNC_METHODS:
                    wrapped_method = unimplemented(
                        getattr(value, method),
                        f"{mod_name}.{attr}",
                        method,
                        fallback=fallback,
                    )
                    setattr(wrapped, method, wrapped_method)
        else:
            new_globals[attr] = value


def should_wrap(obj: object) -> bool:
    from numpy import ufunc as npufunc

    from .._ufunc.ufunc import ufunc as lgufunc

    # Custom callables, e.g. cython functions used in np2, do not inherit
    # anything, so we check callable() instead (and include the __get__/__set__
    # checks to filter out classes). OTOH ufuncs need to be checked specially
    # because they do not have __get__.
    return (
        callable(obj)
        and hasattr(obj, "__get__")
        and not hasattr(obj, "__set__")
    ) or isinstance(obj, (lgufunc, npufunc))


T = TypeVar("T")


def clone_class(
    origin_class: type,
    omit_names: Iterable[str] | None = None,
    fallback: Callable[[Any], Any] | None = None,
) -> Callable[[T], T]:
    """Copy attributes from one class to another

    Method types are wrapped with a decorator to report API calls. All
    other values are copied as-is.

    """

    class_name = f"{origin_class.__module__}.{origin_class.__name__}"

    clean_omit_names = OrderedSet() if omit_names is None else omit_names

    def _clone_class(cls: T) -> T:
        missing = filter_namespace(
            origin_class.__dict__,
            omit_names=set(cls.__dict__).union(clean_omit_names),
        )

        for attr, value in cls.__dict__.items():
            # Only need to wrap things that are also in the origin class
            if not hasattr(origin_class, attr):
                continue
            if should_wrap(value):
                wrapped = implemented(value, class_name, attr)
                setattr(cls, attr, wrapped)

        for attr, value in missing.items():
            if should_wrap(value):
                wrapped = unimplemented(
                    value, class_name, attr, fallback=fallback
                )
                setattr(cls, attr, wrapped)
            else:
                setattr(cls, attr, value)

        return cls

    return _clone_class


def is_wrapped(obj: Any) -> bool:
    return hasattr(obj, "_cupynumeric_metadata")


def is_implemented(obj: Any) -> bool:
    return is_wrapped(obj) and obj._cupynumeric_metadata.implemented


def is_single(obj: Any) -> GPUSupport:
    if not is_wrapped(obj):
        return GPUSupport.NO
    return cast(CuWrapped, obj)._cupynumeric_metadata.single


def is_multi(obj: Any) -> GPUSupport:
    if not is_wrapped(obj):
        return GPUSupport.NO
    return cast(CuWrapped, obj)._cupynumeric_metadata.multi
