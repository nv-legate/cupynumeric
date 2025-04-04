# Copyright 2022 NVIDIA Corporation
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

from dataclasses import dataclass
from functools import WRAPPER_ASSIGNMENTS, wraps
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    MethodType,
    ModuleType,
)
from typing import (
    Any,
    Callable,
    Container,
    Iterable,
    Mapping,
    Optional,
    Union,
    cast,
)

from legate.core import track_provenance
from legate.core.utils import OrderedSet
from typing_extensions import Protocol

from .runtime import runtime
from .settings import settings
from .utils import deep_apply, find_last_user_frames

__all__ = ("clone_module", "clone_class")

FALLBACK_WARNING = (
    "cuNumeric has not implemented {what} "
    + "and is falling back to canonical NumPy. "
    + "You may notice significantly decreased performance "
    + "for this function call."
)

MOD_INTERNAL = {"__dir__", "__getattr__"}

UFUNC_METHODS = ("at", "accumulate", "outer", "reduce", "reduceat")


def filter_namespace(
    ns: Mapping[str, Any],
    *,
    omit_names: Optional[Container[str]] = None,
    omit_types: tuple[type, ...] = (),
) -> dict[str, Any]:
    omit_names = omit_names or OrderedSet()
    return {
        attr: value
        for attr, value in ns.items()
        if attr not in omit_names and not isinstance(value, omit_types)
    }


class AnyCallable(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class CuWrapperMetadata:
    implemented: bool
    single: bool = False
    multi: bool = False


class CuWrapped(AnyCallable, Protocol):
    _cunumeric: CuWrapperMetadata
    __wrapped__: AnyCallable
    __name__: str
    __qualname__: str


def implemented(
    func: AnyCallable, prefix: str, name: str, reporting: bool = True
) -> CuWrapped:
    name = f"{prefix}.{name}"

    wrapper: CuWrapped

    if reporting:

        @wraps(func)
        @track_provenance()
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            location = find_last_user_frames(
                not settings.report_dump_callstack()
            )
            runtime.record_api_call(
                name=name,
                location=location,
                implemented=True,
            )
            return func(*args, **kwargs)

    else:

        @wraps(func)
        @track_provenance()
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

    # This is incredibly ugly and unpleasant, but @wraps(func) doesn't handle
    # ufuncs the way we need it to. The alternative would be to vendor and
    # modify a custom version of @wraps
    if hasattr(wrapper.__wrapped__, "_name"):
        wrapper.__name__ = wrapper.__wrapped__._name
        wrapper.__qualname__ = wrapper.__wrapped__._name

    # TODO (bev) Scraping text to set flags seems a bit fragile. It would be
    # preferable to start with flags, and use those to update docstrings.
    multi = "Multiple GPUs" in (getattr(func, "__doc__", None) or "")
    single = "Single GPU" in (getattr(func, "__doc__", None) or "") or multi

    wrapper._cunumeric = CuWrapperMetadata(
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
    reporting: bool = True,
    fallback: Union[Callable[[Any], Any], None] = None,
) -> CuWrapped:
    name = f"{prefix}.{name}"

    # Previously we were depending on NumPy functions to automatically convert
    # all array-like arguments to `numpy.ndarray` through `__array__()` (taking
    # some care to skip the `__array_function__` dispatch logic, to avoid
    # infinite loops). However, it appears that this behavior is inconsistent
    # in NumPy, so we will instead convert any `cunumeric.ndarray`s manually
    # before calling into NumPy.

    wrapper: CuWrapped

    if reporting:

        @wraps(func, assigned=_UNIMPLEMENTED_COPIED_ATTRS)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            location = find_last_user_frames(
                not settings.report_dump_callstack()
            )
            runtime.record_api_call(
                name=name,
                location=location,
                implemented=False,
            )
            if fallback:
                args = deep_apply(args, fallback)
                kwargs = deep_apply(kwargs, fallback)
            return func(*args, **kwargs)

    else:

        @wraps(func, assigned=_UNIMPLEMENTED_COPIED_ATTRS)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            runtime.warn(
                FALLBACK_WARNING.format(what=name),
                category=RuntimeWarning,
            )
            if fallback:
                args = deep_apply(args, fallback)
                kwargs = deep_apply(kwargs, fallback)
            return func(*args, **kwargs)

    wrapper.__doc__ = f"""
    cuNumeric has not implemented this function, and will fall back to NumPy.

    See Also
    --------
    {name}
    """
    wrapper._cunumeric = CuWrapperMetadata(implemented=False)

    return wrapper


def clone_module(
    origin_module: ModuleType,
    new_globals: dict[str, Any],
    fallback: Union[Callable[[Any], Any], None] = None,
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

    fallback : Union[Callable[[Any], Any], None]
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

    reporting = settings.report_coverage()

    from ._ufunc.ufunc import ufunc as lgufunc

    for attr, value in new_globals.items():
        # Only need to wrap things that are in the origin module to begin with
        if attr not in origin_module.__dict__:
            continue
        if isinstance(value, (FunctionType, lgufunc)) or (
            include_builtin_function_type
            and isinstance(value, BuiltinFunctionType)
        ):
            wrapped = implemented(
                cast(AnyCallable, value), mod_name, attr, reporting=reporting
            )
            new_globals[attr] = wrapped
            if isinstance(value, lgufunc):
                for method in UFUNC_METHODS:
                    wrapped_method = (
                        implemented(
                            getattr(value, method),
                            f"{mod_name}.{attr}",
                            method,
                            reporting=reporting,
                        )
                        if hasattr(value, method)
                        else unimplemented(
                            getattr(getattr(origin_module, attr), method),
                            f"{mod_name}.{attr}",
                            method,
                            reporting=reporting,
                            fallback=fallback,
                        )
                    )
                    setattr(wrapped, method, wrapped_method)

    from numpy import ufunc as npufunc

    for attr, value in missing.items():
        if isinstance(value, (FunctionType, npufunc)) or (
            include_builtin_function_type
            and isinstance(value, BuiltinFunctionType)
        ):
            wrapped = unimplemented(
                value,
                mod_name,
                attr,
                reporting=reporting,
                fallback=fallback,
            )
            new_globals[attr] = wrapped
            if isinstance(value, npufunc):
                for method in UFUNC_METHODS:
                    wrapped_method = unimplemented(
                        getattr(value, method),
                        f"{mod_name}.{attr}",
                        method,
                        reporting=reporting,
                        fallback=fallback,
                    )
                    setattr(wrapped, method, wrapped_method)
        else:
            new_globals[attr] = value


def should_wrap(obj: object) -> bool:
    return isinstance(obj, (FunctionType, MethodType, MethodDescriptorType))


def clone_class(
    origin_class: type,
    omit_names: Union[Iterable[str], None] = None,
    fallback: Union[Callable[[Any], Any], None] = None,
) -> Callable[[type], type]:
    """Copy attributes from one class to another

    Method types are wrapped with a decorator to report API calls. All
    other values are copied as-is.

    """

    class_name = f"{origin_class.__module__}.{origin_class.__name__}"
    clean_omit_names = OrderedSet() if omit_names is None else omit_names

    def _clone_class(cls: type) -> type:
        missing = filter_namespace(
            origin_class.__dict__,
            omit_names=set(cls.__dict__).union(clean_omit_names),
        )

        reporting = settings.report_coverage()

        for attr, value in cls.__dict__.items():
            # Only need to wrap things that are also in the origin class
            if not hasattr(origin_class, attr):
                continue
            if should_wrap(value):
                wrapped = implemented(
                    value,
                    class_name,
                    attr,
                    reporting=reporting,
                )
                setattr(cls, attr, wrapped)

        for attr, value in missing.items():
            if should_wrap(value):
                wrapped = unimplemented(
                    value,
                    class_name,
                    attr,
                    reporting=reporting,
                    fallback=fallback,
                )
                setattr(cls, attr, wrapped)
            else:
                setattr(cls, attr, value)

        return cls

    return _clone_class


def is_implemented(obj: Any) -> bool:
    return hasattr(obj, "_cunumeric") and obj._cunumeric.implemented


def is_single(obj: Any) -> bool:
    return hasattr(obj, "_cunumeric") and obj._cunumeric.single


def is_multi(obj: Any) -> bool:
    return hasattr(obj, "_cunumeric") and obj._cunumeric.multi
