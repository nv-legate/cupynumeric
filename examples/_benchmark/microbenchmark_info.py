# Copyright 2026 NVIDIA Corporation
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

from __future__ import annotations

import inspect
import math

from dataclasses import dataclass, asdict, replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeVar

import numpy as np

from .harness import BenchmarkHarness
from .info import INFO, BenchmarkInfo, create_benchmark_info

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def _ndarray_bytes(
    in_shape: int | tuple[int, ...], dtype: DTypeLike = "float64"
) -> int:
    shape = (in_shape,) if isinstance(in_shape, int) else in_shape
    return int(math.prod(shape) * np.dtype(dtype).itemsize)


def _explain_ndarray_bytes(
    name: str, in_shape: int | tuple[int, ...], in_dtype: DTypeLike = "float64"
) -> str:
    shape = (in_shape,) if isinstance(in_shape, int) else in_shape
    dtype = np.dtype(in_dtype)
    value = int(np.prod(shape) * dtype.itemsize)
    if isinstance(in_shape, int):
        return (
            f"{name}: {in_shape:,} {dtype.name}s"
            f" x {dtype.itemsize} bytes / {dtype.name} = {value:,} bytes"
        )
    else:
        shape_str = tuple(f"{i:,}" for i in shape)
        return (
            f"{name}: ({' x '.join(shape_str)} {dtype.name}s)"
            f" x {dtype.itemsize} bytes / {dtype.name} = {value:,} bytes"
        )


_T = TypeVar("_T")


class _SIZE_TYPE:
    def __repr__(self) -> str:
        return "SIZE"


# A sentinel indicating that an argument should receive the size directly
SIZE = _SIZE_TYPE()


class _MISSING_TYPE:
    def __repr__(self) -> str:
        return "MISSING"


# A sentinel indicating that an argument's value has not been specified
MISSING = _MISSING_TYPE()

Plan = (
    dict[str, Any]
    | Iterable[dict[str, Any]]
    | Callable[[BenchmarkHarness], Iterable[dict[str, Any]]]
    | None
)

# An array description should be either (name, shape) or (name, shape, dtype):
# if dtype is ommited, it is assumed to be "float64".  Shape can be an int
# or a tuple of ints
ArrayDescription = (
    tuple[str, int | tuple[int, ...]] | tuple[str, int | tuple[int, ...], str]
)


@dataclass
class MicrobenchmarkInfo(BenchmarkInfo):
    """Information about a Callable for :py:meth:`MicrobenchmarkSuite.run_suite`."""

    size_to_args: Callable[..., dict[str, Any]] | None
    args_to_bytes: Callable[..., int] | None
    explain_bytes: Callable[..., list[str]] | None
    args_to_arrays: Callable[..., list[ArrayDescription]] | None
    args_to_work: Callable[..., int | float] | None
    explain_work: Callable[..., list[str]] | None
    plan: Plan
    skip: bool | Callable[..., bool]

    def replace(self, /, **changes: Any) -> MicrobenchmarkInfo:
        return replace(self, **changes)

    def complete_args_from_size(
        self, size: int, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Fill in argument values that depend on the ``size`` of the problem."""
        if not any(v in [SIZE, MISSING] for v in args.values()):
            return args
        out_args = args.copy()
        needs_size_to_args: list[str] = []
        for key, v in args.items():
            if v is SIZE:
                # args labeled SIZE receive size directly.
                out_args[key] = size
            elif v is MISSING:
                needs_size_to_args.append(key)
        if not needs_size_to_args:
            return out_args
        if self.size_to_args is None:
            msg = (
                f"Arguments {needs_size_to_args} must be computed from "
                "the size, but `size_to_args` was not passed to "
                "@microbenchmark()"
            )
            raise RuntimeError(msg)
        assert callable(self.size_to_args)

        # determine which arguments to the original function are also arguments
        # that size_to_args needs to compute missing values by inspecting
        # parameter names (names must correspond to the parameter names in the
        # original function)
        sig = inspect.signature(self.size_to_args)
        relevant_args: list[Any] = []
        relevant_kwargs: dict[str, Any] = {}
        for p, v in sig.parameters.items():
            value: Any
            if p == "size":
                # "size" is special: it can be included even if it is not an
                # argument of the original function
                value = size
            else:
                if p not in out_args:
                    msg = (
                        f"`size_to_args` argument '{p}' does not have the "
                        "same name as a microbenchmark argument"
                    )
                    raise RuntimeError(msg)
                value = out_args[p]
                if value is MISSING:
                    msg = (
                        f"`size_to_args` cannot have '{p}' as an argument "
                        "if it is a `MISSING` parameter"
                    )
                    raise RuntimeError(msg)
            if v.kind in [v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD]:
                relevant_args.append(value)
            else:
                relevant_kwargs[p] = value
        computed_args = self.size_to_args(*relevant_args, **relevant_kwargs)
        for arg in needs_size_to_args:
            if arg not in computed_args:
                msg = f"`size_to_args` did not computed a value for '{arg}'"
                raise RuntimeError(msg)
            out_args[arg] = computed_args[arg]
        return out_args

    def _get_args(
        self, attr: str, args: dict[str, Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        """Gather arguments to a callable based on known values."""
        method = getattr(self, attr)
        sig = inspect.signature(method)
        relevant_args: list[Any] = []
        relevant_kwargs: dict[str, Any] = {}
        for p, v in sig.parameters.items():
            if p not in args:
                msg = (
                    f"`{attr}` argument '{p}' does not have the "
                    "same name as a microbenchmark argument"
                )
                raise RuntimeError(msg)
            value = args[p]
            if v.kind in [v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD]:
                relevant_args.append(value)
            else:
                relevant_kwargs[p] = value
        return (relevant_args, relevant_kwargs)

    def get_bytes(self, size: int, args: dict[str, Any]) -> int:
        """Estimate the working set size from ``size`` and known arguments."""
        if self.args_to_bytes is not None:
            complete_args = self.complete_args_from_size(size, args)
            sub_args, sub_kwargs = self._get_args(
                "args_to_bytes", complete_args
            )
            return self.args_to_bytes(*sub_args, **sub_kwargs)
        if self.args_to_arrays is not None:
            complete_args = self.complete_args_from_size(size, args)
            sub_args, sub_kwargs = self._get_args(
                "args_to_arrays", complete_args
            )
            descriptions = self.args_to_arrays(*sub_args, **sub_kwargs)
            return sum(_ndarray_bytes(*v[1:]) for v in descriptions)
        msg = "Neither `args_to_bytes` nor `args_to_arrays` was specified"
        raise RuntimeError(msg)

    def get_work(self, size: int, args: dict[str, Any]) -> float:
        """Estimate the work from ``size`` and known arguments."""
        if self.args_to_work is not None:
            complete_args = self.complete_args_from_size(size, args)
            sub_args, sub_kwargs = self._get_args(
                "args_to_work", complete_args
            )
            return float(self.args_to_work(*sub_args, **sub_kwargs))
        # if `args_to_work` is not specified, assume work is proportional
        # to bytes
        return float(self.get_bytes(size, args))

    def get_plan(
        self, suite: BenchmarkHarness, args: dict[str, Any]
    ) -> Iterable[dict[str, Any]]:
        """Get or compute a plan of arguments to a benchmark function."""
        if self.plan is None:
            return [args]
        if isinstance(self.plan, dict):
            return [self.plan]
        if callable(self.plan):
            return self.plan(suite)
        return self.plan

    def should_skip(
        self, suite: BenchmarkHarness, args: dict[str, Any]
    ) -> bool:
        """Check of a particular call to a function should be skipped."""
        if isinstance(self.skip, bool):
            return self.skip
        sub_args, sub_kwargs = self._get_args("skip", {**args, "suite": suite})
        return self.skip(*sub_args, **sub_kwargs)

    def get_explain_bytes(self, size: int, args: dict[str, Any]) -> list[str]:
        """Get an optional description of how ``get_bytes()`` is computed."""
        if self.explain_bytes is not None:
            sub_args, sub_kwargs = self._get_args("explain_bytes", args)
            return self.explain_bytes(*sub_args, **sub_kwargs)
        if self.args_to_arrays is not None:
            complete_args = self.complete_args_from_size(size, args)
            sub_args, sub_kwargs = self._get_args(
                "args_to_arrays", complete_args
            )
            descriptions = self.args_to_arrays(*sub_args, **sub_kwargs)
            return [_explain_ndarray_bytes(*v) for v in descriptions]
        return []

    def get_explain_work(self, args: dict[str, Any]) -> list[str]:
        """Get an optional description of how ``get_work()`` is computed."""
        if self.explain_work is None:
            return []
        sub_args, sub_kwargs = self._get_args("explain_work", args)
        return self.explain_work(*sub_args, **sub_kwargs)

    def format_search_string(
        self,
        bmark: Callable[..., Any],
        pos_args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Create a string like func(a1=v1,a2=v2) for regex searches."""
        in_order_args: list[str] = []
        sig = inspect.signature(bmark)
        for i, (k, v) in enumerate(sig.parameters.items()):
            value: Any = MISSING
            if v.default != v.empty:
                value = v.default
            if v.kind in [
                v.POSITIONAL_ONLY,
                v.POSITIONAL_OR_KEYWORD,
            ] and i < len(pos_args):
                value = pos_args[i]
            if k in kwargs:
                value = kwargs[k]
            format_name = k
            if k in self.input_names:
                format_name = self.input_names[k]
            if format_name in self.formats:
                value = self.formats[format_name](value)
            in_order_args.append(f"{k}={value}")
        return f"{self.name}({','.join(in_order_args)})"

    def pretty_args(self, arg_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply supplied formatting for easier to read argument strings."""
        out_dict: dict[str, Any] = {}
        for k, v in arg_dict.items():
            name = self.input_names.get(k, k)
            if v in [MISSING, SIZE]:
                out_dict[k] = v
            elif name in self.formats:
                out_dict[k] = self.formats[name](v)
            else:
                out_dict[k] = v
        return out_dict


def _create_microbenchmark_info(
    f: Callable[..., Any],
    name: str | None = None,
    input_names: dict[str, str] | None = None,
    output_names: str | tuple[str, ...] | None = None,
    formats: dict[str, Callable[..., str]] | None = None,
    returns_time: int = 0,
    size_to_args: Callable[..., dict[str, Any]] | None = None,
    args_to_bytes: Callable[..., int] | None = None,
    explain_bytes: Callable[..., list[str]] | None = None,
    args_to_arrays: Callable[..., list[ArrayDescription]] | None = None,
    args_to_work: Callable[..., int | float] | None = None,
    explain_work: Callable[..., list[str]] | None = None,
    plan: Plan = None,
    skip: bool | Callable[..., bool] = False,
) -> MicrobenchmarkInfo:
    # change time default to "time per run (ms)"
    time_string = "time per run (ms)"
    if input_names is None:
        input_names = {}
    if formats is None:
        formats = {}
    if returns_time >= 0:
        if isinstance(output_names, str) or output_names is None:
            assert returns_time == 0
            output_names = time_string
        else:
            new_output_names = [name for name in output_names]
            new_output_names[returns_time] = time_string
            output_names = tuple(new_output_names)
    info = create_benchmark_info(
        f, name, input_names, output_names, formats, returns_time
    )
    micro_info = MicrobenchmarkInfo(
        **asdict(info),
        size_to_args=size_to_args,
        args_to_bytes=args_to_bytes,
        explain_bytes=explain_bytes,
        args_to_arrays=args_to_arrays,
        args_to_work=args_to_work,
        explain_work=explain_work,
        plan=plan,
        skip=skip,
    )
    return micro_info


def microbenchmark(
    *,
    name: str | None = None,
    input_names: dict[str, str] | None = None,
    output_names: str | tuple[str, ...] | None = None,
    formats: dict[str, Callable[..., str]] | None = None,
    returns_time: bool | int = True,
    size_to_args: Callable[..., dict[str, Any]] | None = None,
    args_to_bytes: Callable[..., int] | None = None,
    explain_bytes: Callable[..., list[str]] | None = None,
    args_to_arrays: Callable[..., list[ArrayDescription]] | None = None,
    args_to_work: Callable[..., int | float] | None = None,
    explain_work: Callable[..., list[str]] | None = None,
    plan: Plan = None,
    skip: bool | Callable[..., bool] = False,
) -> Callable[..., Any]:
    """Decorator for a microbenchmark that can be run by a ``MicrobenchmarkSuite``.

    See :py:func:`benchmark_info` for details on shared parameters.
    ``microbenchmark_info()`` has additional arguments that help
    a ``MicrobenchmarkSuite`` generate calls to a microbenchmark from
    :py:class:`SizeRequest`.

    Parameters
    ----------
    size_to_args: Callable[..., dict[str, Any]] | None = None
        If not ``None``, a function that computes values for microbenchmark
        arguments that should be computed from the size.  The names
        of the arguments of the callable are used to determine how it is called:
        these can be ``size`` or any argument to the microbenchmark that does
        not depend on ``size``.  This is only used if the microbenchmark has
        arguments that are marked ``MISSING``.
    args_to_bytes: Callable[..., int] | None = None
        If not ``None``, a function that computes the working set size of a
        microbenchmark.  The names of the arguments of the callable determine
        how it is called: it will be called with values from the matching
        arguments of the microbenchmark.  If ``None``, the working set size
        will be determined directly from ``args_to_arrays``.
    explain_bytes: Callable[..., int] | None = None
        Provide optional strings explaining how the value in `args_to_bytes`
        is computed.
    args_to_arrays: Callable[..., list[ArrayDescription]]
        If not ``None``, a callable that returns a list of array descriptions
        (tuples of the form (name, shape) or (name, shape, dtype)).  Those
        array descriptions will be used to compute ``args_to_bytes`` and
        ``explain_bytes``.
    args_to_work: Callable[..., int | float] | None = None
        If not ``None``, a function that estimates the work performed by a
        microbenchmark.  The names of the arguments of the callable determine
        how it is called: it will be called with values from the matching
        arguments of the microbenchmark.  If ``None``, the work is assumed
        to be proportional to the working set size and will be determined
        from ``args_to_bytes`` or ``args_to_arrays``.
    explain_work: Callable[..., int] | None = None
        Provide optional strings explaining how the value in `args_to_work`
        is computed.
    plan: Plan = None
        A way to get all of the calls that ``run_suite`` should make to the
        microbenchmark.  Either a dictionary mapping argument names to values,
        a list of the same, or a function that generates a list of the same
        from the microbenchmark suite.  An argument can have the value
        ``SIZE``, in which case its value is taken directly from the size,
        or ``MISSING`` in which case its values will be calculated from the
        size (see ``args_to_bytes``).
    skip: bool | Callable[..., bool] = False,
        If ``True`` or a function that evaluates to ``True``, the
        microbenchmark will be skipped for certain arguments.  The names of the
        arguments of the callable are used to determine how it is called: these
        can be ``suite`` or any argument to the microbenchmark.
    """
    this_returns_time: int
    if isinstance(returns_time, bool):
        this_returns_time = 0 if returns_time else -1
    else:
        this_returns_time = returns_time

    def inner(f: Callable[..., _T]) -> Callable[..., _T]:
        info = _create_microbenchmark_info(
            f,
            name,
            input_names,
            output_names,
            formats,
            this_returns_time,
            size_to_args,
            args_to_bytes,
            explain_bytes,
            args_to_arrays,
            args_to_work,
            explain_work,
            plan,
            skip,
        )
        # we have to set the attribute before and after staticmethod(). If we
        # don't set it before, then decorator-style use of @microbenchmark
        # won't work; if we don't set if after, function style using
        # setattr(self, ..., microbenchmark()) won't work.
        setattr(f, INFO, info)
        f_static = staticmethod(f)
        setattr(f_static, INFO, info)
        return f_static

    return inner


def get_microbenchmark_info(f: Callable[..., Any]) -> MicrobenchmarkInfo:
    """Get the :py:class:`MicrobenchmarkInfo` of a Callable.

    Parameters
    ----------
    f: Callable[...,Any]
        A callable.  If it was decorated with :py:func:`microbenchmark`,
        that information will be returned; otherwise default information
        will be generated.

    Returns
    -------
    MicrobenchmarkInfo
    """
    if hasattr(f, INFO):
        v = getattr(f, INFO)
        assert isinstance(v, MicrobenchmarkInfo)
        return v
    return _create_microbenchmark_info(f)
