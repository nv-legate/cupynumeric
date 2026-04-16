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

import atexit
import builtins
import inspect
import io
import sys
import tokenize
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
from types import FrameType
from typing import Any, Final, Type

import numpy as np

from .._utils.stack import find_last_user_frame
from .._utils.array import is_true_unoptimized_advanced_indexing
from ..settings import settings


# Selected built-ins without a dedicated DUNDER (unlike len -> __len__).
# Using these on cuPyNumeric arrays is inefficient; use np.min(arr) / arr.min() etc.
# see entire list of functions: https://docs.python.org/3/library/functions.html
_DISCOURAGED_BUILTINS: Final[frozenset[str]] = frozenset(
    {
        "all",
        "any",
        "bool",
        "float",
        "int",
        "max",
        "min",
        "pow",
        "round",
        "sorted",
        "sum",
    }
)

_BUILTIN_ALTERNATIVES: Final[dict[str, str]] = {
    "all": "np.all(arr) or arr.all()",
    "any": "np.any(arr) or arr.any()",
    "bool": "arr.astype(bool)",
    "float": "arr.astype(float)",
    "int": "arr.astype(int)",
    "max": "np.max(arr) or arr.max()",
    "min": "np.min(arr) or arr.min()",
    "pow": "np.power(arr, n) or arr ** n",
    "round": "np.round(arr) or arr.round()",
    "sorted": "np.sort(arr)",
    "sum": "np.sum(arr) or arr.sum()",
}


def lookup_source(filename: str, lineno: int) -> str | None:
    """
    Attempt to lookup a line of source code from a given filename at a given
    line number.

    Args:
        filename (str):
            The name of the file to attempt to open
        lineno (int):
            The line number of the line in the file to return

    Returns:
        A stripped string containing the requested source code line, if it
        can be found, otherwise None.

    """
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            if 1 <= lineno <= len(lines):
                return lines[lineno - 1].strip()
    except Exception:
        return None
    return None


def is_scalar_key(key: Any, ndim: int) -> bool:
    """
    Whether the input is something like a key for accessing a "single" item
    from an array. i.e. a single scalar, or a tuple of all scalars.

    Args:
        key (Any):
            the key to check

        ndim (int):
            the number of expected dimensions for the key

    Returns
        True if the key is a scalar key, otherwise False

    """
    if np.isscalar(key) and ndim == 1:
        return True

    if (
        isinstance(key, tuple)
        and len(key) == ndim
        and all(np.isscalar(x) for x in key)
    ):
        return True

    return False


def is_slice_assignment_key(key: Any, ndim: int) -> bool:
    """
    Whether the key looks like "assign to a slice in one dimension while
    indexing by scalar(s) in others", e.g. ``x[i, :] = 2``, ``x[:, j] = 2``,
    or ``x[i, ...] = 2`` for a multi-dimensional array.

    Such a pattern repeated in a loop is inefficient and can be vectorized.

    Args:
        key (Any):
            the key passed to __setitem__
        ndim (int):
            the number of dimensions of the array

    Returns:
        True if the key is a tuple containing at least one slice or Ellipsis
        and at least one scalar index, otherwise False. When Ellipsis is
        present the key may be shorter than ndim (e.g. (i, Ellipsis) for 4D).
    """
    if not isinstance(key, tuple) or len(key) < 2:
        return False
    has_slice_or_ellipsis = any(
        isinstance(k, slice) or k is Ellipsis for k in key
    )
    has_scalar = any(np.isscalar(k) for k in key)
    if not (has_slice_or_ellipsis and has_scalar):
        return False
    # Without Ellipsis, key length must match ndim (e.g. (i, :, j, :))
    if Ellipsis not in key and len(key) != ndim:
        return False
    return True


@dataclass(frozen=True)
class CheckupLocator:
    filename: str
    lineno: int
    traceback: str


SOURCE_NOT_FOUND: Final = "(could not locate source code for line)"


@dataclass(frozen=True)
class Diagnostic(CheckupLocator):
    source: str | None
    description: str
    reference: str | None = None

    def __str__(self) -> str:
        msg = f"""\
- issue: {self.description}
  detected on: line {self.lineno} of file {self.filename!r}:\n\n"""
        if self.traceback and settings.doctor_traceback():
            msg += f"  FULL TRACEBACK:\n\n{self.traceback.rstrip()}"
        else:
            msg += f"    {self.source or SOURCE_NOT_FOUND}"
        if self.reference:
            msg += f"\n\n  refer to: {self.reference}"
        return msg


class Checkup(ABC):
    """
    Base class for cuPyNumeric Doctor checkups.

    Subclasses must implement the ``run`` method that returns a
    ``Diagnostic`` in case the checkup heuristic detected a warnable
    condition.
    """

    #: A brief description for the heuristic condition that that this
    #: checkup attempts to warn about. This description will be included
    #: in cuPyNumeric Doctor output
    description: str

    #: A reference (e.g. a URL) that elaborates on best pratices related
    #: to this checkup heuristic
    reference: str | None = None

    _locators: set[CheckupLocator]

    def __init__(self) -> None:
        self._locators = set()

        # demand that checkup subclasses provide this information
        assert self.description

    @abstractmethod
    def run(self, func: str, args: Any, kwargs: Any) -> Diagnostic | None:
        """
        Run a cuPyNumeric Doctor heuristic check.

        Args:
            name (str):
                Name of the function being invoked
            args (tuple):
                Any positional arguments the function is being called with
            kwargs (dict):
                Any keyword arguments the function is being called with

        Returns:
            a ``Diagnostic`` in case a new detection at the current location
            is reported, otherwise None

        """
        ...

    def report(self, locator: CheckupLocator) -> Diagnostic | None:
        """
        Report a heuristic detection.

        Args:
            locator (CheckupLocator):
                A source locator for the report

        Returns:
            Diagnostic, in case the report for this checkup is new for
            this location, otherwise None

        """
        if locator in self._locators:
            return None
        self._locators.add(locator)
        return self.info(locator)

    def locate(self) -> CheckupLocator | None:
        """
        Generate a ``CheckupLocator`` for the source location in the user's
        code that the checkup heuristic warned about.

        Returns:
            CheckupLocator | None

        """
        import inspect

        if (frame := find_last_user_frame()) is None:
            return None

        info = inspect.getframeinfo(frame)

        stack = traceback.extract_stack(frame)

        return CheckupLocator(
            filename=info.filename,
            lineno=info.lineno,
            traceback="".join(traceback.format_list(stack)),
        )

    def info(self, locator: CheckupLocator) -> Diagnostic:
        """
        Generate a full ``Diagnostic`` for a reported checkup location.

        Args:
            locator (CheckupLocator):
                location where a report for this checkup occurred

        Returns:
            Diagnostic

        """
        filename = locator.filename
        lineno = locator.lineno

        return Diagnostic(
            filename=filename,
            lineno=lineno,
            traceback=locator.traceback,
            source=lookup_source(filename, lineno),
            description=self.description,
            reference=self.reference,
        )


class RepeatedItemOps(Checkup):
    """
    Attempt to detect and warn about repeated scalar accesses to arrays on
    the same line.

    """

    ITEMOP_THRESHOLD: int = 10

    description = "multiple scalar item accesses repeated on the same line"
    reference = "https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-array-based-operations-avoid-loops-with-indexing"  # noqa

    def __init__(self) -> None:
        super().__init__()
        self._itemop_counts: dict[int, int] = defaultdict(int)

    def run(self, func: str, args: Any, _kwargs: Any) -> Diagnostic | None:
        """
        Check for repeated scalar accesses to arrays.

        Args:
            func (str):
                Name of the function being invoked
            args (tuple):
                Any positional arguments the function is being called with
            kwargs (dict):
                Any keyword arguments the function is being called with

        Returns:
            a ``Diagnostic`` in case a new detection at the current location
            is reported, otherwise None

        """
        if func in {"__setitem__", "__getitem__"}:
            ndim: int = args[0].ndim
            if is_scalar_key(args[1], ndim):
                # if we can't find a user frame, then it is probably due to a
                # detection inside cupynumeric itself. Either way, there is no
                # actionable information to provide users, so just punt here.
                if (locator := self.locate()) is None:
                    return None

                self._itemop_counts[locator.lineno] += 1
                if self._itemop_counts[locator.lineno] > self.ITEMOP_THRESHOLD:
                    return self.report(locator)

        return None


class RepeatedSliceAccessCheck(Checkup):
    """
    Attempt to detect and warn about repeated slice access (set/get) in a loop
    that could be vectorized (e.g. ``for i in range(n): x[i, :] = 2``).
    """

    SLICE_ASSIGN_THRESHOLD: int = 10

    description = (
        "repeated slice access on the same line (e.g. x[i,:] = ... in a "
        "loop); consider vectorizing instead"
    )
    reference = "https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-array-based-operations-avoid-loops-with-indexing"  # noqa

    def __init__(self) -> None:
        super().__init__()
        self._counts: dict[tuple[str, int], int] = defaultdict(int)

    def run(self, func: str, args: Any, _kwargs: Any) -> Diagnostic | None:
        if func not in ("__setitem__", "__getitem__"):
            return None
        ndim: int = args[0].ndim
        if not is_slice_assignment_key(args[1], ndim):
            return None
        if (locator := self.locate()) is None:
            return None
        key = (locator.filename, locator.lineno)
        self._counts[key] += 1
        if self._counts[key] > self.SLICE_ASSIGN_THRESHOLD:
            return self.report(locator)
        return None


class ArrayGatherCheck(Checkup):
    """
    Attempt to detect and warn about inefficient full-array gathers.

    """

    description = (
        "entire cuPyNumeric array is being gathered into one memory, "
        "and blocking on related outstanding asynchronous work"
    )
    reference = None

    SIZE_THRESHOLD: int = 10

    def run(self, func: str, args: Any, _kwargs: Any) -> Diagnostic | None:
        """
        Check for expensive array gathers of deferred arrays.

        Args:
            func (str):
                Name of the function being invoked
            args (tuple):
                Any positional arguments the function is being called with.
                For __numpy_array__, args[0] is the array size.
            kwargs (dict):
                Any keyword arguments the function is being called with

        Returns:
            a ``Diagnostic`` in case a new detection at the current location
            is reported, otherwise None

        """
        # We are abusing the doctor API a bit here. Usually intended for func
        # to be a numpy API name. But "bad" gathers happen in a __numpy_array__
        # method on thunks. We've made it so that __numpy_array__ will only
        # invoke doctor.diagnose in case the expensive gather is actually
        # definitely happening, so there is nothing to check here besides func
        if func == "__numpy_array__":
            if args and args[0] <= self.SIZE_THRESHOLD:
                return None

            # if we can't find a user frame, then it is probably due to a
            # detection inside cupynumeric itself. Either way, there is no
            # actionable information to provide users, so just punt here.
            if (locator := self.locate()) is None:
                return None

            return self.report(locator)

        return None


class StackOpsCheck(Checkup):
    """
    Attempt to detect and warn about usage of hstack or vstack inside
    iterative loops, which can result in performance penalties in
    cuPyNumeric.

    A single call to hstack/vstack is generally fine. But calling them
    repeatedly in a loop (e.g. accumulating results) is an anti-pattern;
    users should pre-allocate and fill, or collect and stack once at the end.
    """

    LOOP_THRESHOLD: int = 2

    description = (
        "hstack/vstack called repeatedly (likely in a loop); "
        "consider pre-allocating or collecting results and stacking once"
    )
    reference = "https://docs.nvidia.com/cupynumeric/latest/user/practices.html#stack-results-in-a-performance-penalty"

    def __init__(self) -> None:
        super().__init__()
        self._call_counts: dict[tuple[str, int], int] = defaultdict(int)

    def run(self, func: str, _args: Any, _kwargs: Any) -> Diagnostic | None:
        """
        Check for use of hstack or vstack inside a loop.

        Only reports a diagnostic when the same source location has been
        observed more than ``LOOP_THRESHOLD`` times, indicating the call
        is inside an iterative loop.

        Args:
            func (str):
                Name of the function being invoked
            args (tuple):
                Any positional arguments the function is being called with
            kwargs (dict):
                Any keyword arguments the function is being called with

        Returns:
            a ``Diagnostic`` in case a new detection at the current location
            is reported, otherwise None

        """
        if func in {"hstack", "vstack"}:
            if (locator := self.locate()) is None:
                return None

            key = (locator.filename, locator.lineno)
            self._call_counts[key] += 1
            if self._call_counts[key] > self.LOOP_THRESHOLD:
                return self.report(locator)

        return None


class AdvancedIndexingCheck(Checkup):
    """
    Attempt to detect and warn about usage of advanced indexing, which can
    cause performance penalties in cuPyNumeric.

    """

    description = "use of advanced indexing can be slow in cuPyNumeric"
    reference = "https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-boolean-masks-avoid-advanced-indexing"

    def run(self, func: str, args: Any, _kwargs: Any) -> Diagnostic | None:
        """
        Check for use of advanced indexing in __getitem__ and __setitem__.

        Args:
            func (str):
                Name of the function being invoked
            args (tuple):
                Any positional arguments the function is being called with
            kwargs (dict):
                Any keyword arguments the function is being called with

        Returns:
            a ``Diagnostic`` in case a new detection at the current location
            is reported, otherwise None

        """
        if func in {"__getitem__", "__setitem__"}:
            key = args[1]
            if is_true_unoptimized_advanced_indexing(key, args[0].ndim):
                if (locator := self.locate()) is None:
                    return None

                return self.report(locator)

        return None


class NonzeroCheck(Checkup):
    """
    Attempt to detect and warn about usage of nonzero, which forces a
    synchronization point in cuPyNumeric because the output size is not
    known until the computation completes.

    """

    description = "use of nonzero can be slow in cuPyNumeric"
    reference = "https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-boolean-masks-avoid-advanced-indexing"

    def run(self, func: str, _args: Any, _kwargs: Any) -> Diagnostic | None:
        """
        Check for use of nonzero.

        Args:
            func (str):
                Name of the function being invoked
            args (tuple):
                Any positional arguments the function is being called with
            kwargs (dict):
                Any keyword arguments the function is being called with

        Returns:
            a ``Diagnostic`` in case a new detection at the current location
            is reported, otherwise None

        """
        if func == "nonzero":
            if (locator := self.locate()) is None:
                return None

            return self.report(locator)

        return None


class BuiltinReductionCheck(Checkup):
    """
    Detect use of Python built-ins that do not dispatch via a dunder
    (e.g. min, max) on cuPyNumeric arrays.
    """

    description = "Python built-in used with cuPyNumeric array"
    reference = (
        "https://docs.nvidia.com/cupynumeric/latest/user/practices.html"
    )

    @staticmethod
    def _builtins_in_source_line(line: str) -> set[str]:
        """
        Tokenize a source line and return discouraged built-in names that
        appear as function calls (``NAME`` immediately followed by ``(``).

        This detects direct calls like ``min(arr)`` or ``sum(arr)`` without
        false-positives on variables like ``min_value`` or method calls like
        ``arr.min()``.
        """
        found: set[str] = set()
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(line).readline))
            for i, tok in enumerate(tokens):
                if (
                    tok.type == tokenize.NAME  # 1. it's an identifier
                    and tok.string
                    in _DISCOURAGED_BUILTINS  # 2. check if it is a discouraged builtin
                    and i + 1 < len(tokens)  # 3. there's a next token
                    and tokens[i + 1].string == "("  # 4. followed by "("
                    and (
                        i == 0 or tokens[i - 1].string != "."
                    )  # 5. preceded by "."
                ):
                    # For x = min(arr) will pass all the above checks
                    # but np.min(arr) wouldn't since, e.g., the last check would fail
                    found.add(tok.string)
        except tokenize.TokenError:
            pass
        return found

    def _builtin_names_in_frame(self, frame: FrameType | None) -> set[str]:
        """
        Return the set of discouraged built-in names (min, max, etc.) that
        appear as values in the frame's locals or globals (e.g. ``func`` in
        ``for func in (max, min, ...): m = func(x)``).
        """
        if frame is None:
            return set()
        found: set[str] = set()
        vals = (*frame.f_locals.values(), *frame.f_globals.values())
        for name in _DISCOURAGED_BUILTINS:
            target = getattr(builtins, name)
            if any(v is target for v in vals):
                found.add(name)
        return found

    def run(self, func: str, _args: Any, _kwargs: Any) -> Diagnostic | None:
        if func not in ("__numpy_array__", "__iter__"):
            return None

        if (locator := self.locate()) is None:
            return None

        # Primary detection: tokenize the source line for direct calls
        # like min(arr), sum(arr). This works because C-implemented
        # builtins don't create Python frames, so stack inspection
        # cannot identify them.
        source = lookup_source(locator.filename, locator.lineno)
        builtin_names: set[str] = set()
        if source:
            builtin_names = self._builtins_in_source_line(source)

        # Secondary detection: check if a builtin function object is
        # stored as a value in frame locals/globals, e.g.
        # ``for func in (max, min): func(arr)``
        if not builtin_names:
            user_frame = find_last_user_frame()
            builtin_names = self._builtin_names_in_frame(user_frame)

        if not builtin_names:
            return None

        # Dedup manually instead of using self.report() because we need
        # a custom Diagnostic with builtin-specific descriptions and
        # alternatives. The builtin names are embedded in the dedup key
        # so that different builtins at the same location (e.g. max and
        # min in ``for func in (max, min): func(arr)``) each get their
        # own diagnostic.
        names_key = ",".join(sorted(builtin_names))
        synthetic = CheckupLocator(
            filename=locator.filename,
            lineno=locator.lineno,
            traceback=locator.traceback + f"\n[builtins:{names_key}]",
        )
        if synthetic in self._locators:
            return None
        self._locators.add(synthetic)

        sorted_names = sorted(builtin_names)
        label = "built-in" if len(sorted_names) == 1 else "built-ins"
        suggestions = ", ".join(
            f"{n} -> {_BUILTIN_ALTERNATIVES.get(n, f'np.{n}(arr)')}"
            for n in sorted_names
        )
        desc = (
            f"Python {label} {', '.join(sorted_names)} used with "
            f"cuPyNumeric array; prefer: {suggestions}"
        )
        return Diagnostic(
            filename=locator.filename,
            lineno=locator.lineno,
            traceback=locator.traceback,
            source=source,
            description=desc,
            reference=self.reference,
        )


class IterCheck(Checkup):
    """
    Attempt to detect and warn about usage of __iter__ on cuPyNumeric arrays,
    which forces element-by-element access and can be slow.
    """

    description = (
        "iterating over a cuPyNumeric array with __iter__ is slow; "
        "use vectorized operations instead"
    )
    reference = "https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-array-based-operations-avoid-loops-with-indexing"  # noqa

    def run(self, func: str, _args: Any, _kwargs: Any) -> Diagnostic | None:
        if func == "__iter__":
            if (locator := self.locate()) is None:
                return None
            return self.report(locator)
        return None


class Mpi4pyCheck(Checkup):
    """
    Detect when mpi4py has been imported in the same process as cuPyNumeric.
    """

    description = (
        "mpi4py is imported in this application; using mpi4py with "
        "cuPyNumeric is not permitted because Legate manages its own "
        "communication layer"
    )
    reference = None

    def __init__(self) -> None:
        super().__init__()
        self._reported = False

    def run(self, func: str, _args: Any, _kwargs: Any) -> Diagnostic | None:
        if self._reported:
            return None

        if "mpi4py" not in sys.modules and "mpi4py.MPI" not in sys.modules:
            return None

        if (locator := self.locate()) is None:
            return None

        self._reported = True
        return self.report(locator)


ALL_CHECKS: Final[tuple[Type[Checkup], ...]] = (
    RepeatedItemOps,
    RepeatedSliceAccessCheck,
    ArrayGatherCheck,
    StackOpsCheck,
    NonzeroCheck,
    AdvancedIndexingCheck,
    BuiltinReductionCheck,
    Mpi4pyCheck,
    IterCheck,
)


class Doctor:
    """
    Attempt to warn against sub-optimal usage patterns with runtime heuristics.

    """

    _results: list[Diagnostic] = []

    def __init__(
        self, *, checks: tuple[Type[Checkup], ...] = ALL_CHECKS
    ) -> None:
        self.checks = [check() for check in checks]

    def diagnose(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """
        Run cuPyNumeric Doctor heuristic checks on the current cuPyNumeric
        funtion invocation.

        Results are collected throughout execution. Call the ``output`` method
        to generate output for the results according to the current settings.

        Args:
            name (str):
                Name of the function being invoked
            args (tuple):
                Any positional arguments the function is being called with
            kwargs (dict):
                Any keyword arguments the function is being called with

        Returns:
            None

        """
        for check in self.checks:
            if info := check.run(name, args, kwargs):
                self._results.append(info)

    @property
    def results(self) -> tuple[Diagnostic, ...]:
        return tuple(self._results)

    @property
    def output(self) -> str | None:
        """
        Generate output for any cuPyNumeric Doctor results in the specified
        format.

        Returns:
            str

        """
        if not self.results:
            return None

        try:
            out = io.StringIO()
            match settings.doctor_format():
                case "plain":
                    self._write_plain(out)
                case "json":
                    self._write_json(out)
                case "csv":
                    self._write_csv(out)
            return out.getvalue()
        except Exception as e:
            warnings.warn(
                "cuPyNumeric Doctor detected issues, but an exception "
                f"occurred generating output (no output was written): {e}"
            )
            return None

    def _write_plain(self, out: io.StringIO) -> None:
        print("\n!!! cuPyNumeric Doctor reported issues !!!", file=out)
        for result in self.results:
            print(f"\n{result}", file=out)

    def _write_json(self, out: io.StringIO) -> None:
        import json

        entries = []
        for result in self.results:
            entry = asdict(result)
            if not settings.doctor_traceback():
                entry["traceback"] = ""
            entries.append(entry)
        print(json.dumps(entries), file=out)

    def _write_csv(self, out: io.StringIO) -> None:
        import csv

        assert self.results
        writer = csv.DictWriter(
            out,
            fieldnames=asdict(self.results[0]),
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for result in self.results:
            row = asdict(result)
            if settings.doctor_traceback():
                row["traceback"] = row["traceback"].replace("\n", "\\n")
            else:
                row["traceback"] = ""
            writer.writerow(row)


doctor = Doctor()

if settings.doctor():

    def _doctor_atexit() -> None:
        if (output := doctor.output) is None:
            return

        if filename := settings.doctor_filename():
            with open(filename, "w") as f:
                f.write(output)
        else:
            print(output)

    atexit.register(_doctor_atexit)
