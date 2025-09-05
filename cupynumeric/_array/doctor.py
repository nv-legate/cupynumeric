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
import io
import traceback
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Final, Type

import numpy as np

from .._utils.stack import find_last_user_frame
from ..settings import settings


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


class ArrayGatherCheck(Checkup):
    """
    Attempt to detect and warn about inefficient full-array gathers.

    """

    description = (
        "entire cuPyNumeric array is being gathered into one memory, "
        "and blocking on related outstanding asynchronous work"
    )
    reference = None

    def run(self, func: str, _args: Any, _kwargs: Any) -> Diagnostic | None:
        """
        Check for expensive array gathers of deferred arrays.

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
        # We are abusing the doctor API a bit here. Usually intended for func
        # to be a numpy API name. But "bad" gathers happen in a __numpy_array__
        # method on thunks. We've made it so that __numpy_array__ will only
        # invoke doctor.diagnose in case the expensive gather is actually
        # definitely happening, so there is nothing to check here besides func
        if func == "__numpy_array__":
            # if we can't find a user frame, then it is probably due to a
            # detection inside cupynumeric itself. Either way, there is no
            # actionable information to provide users, so just punt here.
            if (locator := self.locate()) is None:
                return None

            return self.report(locator)

        return None


ALL_CHECKS: Final[tuple[Type[Checkup], ...]] = (
    RepeatedItemOps,
    ArrayGatherCheck,
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
