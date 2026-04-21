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
#

from __future__ import annotations

import gc
import importlib
import inspect
import json

from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum, StrEnum, auto
from functools import cache
from itertools import product
from legate.util.benchmark import (
    BenchmarkLog,
    BenchmarkLogFromFilename,
    benchmark_log,
)
from legate.util.info import info as legate_info
from subprocess import CalledProcessError, check_output
from types import ModuleType
from typing import Any, Iterable

from .info import BenchmarkInfo, get_benchmark_info, _TIME
from .log_null import BenchmarkLogNull
from .summarize import Summarize
from .timer import Timer, get_timer


class ArrayPackage(StrEnum):
    NUMPY = auto()
    CUPY = auto()
    LEGATE = auto()


class CupyAllocator(StrEnum):
    DEFAULT = auto()
    MANAGED = auto()
    OFF = auto()


class RunMode(Enum):
    TIMED_EXTERNAL = auto()
    TIMED_INTERNAL = auto()
    UNTIMED = auto()


class SummarizeFlush(StrEnum):
    RUN = auto()
    EXIT = auto()
    NEVER = auto()


@dataclass
class BenchmarkHarnessConfig:
    """Configuration object for :py:class:`BenchmarkHarness`."""

    repeat: int
    package: ArrayPackage
    short: bool
    cupy_allocator: CupyAllocator
    log_conda_list: bool
    log_metadata_extra: dict[str, Any]
    summarize: Summarize | None
    summarize_flush: SummarizeFlush

    @classmethod
    def add_parser_group(cls, parser: ArgumentParser, name: str) -> Any:
        """Add arguments to a parser to configure a benchmark harness.

        Parameter
        ---------
        parser: ArgumentParser
            parser to modify
        name: str
            name for the parser group

        Returns
        -------
        Any
            The group created by ``parser.add_argument_group()``

        Command-Line Arguments
        ----------------------
        -b/--benchmark: int (default: 0)
            Number of times to benchmark an example.  `0` corresponds to running
            the example without creating a log of benchmark performance data.
            Anything greater than 0 will create a log from running the example
            that many times.

        --package: 'legate', 'numpy', or 'cupy' (default: 'legate')
            Specify the package that will provide the `np` implementation
            used by the example.

        --cupy-allocator: 'default', 'off', or 'managed' (default: 'default')
            cupy allocator to use

        --log-conda-list: bool
            Add the output of `conda list` to the benchmark log's metadata

        --log-metadata-extra [key1=value1 [key2=value2 ...]]
            Additional data to added to the benchmark log's metadata

        --short: bool
            Show the short version of benchmark metadata

        --summarize: bool
            Gather summary statistics of times from
            :py:meth:`BenchmarkHarness.run_timed`.

        --summarize-flush: 'run', 'exit', or 'never' (default: 'run')
            When to display summary statistics: after each run, when
            exiting a context manager, or never.
        """

        def metadata_tuple(arg: str) -> tuple[str, str]:
            result = tuple(arg.split("="))
            if len(result) != 2:
                raise RuntimeError(f"expected 'key=value' pair, got {arg}")
            return result

        group = parser.add_argument_group(name)
        group.add_argument(
            "-b",
            "--benchmark",
            dest="__cpn_repeat",
            metavar="N",
            type=int,
            default=0,
            help="number of times to benchmark to repeat benchmarks (0 means "
            "execute without benchmarking)",
        )
        group.add_argument(
            "--package",
            dest="__cpn_package",
            # metavar="PACKAGE",
            type=ArrayPackage,
            choices=list(ArrayPackage),
            default=ArrayPackage.LEGATE,
            help="array package package to use",
        )
        group.add_argument(
            "--short",
            dest="__cpn_short",
            action="store_true",
            help="show the short version of benchmark metadata",
        )
        group.add_argument(
            "--summarize",
            dest="__cpn_summarize",
            action="store_true",
            help="summarize times collected from runs",
        )
        group.add_argument(
            "--summarize-flush",
            type=SummarizeFlush,
            dest="__cpn_summarize_flush",
            choices=list(SummarizeFlush),
            default=SummarizeFlush.RUN,
            help="when to automatically flush the summary queue",
        )
        group.add_argument(
            "--cupy-allocator",
            dest="__cpn_cupy_allocator",
            # metavar="CUPY_ALLOCATOR",
            type=CupyAllocator,
            choices=list(CupyAllocator),
            default=CupyAllocator.DEFAULT,
            help="cupy allocator to use",
        )
        group.add_argument(
            "--log-conda-list",
            dest="__cpn_log_conda_list",
            action="store_true",
            help="add `conda list` to the benchmark metadata log",
        )
        group.add_argument(
            "--log-metadata-extra",
            dest="__cpn_metadata_extra",
            metavar="KEY=VALUE",
            type=metadata_tuple,
            nargs="+",
            default=[],
            help="additional strings to add to benchmark log metadata",
        )
        return group

    @classmethod
    def from_args(cls, args: Namespace) -> BenchmarkHarnessConfig:
        """Construct a configuration from a Namespace."""
        vargs = vars(args)
        summarize: Summarize | None = None
        if vargs["__cpn_summarize"]:
            if vargs["__cpn_package"] == ArrayPackage.LEGATE:
                # import here because legate may patch sys.stdout
                importlib.import_module("legate.core")

            from sys import stdout

            summarize = Summarize(out=stdout)
        return BenchmarkHarnessConfig(
            repeat=vargs["__cpn_repeat"],
            package=vargs["__cpn_package"],
            short=vargs["__cpn_short"],
            cupy_allocator=vargs["__cpn_cupy_allocator"],
            log_conda_list=vargs["__cpn_log_conda_list"],
            log_metadata_extra=dict(vargs["__cpn_metadata_extra"]),
            summarize=summarize,
            summarize_flush=vargs["__cpn_summarize_flush"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Like dataclasses.asdict(), but not a deep copy"""
        return dict(
            [(field.name, getattr(self, field.name)) for field in fields(self)]
        )


FAILED_TO_DETECT: str = "(failed to detect)"


@cache
def _conda_list() -> dict[str, Any] | str:
    try:
        if out := check_output(["conda", "list", "--json"]):
            info = json.loads(out.decode("utf-8"))
            names = [pkg["name"] for pkg in info]
            versions = [pkg["version"] for pkg in info]
            channels = [pkg["channel"] for pkg in info]
            builds = [pkg["build_string"] for pkg in info]
            version_len = max([len(v) for v in versions])
            build_len = max([len(b) for b in builds])
            entries = [
                f"{v:{version_len}}  {b:{build_len}}  {c}"
                for v, b, c in zip(versions, builds, channels, strict=True)
            ]
            return dict(zip(names, entries, strict=True))

    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(conda missing)"
    else:
        return FAILED_TO_DETECT


def product_args(args: tuple[Any, ...]) -> Iterable[tuple[Any, ...]]:
    """Create a generator that is a product over list arguments."""
    arg_lists = []
    for arg in args:
        if not isinstance(arg, list):
            arg_lists.append([arg])
        else:
            arg_lists.append(arg)
    return product(*arg_lists)


class BenchmarkHarness:
    """Harness for running performance benchmarks."""

    name: str = "benchmark harness"

    _config: BenchmarkHarnessConfig

    #: array package of this harness
    np: ModuleType

    #: synchronizing timer with :py:meth:`start` and :py:meth:`stop` methods.
    timer: Timer

    metadata: dict[str, Any]

    def __init__(self, arg: BenchmarkHarnessConfig | Namespace) -> None:
        """Initialize a harness.

        A harness can be initiaized from an `argparse.Namespace` that
        has been updated with :py:meth:`add_parser_group`,

        .. code-block:: python

            from argparse import ArgumentParser
            from benchmark import BenchmarkHarness

            parser = ArgumentParser()
            BenchmarkHarness.add_parser_group(parser)
            # ... add other program arguments to parser ...
            args = parser.parse_args()
            harness = BenchmarkHarness(args)

        or from a :py:class:`BenchmarkHarnessConfig` object.
        """
        if isinstance(arg, BenchmarkHarnessConfig):
            self._config = arg
        else:
            if not isinstance(arg, Namespace):
                raise RuntimeError(
                    f"argument has unrecognized type {type(arg)}"
                )
            self._config = BenchmarkHarnessConfig.from_args(arg)
        metadata: dict[str, Any] = {}
        if not self.short_metadata:
            metadata.update(
                legate_info(
                    start_runtime=(self._config.package == ArrayPackage.LEGATE)
                )
            )
        if self._config.log_conda_list:
            metadata.update({"Conda list": _conda_list()})
        metadata.update(self._config.log_metadata_extra)
        self.metadata = metadata
        match self._config.package:
            case ArrayPackage.NUMPY:
                import numpy

                self.np = numpy
            case ArrayPackage.CUPY:
                import cupy  # type: ignore[import-untyped]

                self.np = cupy
                if self._config.cupy_allocator == CupyAllocator.OFF:
                    self.np.cuda.set_allocator(None)
                elif self._config.cupy_allocator == CupyAllocator.MANAGED:
                    self.np.cuda.set_allocator(
                        self.np.cuda.MemoryPool(
                            self.np.cuda.malloc_managed
                        ).malloc
                    )
            case ArrayPackage.LEGATE:
                import cupynumeric

                self.np = cupynumeric
        self.timer = get_timer(self.np)

    @property
    def repeat(self) -> int:
        """How many times each function is repeated by :py:meth:`run`."""
        return self._config.repeat

    @property
    def package(self) -> str:
        """Value of --package."""
        return str(self._config.package)

    @property
    def short_metadata(self) -> bool:
        """Whether the benchmark will print short metadata for each :py:meth:`run`."""
        return self._config.short

    @property
    def summarize(self) -> Summarize | None:
        return self._config.summarize

    @property
    def summarize_flush(self) -> SummarizeFlush:
        return self._config.summarize_flush

    def _run(
        self,
        info: BenchmarkInfo,
        f: Callable[..., Any],
        args: list[Any] | None,
        kwargs: dict[str, Any],
        plan: Iterable[tuple[Any, ...]],
        mode: RunMode,
    ) -> None:
        benchmark_name = info.name
        sig = inspect.signature(f)
        input_columns = [
            p
            for p, v in sig.parameters.items()
            if v.kind in [v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD]
        ]
        if args is not None and len(args) != len(input_columns):
            raise RuntimeError(
                f"{benchmark_name}: {len(args)} positional arguments, expected {len(input_columns)}"
            )
        for i, c in enumerate(input_columns):
            if c in info.input_names:
                input_columns[i] = info.input_names[c]
        output_columns: list[str]
        output_names = info.output_names
        if isinstance(info.output_names, str):
            output_columns = [info.output_names]
        else:
            output_columns = list(info.output_names)

        orig_output_columns = output_columns
        if mode == RunMode.TIMED_EXTERNAL:
            output_columns = [*output_columns, _TIME]

        columns = [*input_columns, *output_columns]

        bmark: BenchmarkLog | BenchmarkLogFromFilename
        repeat = self.repeat
        if repeat == 0:
            bmark = BenchmarkLogNull()
        else:
            bmark = benchmark_log(
                benchmark_name,
                columns,
                metadata=self.metadata,
                start_runtime=False,
            )
        repeat = max(1, repeat)

        timer = self.timer
        summarize = self.summarize if mode != RunMode.UNTIMED else None
        with bmark as b:
            for plan_args in plan:
                times = []
                for _i in range(repeat):
                    input_dict = dict(
                        zip(input_columns, plan_args, strict=True)
                    )
                    time: float = 0.0
                    if mode == RunMode.TIMED_EXTERNAL:
                        timer.start()
                    output_vals = f(*plan_args, **kwargs)
                    if mode == RunMode.TIMED_EXTERNAL:
                        time = timer.stop()
                    output_dict: dict[str, Any]
                    if isinstance(output_names, str):
                        output_dict = {output_names: output_vals}
                        if mode == RunMode.TIMED_INTERNAL:
                            time = float(output_vals)
                    else:
                        if output_vals is None:
                            output_vals = ()
                        output_dict = dict(
                            zip(orig_output_columns, output_vals, strict=True)
                        )
                        if mode == RunMode.TIMED_INTERNAL:
                            time = float(output_vals[info.returns_time])
                    if mode == RunMode.TIMED_EXTERNAL:
                        output_dict[_TIME] = time
                    row = {**input_dict, **output_dict}
                    for k, v in row.items():
                        if k in info.formats:
                            row[k] = info.formats[k](v)
                    b.log(**row)
                    times.append(time)
                    # gc to encourage freeing resources between iterations
                    # for large problems
                    gc.collect()
                if summarize is not None:
                    summarize_dict = input_dict.copy()
                    for k, v in summarize_dict.items():
                        if k in info.formats:
                            summarize_dict[k] = info.formats[k](v)
                    summarize.write(benchmark_name, summarize_dict, times)
                # Flush Legate's deferred GPU deallocations between size
                # iterations.  Without the fence, large arrays from the
                # previous size may still occupy GPU memory when the next
                # (larger) size tries to allocate.
                if self._config.package == ArrayPackage.LEGATE:
                    try:
                        from legate.core import get_legate_runtime

                        get_legate_runtime().issue_execution_fence(block=True)
                    except Exception as e:
                        import warnings

                        warnings.warn(
                            f"issue_execution_fence failed; GPU memory may not be released: {e}",
                            stacklevel=2,
                        )
        if summarize and self.summarize_flush == SummarizeFlush.RUN:
            summarize.flush(title=benchmark_name)

    def run_with_generator(
        self,
        info: BenchmarkInfo | None,
        f: Callable[..., Any],
        arg_gen: Iterable[tuple[Any, ...]],
        **kwargs: Any,
    ) -> None:
        """Run a function in the harness on arguments from a generator."""
        if info is None:
            info = get_benchmark_info(f)
        self._run(info, f, None, kwargs, arg_gen, RunMode.UNTIMED)

    def run_timed_with_generator(
        self,
        info: BenchmarkInfo | None,
        f: Callable[..., Any],
        arg_gen: Iterable[tuple[Any, ...]],
        **kwargs: Any,
    ) -> None:
        """Run and time function in the harness on arguments from a generator."""
        if info is None:
            info = get_benchmark_info(f)
        if info.returns_time >= 0:
            self._run(info, f, None, kwargs, arg_gen, RunMode.TIMED_INTERNAL)
        else:
            self._run(info, f, None, kwargs, arg_gen, RunMode.TIMED_EXTERNAL)

    def run_with_info(
        self,
        info: BenchmarkInfo,
        f: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run a function with ``info`` that overrides benchmark details attached
        to ``f`` by :py:func:`benchmark_info`.
        """
        self._run(
            info, f, list(args), kwargs, product_args(args), RunMode.UNTIMED
        )

    def run_timed_with_info(
        self,
        info: BenchmarkInfo,
        f: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run and time a function with ``info`` that overrides benchmark
        details attached to ``f`` by :py:func:`benchmark_info`.
        """
        if info.returns_time >= 0:
            self._run(
                info,
                f,
                list(args),
                kwargs,
                product_args(args),
                RunMode.TIMED_INTERNAL,
            )
        else:
            self._run(
                info,
                f,
                list(args),
                kwargs,
                product_args(args),
                RunMode.TIMED_EXTERNAL,
            )

    def run(self, f: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run a function in the benchmark harness.

        Parameters
        ----------
        f: Callable
            The function to benchmark. ``f`` can have positional arguments and
            keyword arguments.  It is expected that each positional argument should
            generate a column in the table of benchmark data and keyword arguments
            should not.  It is also expected that each return value of ``f`` should
            generate a column in the table. (See :py:func:`benchmark_info`).
        *args:
            positional arguments for ``f``.  When any arguments in ``args`` is
            a ``list``, ``run`` will generate a separate call to ``f`` for each
            combination of arguments in the product of the lists.

        **kwargs:
            keyword arguments for ``f``.
        """
        info = get_benchmark_info(f)
        self.run_with_info(info, f, *args, **kwargs)

    def run_timed(
        self, f: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Run and time a function in the benchmark harness.

        Same as :py:meth:`run`, but will report runtimes, which
        can either be self-reported by ``f`` or timed externally (see
        :py:func:`benchmark_info`).
        """
        info = get_benchmark_info(f)
        self.run_timed_with_info(info, f, *args, **kwargs)

    def __enter__(self) -> BenchmarkHarness:
        return self

    def __exit__(self, _: Any, __: Any, ___: Any) -> None:
        summarize = self.summarize
        if (
            summarize is not None
            and self.summarize_flush == SummarizeFlush.EXIT
        ):
            summarize.flush()


def parse_with_harness(
    parser: ArgumentParser,
) -> tuple[Namespace, BenchmarkHarness]:
    """Convenience function for the common pattern of adding
    BenchmarkHarness arugments to a parser, parsing, and getting
    both the parsed arguments and the harness."""
    BenchmarkHarnessConfig.add_parser_group(parser, BenchmarkHarness.name)
    args = parser.parse_known_args()[0]
    harness = BenchmarkHarness(args)
    return (args, harness)
