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
"""
Shared benchmark utilities for microbenchmark suite.

This module provides a lightweight coordination layer for running multiple
benchmarks within a suite using the standard benchmark.py infrastructure.

MicrobenchmarkSuite: Coordinates benchmark execution
   - Each benchmark uses run_benchmark() to create separate CSV files
   - Tracks benchmark count for suite summary

Design:
- Each benchmark calls run_timed() which creates a separate CSV file
- This preserves compatibility with existing visualization tools
- Optional unified summary table can be generated at the end
"""

from __future__ import annotations

import importlib
import inspect
import itertools
import re

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .harness import ArrayPackage, BenchmarkHarness, BenchmarkHarnessConfig
from .info import INFO
from .microbenchmark_info import (
    MISSING,
    SIZE,
    MicrobenchmarkInfo,
    get_microbenchmark_info,
)
from .sizing import SizeRequest, resolve_size_by_monotonic_search
from .use_rich import use_rich, HAVE_RICH

if TYPE_CHECKING:
    from collections.abc import Iterator
    from numpy.typing import DTypeLike


@dataclass
class MicrobenchmarkConfig(BenchmarkHarnessConfig):
    runs: int
    warmup: int
    include: re.Pattern[str] | None
    exclude: re.Pattern[str] | None
    verbosity: int
    dry_run: bool

    @classmethod
    def add_parser_group(cls, parser: ArgumentParser, name: str) -> Any:
        group = super(MicrobenchmarkConfig, cls).add_parser_group(parser, name)
        group.add_argument(
            "--runs",
            dest="__cpn_runs",
            metavar="RUNS",
            type=int,
            default=5,
            help="Number of timing runs per benchmark",
        )
        group.add_argument(
            "--warmup",
            dest="__cpn_warmup",
            metavar="WARMUP",
            type=int,
            default=2,
            help="Number of warmup runs",
        )
        group.add_argument(
            "--include",
            dest="__cpn_include",
            metavar="REGEX",
            type=str,
            help="Filter to restrict to benchmarks matching a regular expression",
        )
        group.add_argument(
            "--exclude",
            dest="__cpn_exclude",
            metavar="REGEX",
            type=str,
            help="Filter to exclude benchmarks matching a regular expression",
        )
        group.add_argument(
            "--verbose",
            dest="__cpn_verbose",
            action="store_true",
            help="Print benchmark names as they are being called",
        )
        group.add_argument(
            "--debug",
            dest="__cpn_debug",
            action="store_true",
            help="Print debugging information",
        )
        group.add_argument(
            "--dry-run",
            dest="__cpn_dry_run",
            action="store_true",
            help="Plan but do not run benchmarks",
        )
        return group

    @classmethod
    def from_args(cls, args: Namespace) -> MicrobenchmarkConfig:
        super_conf = super(MicrobenchmarkConfig, cls).from_args(args)
        vargs = vars(args)
        include: re.Pattern[str] | None = None
        exclude: re.Pattern[str] | None = None
        if vargs.get("__cpn_include", None) is not None:
            include = re.compile(vargs["__cpn_include"])
        if vargs.get("__cpn_exclude", None) is not None:
            exclude = re.compile(vargs["__cpn_exclude"])
        verbosity = 0
        if bool(vargs["__cpn_verbose"]):
            verbosity = 1
        if bool(vargs["__cpn_debug"]):
            verbosity = 2
        return MicrobenchmarkConfig(
            **super_conf.to_dict(),
            runs=int(vargs["__cpn_runs"]),
            warmup=int(vargs["__cpn_warmup"]),
            include=include,
            exclude=exclude,
            verbosity=verbosity,
            dry_run=bool(vargs["__cpn_dry_run"]),
        )

    def _print_msg(self, msg: str, console: Any = None) -> None:
        if self.package == ArrayPackage.LEGATE:
            # import here because legate may patch stdout
            importlib.import_module("legate.core")

        from sys import stdout

        if HAVE_RICH and use_rich(
            stdout, start_runtime=self.package == ArrayPackage.LEGATE
        ):
            from rich.console import Console

            if console is not None:
                assert isinstance(console, Console)
                console.print(msg)
            else:
                console = Console(file=stdout)
                console.print(msg)
        else:
            print(msg)

    def info(self, msg: str, console: Any = None) -> None:
        if self.verbosity >= 1:
            self._print_msg(msg, console=console)

    def debug(self, msg: str, console: Any = None) -> None:
        if self.verbosity >= 2:
            self._print_msg(msg, console=console)

    def print_panel(
        self, lines: list[str], title: str = "", console: Any = None
    ) -> None:
        if self.package == ArrayPackage.LEGATE:
            # import here because legate may patch stdout
            importlib.import_module("legate.core")

        from sys import stdout

        if HAVE_RICH and use_rich(
            stdout, start_runtime=self.package == ArrayPackage.LEGATE
        ):
            from rich.console import Console
            from rich.panel import Panel

            rich_console: Console
            if console is not None:
                assert isinstance(console, Console)
                rich_console = console
            else:
                rich_console = Console(file=stdout)
            rich_console.print(
                Panel.fit(
                    "\n".join(lines),
                    title=f"[bold]{title}[/bold]" if title else None,
                )
            )
        else:
            print("=" * 80)
            if title:
                print(title)
            print("-" * 80)
            for line in lines:
                print(line)
            print("=" * 80)

    def check_run(self, name: str) -> bool:
        allowed_include = True
        allowed_exclude = True
        if self.include is not None:
            allowed_include = self.include.search(name) is not None
        if self.exclude is not None:
            allowed_exclude = self.exclude.search(name) is None
        return allowed_include and allowed_exclude


class MicrobenchmarkSuite(BenchmarkHarness):
    """
    Handles common behavior of all microbenchmark suites.
    """

    name: str = "microbenchmark_suite_base"

    @classmethod
    def add_suite_parser_group(cls, parser: ArgumentParser) -> None:
        """Add arguments to configure a suite to the parser."""
        pass

    def __init__(self, config: MicrobenchmarkConfig, args: Namespace) -> None:
        """
        Initialize microbenchmark suite.

        Parameters
        ----------
        config: MicrobenchmarkConfig
            Configuration object common to all subclasses.
        args: Namespace
            parsed arguments to get arguments added with :py:meth:`add_suite_parser_group`.
        """
        self._config: MicrobenchmarkConfig
        super().__init__(config)
        self.benchmark_count: int = 0
        self.benchmark_variant_count: int = 0

    @property
    def runs(self) -> int:
        return self._config.runs

    @property
    def warmup(self) -> int:
        return self._config.warmup

    @property
    def dry_run(self) -> bool:
        return self._config.dry_run

    def check_run(self, name: str) -> bool:
        return self._config.check_run(name)

    def info(self, msg: str) -> None:
        self._config.info(msg, console=self._console)

    def debug(self, msg: str) -> None:
        self._config.debug(msg, console=self._console)

    def print_panel(self, lines: list[str], title: str = "") -> None:
        self._config.print_panel(lines, title, console=self._console)

    def print_config(self) -> None:
        pass

    def print_suite_summary(self) -> None:
        """Print suite-level summary."""
        if self.benchmark_count > 0:
            msg = f"Total benchmarks run: {self.benchmark_count}"
            if self.benchmark_variant_count > self.benchmark_count:
                msg += f" ({self.benchmark_variant_count} variants)"
            self.print_panel([msg], f"SUITE COMPLETE: {self.name}")

    def default_arguments(self) -> dict[str, Any]:
        """Default values for arguments used by all benchmarks in a suite."""
        return {
            "np": self.np,
            "runs": self.runs,
            "warmup": self.warmup,
            "timer": self.timer,
        }

    def dtypes(self) -> list[DTypeLike]:
        """values that a 'dtype' argument should assume in ``run_suite``."""
        return []

    def _plan_microbenchmark(
        self,
        b_info: MicrobenchmarkInfo,
        bmark: Callable[..., Any],
        size_request: SizeRequest,
    ) -> tuple[list[tuple[Any, ...]], dict[str, Any]]:
        """Get the plan of args (and common kwargs) for the size request."""

        plan_args: list[tuple[Any, ...]] = []
        plan_kwargs: list[dict[str, Any]] = []

        # Bucket parameters by kind and collect their defaults in one pass.
        sig = inspect.signature(bmark)
        pos_args: list[str] = []
        kw_args: list[str] = []
        initial_args: dict[str, Any] = {}
        for k, v in sig.parameters.items():
            if v.kind in (v.POSITIONAL_ONLY, v.POSITIONAL_OR_KEYWORD):
                pos_args.append(k)
            else:
                kw_args.append(k)
            initial_args[k] = MISSING if v.default == v.empty else v.default

        if initial_args.get("size") is MISSING:
            initial_args["size"] = SIZE
        initial_args.update(self.default_arguments())

        self.debug(f"- base arguments: {b_info.pretty_args(initial_args)}")
        plan = b_info.get_plan(self, initial_args)
        if initial_args.get("dtype") is MISSING and (dtypes := self.dtypes()):
            plan = [
                {**p, "dtype": dtype}
                for p, dtype in itertools.product(plan, dtypes)
            ]

        uses_rescale = size_request.uses_work_rescale

        def _debug_details(
            size: int,
            complete: dict[str, Any],
            full: dict[str, Any],
            indent: str,
        ) -> None:
            deps = {
                k: v for k, v in complete.items() if full[k] in [SIZE, MISSING]
            }
            if deps:
                self.debug(f"{indent}- size-dependent arguments: {deps}")
            lines = b_info.get_explain_bytes(size, complete)
            if lines:
                self.debug(
                    f"{indent}- memory size explanation:\n"
                    + "\n".join(f"{indent}  - {s}" for s in lines)
                )
            lines = b_info.get_explain_work(complete)
            if lines:
                self.debug(
                    f"{indent}- work explanation:\n"
                    + "\n".join(f"{indent}  - {s}" for s in lines)
                )

        use_memory_size = size_request.memory_target_bytes is not None

        sizes = (
            size_request.exact_size
            if size_request.exact_size is not None
            else size_request.memory_target_bytes
        )
        assert sizes is not None

        args_and_sizes = itertools.product(plan, sizes)

        for args, sizelike in args_and_sizes:
            self.debug(f"- plan case {b_info.pretty_args(args)}:")
            full_args = {**initial_args, **args}

            size: int

            size_type = "resolved" if use_memory_size else "requested"

            if not use_memory_size:
                size = sizelike
            else:
                target_bytes = sizelike
                self.debug(
                    f"  - requested memory size: {target_bytes:,} bytes"
                )
                size = resolve_size_by_monotonic_search(
                    target_bytes,
                    estimate_value=lambda s: b_info.get_bytes(s, full_args),
                    initial_guess=max(1, target_bytes),
                )

            complete_args = b_info.complete_args_from_size(size, full_args)
            base_bytes = b_info.get_bytes(size, complete_args)
            base_work = b_info.get_work(size, complete_args)
            self.debug(
                f"  - {size_type} size: {size:,} "
                f"(memory size: {base_bytes:,} bytes; work: {base_work:e})"
            )

            indent = "    " if uses_rescale else "  "

            for rescale in size_request.rescale_by_work:
                if uses_rescale:
                    self.debug(f"  - rescale case {rescale}:")
                rescaled_size = size
                rescaled_args = complete_args
                rescaled_bytes = base_bytes
                rescaled_work = base_work
                if rescale != 1.0:
                    target_work = base_work * rescale
                    rescaled_size = resolve_size_by_monotonic_search(
                        target_work,
                        estimate_value=lambda s: b_info.get_work(s, full_args),
                        initial_guess=max(1, round(size * rescale)),
                    )
                    rescaled_args = b_info.complete_args_from_size(
                        rescaled_size, full_args
                    )
                    rescaled_bytes = b_info.get_bytes(
                        rescaled_size, rescaled_args
                    )
                    rescaled_work = b_info.get_work(
                        rescaled_size, rescaled_args
                    )
                if uses_rescale:
                    self.debug(
                        f"    - resolved size: {rescaled_size:,} "
                        f"(memory size: {rescaled_bytes:,} bytes; work: {rescaled_work:e})"
                    )
                _debug_details(rescaled_size, rescaled_args, full_args, indent)

                if b_info.should_skip(self, rescaled_args):
                    self.debug(
                        "- skipping due to @microbenchmark(skip=...) evaluating True"
                    )
                    continue
                search_string = b_info.format_search_string(
                    bmark, (), rescaled_args
                )
                if self.check_run(search_string):
                    self.debug(
                        f"{indent}- adding to the plan:\n      "
                        f"{b_info.pretty_args(rescaled_args)}"
                    )
                    plan_args.append(tuple(rescaled_args[k] for k in pos_args))
                    plan_kwargs.append({k: rescaled_args[k] for k in kw_args})
                else:
                    self.debug(f"{indent}- skipping due to regex filters:")
                    self.debug(f"{indent}  - search string: '{search_string}'")
                    if self._config.include:
                        self.debug(
                            f"{indent}  - inclusion filter: "
                            f"'{self._config.include.pattern}'"
                        )
                    if self._config.exclude:
                        self.debug(
                            f"{indent}  - exclusion filter: "
                            f"'{self._config.exclude.pattern}'"
                        )

        if not plan_kwargs:
            return plan_args, {}
        first_kwargs = plan_kwargs[0]
        if any(kw != first_kwargs for kw in plan_kwargs):
            raise RuntimeError(
                "keyword must be the same for every argument set in "
                "@microbenchmark(plan)"
            )
        return plan_args, first_kwargs

    def run_suite(self, size_request: SizeRequest) -> None:
        """Run a suite of microbenchmarks.

        Methods of the suite that have been decorate with ``@microbenchmark()``
        are gathered, and the information provided to the decorator about
        how

        - arguments are determined from size,
        - memory size depends on arguments, and
        - work depends on arguments

        are used to generate a plan of calls to the microbenchmark
        using ``BenchmarkHarness.run_timed_with_generator()``.
        """
        bmarks = inspect.getmembers(self, lambda m: hasattr(m, INFO))
        for _, bmark in bmarks:
            # update the name
            b_info = get_microbenchmark_info(bmark)
            b_name = f"{self.name}_{b_info.name}"
            b_info = b_info.replace(name=b_name)

            self.debug(f"Processing {b_name}():")

            # check for early exit
            if isinstance(b_info.skip, bool) and b_info.skip:
                self.debug("- skipping due to @microbenchmark(skip=True)")
                continue

            plan_args, plan_kwargs = self._plan_microbenchmark(
                b_info, bmark, size_request
            )
            if not plan_args:
                continue

            def gen() -> Iterator[tuple[Any, ...]]:
                for args in plan_args:
                    search_string = b_info.format_search_string(
                        bmark, args, plan_kwargs
                    )
                    self.info(search_string)
                    yield args

            if self.dry_run:
                # step through the generator to print calls to info stream
                for _ in gen():
                    pass
            else:
                super().run_timed_with_generator(
                    b_info, bmark, gen(), **plan_kwargs
                )
                self.benchmark_count += 1
                self.benchmark_variant_count += len(plan_args)

    def __enter__(self) -> MicrobenchmarkSuite:
        self.print_config()
        s = super().__enter__()
        assert isinstance(s, MicrobenchmarkSuite)
        return s

    def __exit__(self, a: Any, b: Any, c: Any) -> None:
        self.print_suite_summary()
        super().__exit__(a, b, c)
