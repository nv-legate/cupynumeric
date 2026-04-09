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

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Iterable

from .harness import ArrayPackage, BenchmarkHarness, BenchmarkHarnessConfig
from .info import BenchmarkInfo, get_benchmark_info
from .sizing import SizeRequest, SizeResolution
from .use_rich import use_rich, HAVE_RICH


@dataclass
class MicrobenchmarkConfig(BenchmarkHarnessConfig):
    runs: int
    warmup: int

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
        return group

    @classmethod
    def from_args(cls, args: Namespace) -> MicrobenchmarkConfig:
        super_conf = super(MicrobenchmarkConfig, cls).from_args(args)
        vargs = vars(args)
        return MicrobenchmarkConfig(
            **super_conf.to_dict(),
            runs=int(vargs["__cpn_runs"]),
            warmup=int(vargs["__cpn_warmup"]),
        )

    def print_panel(self, lines: list[str], title: str = "") -> None:
        if self.package == ArrayPackage.LEGATE:
            # import here because legate may patch stdout
            importlib.import_module("legate.core")

        from sys import stdout

        if HAVE_RICH and use_rich(
            stdout, start_runtime=self.package == ArrayPackage.LEGATE
        ):
            from rich.console import Console
            from rich.panel import Panel

            console = Console(file=stdout)
            console.print(
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


class MicrobenchmarkSuite(BenchmarkHarness):
    """
    Handles common behavior of all microbenchmark suites.

    When adding to the microbenchmarks, Do not subclass directly: use
    :py:class:`MicrobenchmarkSuite`.
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

    def _modify_info(self, info: BenchmarkInfo) -> BenchmarkInfo:
        """Prepend the suite name and change the default time output column to
        'time per run (ms)'"""
        full_name = f"{self.name}_{info.name}"
        info = info.replace(name=full_name)
        time_output = info.returns_time
        time_string = "time per run (ms)"
        if time_output >= 0:
            if isinstance(info.output_names, str):
                assert time_output == 0
                info = info.replace(output_names=time_string)
            else:
                new_output_names = [name for name in info.output_names]
                new_output_names[time_output] = time_string
                info = info.replace(output_names=new_output_names)
        return info

    def _update_counts(self, args: Iterable[Any]) -> None:
        lengths = [(len(a) if isinstance(a, list) else 1) for a in args]
        num_variants = int(prod(lengths))
        self.benchmark_count += 1
        self.benchmark_variant_count += num_variants

    def _counted_arg_gen(
        self, arg_gen: Iterable[tuple[Any, ...]]
    ) -> Iterable[tuple[Any, ...]]:
        for args in arg_gen:
            self.benchmark_variant_count += 1
            yield args

    def run_with_generator(
        self,
        info: BenchmarkInfo | None,
        f: Callable[..., Any],
        arg_gen: Iterable[tuple[Any, ...]],
        **kwargs: Any,
    ) -> None:
        if info is None:
            info = get_benchmark_info(f)
        info = self._modify_info(info)
        super().run_with_generator(
            info, f, self._counted_arg_gen(arg_gen), **kwargs
        )
        self.benchmark_count += 1

    def run_timed_with_generator(
        self,
        info: BenchmarkInfo | None,
        f: Callable[..., Any],
        arg_gen: Iterable[tuple[Any, ...]],
        **kwargs: Any,
    ) -> None:
        if info is None:
            info = get_benchmark_info(f)
        info = self._modify_info(info)
        super().run_timed_with_generator(
            info, f, self._counted_arg_gen(arg_gen), **kwargs
        )
        self.benchmark_count += 1

    def run_with_info(
        self,
        info: BenchmarkInfo,
        f: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        info = self._modify_info(info)
        super().run_with_info(info, f, *args, **kwargs)
        self._update_counts(args)

    def run_timed_with_info(
        self,
        info: BenchmarkInfo,
        f: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        info = self._modify_info(info)
        super().run_timed_with_info(info, f, *args, **kwargs)
        self._update_counts(args)

    def run(self, f: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.run_with_info(get_benchmark_info(f), f, *args, **kwargs)

    def run_timed(
        self, f: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        self.run_timed_with_info(get_benchmark_info(f), f, *args, **kwargs)

    def print_panel(self, lines: list[str], title: str = "") -> None:
        self._config.print_panel(lines, title)

    def print_config(self) -> None:
        pass

    def print_suite_summary(self) -> None:
        """Print suite-level summary."""
        if self.benchmark_count > 0:
            msg = f"Total benchmarks run: {self.benchmark_count}"
            if self.benchmark_variant_count > self.benchmark_count:
                msg += f" ({self.benchmark_variant_count} variants)"
            self.print_panel([msg], f"SUITE COMPLETE: {self.name}")

    def print_size_resolution(self, resolutions: list[SizeResolution]) -> None:
        lines = []
        for i, resolution in enumerate(resolutions):
            if i > 0:
                lines.append("")
            lines.extend(resolution.panel_lines())
        self.print_panel(lines, title=f"Memory Size Heuristic: {self.name}")

    def run_suite(self, size_request: SizeRequest) -> None:
        pass

    def __enter__(self) -> MicrobenchmarkSuite:
        self.print_config()
        s = super().__enter__()
        assert isinstance(s, MicrobenchmarkSuite)
        return s

    def __exit__(self, a: Any, b: Any, c: Any) -> None:
        self.print_suite_summary()
        super().__exit__(a, b, c)
