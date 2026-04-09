# copyright 2026 nvidia corporation
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
#

from __future__ import annotations

from math import sqrt
from typing import Any, TextIO
from dataclasses import dataclass

from .info import _TIME
from .use_rich import use_rich, HAVE_RICH

if HAVE_RICH:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text


@dataclass
class Summary:
    id_string: str
    num_samples: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float


class Summarize:
    def __init__(
        self, out: TextIO, *, summary_column_name: str = _TIME
    ) -> None:
        self.data: dict[str, list[tuple[dict[str, Any], list[float]]]] = {}
        self.stream = out
        self.use_rich = use_rich(out)
        self.summary_column_name = summary_column_name
        if use_rich(out):
            self.console = Console(file=out)

    def write(
        self, name: str, args: dict[str, Any], times: list[float]
    ) -> None:
        if name not in self.data:
            self.data[name] = []
        self.data[name].append((args, times))

    def flush(self, title: str | None = None) -> None:
        summaries = []
        for name, entries in self.data.items():
            all_args = set(k for e in entries for k in e[0].keys())

            # find function arguments that varied from entry to entry
            varying_args = []
            for arg in all_args:
                all_values = set()
                for e in entries:
                    d = e[0]
                    if arg in d:
                        all_values.add(d[arg])
                if len(all_values) > 1:
                    varying_args.append(arg)

            # write a summary for each entry
            for args, times in entries:
                n = len(times)
                if n < 1:
                    continue

                # identify each entry by the values of its varying arguments
                varying = {}
                for arg in args:
                    if arg in varying_args:
                        varying[arg] = args[arg]
                id_string = name
                if varying:
                    id_string += (
                        "::("
                        + ",".join(f"{i[0]}={i[1]}" for i in varying.items())
                        + ")"
                    )

                v_min = min(times)
                v_max = max(times)
                v_stddev = 0.0
                v_mean = 0.0
                if n == 1:
                    v_mean = times[0]
                else:
                    samples = [t for t in times]
                    # Remove the largest and the smallest ones
                    if n >= 3:
                        samples.remove(v_max)
                    if n >= 2:
                        samples.remove(v_min)
                    v_mean = sum(samples) / len(samples)
                    variance = sum(
                        map(lambda x: (x - v_mean) ** 2, samples)
                    ) / len(samples)
                    v_stddev = sqrt(variance)
                summaries.append(
                    Summary(id_string, n, v_min, v_max, v_mean, v_stddev)
                )
        self.data.clear()

        def render(o: Any) -> Text | str:
            if self.use_rich:
                return self.console.render_str(str(o))
            else:
                return str(o)

        if len(summaries) == 0:
            return
        max_n = max(s.num_samples for s in summaries)
        max_width = max(len(s.id_string) for s in summaries)
        max_width = max(max_width, len("benchmark"))
        table: Table | None = None
        if self.use_rich:
            table = Table(
                title=f"Summary: [bold]{title}[/bold]"
                if title
                else "[bold]Summary[/bold]"
            )
        if max_n == 1:
            line_width = 0
            if table:
                table.add_column("benchmark")
                table.add_column(self.summary_column_name)
            else:
                header_string = f"{'benchmark':<{max_width}} | {self.summary_column_name}\n"
                line_width = max(80, len(header_string))

                self.stream.write("=" * line_width + "\n")
                if title:
                    self.stream.write(f"SUMMARY: {title}\n")
                else:
                    self.stream.write("SUMMARY\n")
                self.stream.write("-" * line_width + "\n")
                self.stream.write(header_string)
                self.stream.write("-" * line_width + "\n")
            for s in summaries:
                if table:
                    table.add_row(render(s.id_string), render(s.mean))
                else:
                    self.stream.write(
                        f"{s.id_string:<{max_width}} | {s.mean}\n"
                    )
            if not table:
                self.stream.write("=" * line_width + "\n")
        else:
            line_width = 0
            if table:
                table.add_column("benchmark")
                table.add_column("# samples")
                table.add_column("min (ms)")
                table.add_column("max (ms)")
                table.add_column("mean ± std. dev. (ms)")
            else:
                header_string = f"{'benchmark':<{max_width}} | # samples | min       | max       | mean ± std. dev.\n"
                line_width = max(80, len(header_string))

                self.stream.write("=" * line_width + "\n")
                if title:
                    self.stream.write(f"SUMMARY: {title}\n")
                else:
                    self.stream.write("SUMMARY\n")
                self.stream.write("-" * line_width + "\n")
                self.stream.write(
                    f"{'':<{max_width}}   {self.summary_column_name:^53}\n"
                )
                self.stream.write(header_string)
                self.stream.write("-" * line_width + "\n")
            for s in summaries:
                if table:
                    table.add_row(
                        render(s.id_string),
                        render(s.num_samples),
                        render(f"{s.minimum:5g}"),
                        render(f"{s.maximum:5g}"),
                        render(f"{s.mean:5g} ± {s.standard_deviation:3g}"),
                    )
                else:
                    self.stream.write(
                        f"{s.id_string:<{max_width}} | {s.num_samples:<9} | {s.minimum:9g} | {s.maximum:9g} | {s.mean:5g} ± {s.standard_deviation:3g}\n"
                    )
            if not table:
                self.stream.write("=" * line_width + "\n")
        if table is not None:
            assert self.console is not None
            self.console.print(table)
        self.stream.flush()
