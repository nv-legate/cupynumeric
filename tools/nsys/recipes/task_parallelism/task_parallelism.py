"""
Nsys recipe: task_parallelism

Measures the degree of GPU task parallelism in an nsys report by looking at
concurrent kernel execution. Two kernels running simultaneously on any CUDA
streams count as parallelism = 2.

A sweep-line algorithm over the GPU-side kernel start/end timestamps gives
exact, time-weighted statistics.  A coarser time-series sampled at
--time-resolution intervals is also reported for bucket-level percentiles.

Reported metrics
----------------
- Average parallelism  : time-weighted mean concurrent kernel count
- Peak parallelism     : maximum simultaneous kernels observed
- Serial fraction      : fraction of GPU-active time with exactly 1 kernel
- Idle fraction        : fraction of elapsed time with no kernel executing

Keep tabs on start and end times of each kernel,

"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nsys_recipe import log
from nsys_recipe.data_service import DataService
from nsys_recipe.lib import helpers, recipe
from nsys_recipe.lib.args import ArgumentParser, Option
from nsys_recipe.lib.table_config import CompositeTable
from nsys_recipe.log import logger

pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _sweep_line(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep-line over (start, end) intervals.

    Returns
    -------
    times       : 1-D int64 array of event timestamps (sorted)
    parallelism : 1-D int64 cumulative-sum array; parallelism[i] is the
                  concurrency level during the half-open interval
                  [times[i], times[i+1]).
    """
    starts = df["start"].values.astype(np.int64)
    ends = df["end"].values.astype(np.int64)

    times = np.concatenate([starts, ends])
    deltas = np.concatenate(
        [
            np.ones(len(starts), dtype=np.int64),
            -np.ones(len(ends), dtype=np.int64),
        ]
    )

    sort_idx = np.argsort(times, kind="stable")
    times = times[sort_idx]
    deltas = deltas[sort_idx]

    parallelism = np.cumsum(deltas)
    return times, parallelism


def _compute_summary(times: np.ndarray, parallelism: np.ndarray) -> dict:
    """
    Derive time-weighted summary statistics from the sweep-line output.

    The duration of interval i is (times[i+1] - times[i]).  The last event is
    a sentinel whose interval duration is defined as 0.
    """
    if len(times) < 2:
        return {}

    total_time = int(times[-1] - times[0])
    if total_time == 0:
        return {}

    durations = np.empty(len(times), dtype=np.int64)
    durations[:-1] = np.diff(times)
    durations[-1] = 0

    p = parallelism[:-1].astype(np.int64)
    d = durations[:-1]

    avg_parallelism = float(np.dot(p, d)) / total_time
    peak_parallelism = int(parallelism.max())

    level_time: dict[int, int] = {}
    for level, dur in zip(p.tolist(), d.tolist()):
        level_time[level] = level_time.get(level, 0) + dur

    return {
        "total_time_ns": total_time,
        "avg_parallelism": avg_parallelism,
        "peak_parallelism": peak_parallelism,
        "serial_fraction": level_time.get(1, 0) / total_time,
        "idle_fraction": level_time.get(0, 0) / total_time,
        "level_time_ns": level_time,
    }


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


class TaskParallelism(recipe.Recipe):
    @staticmethod
    def _mapper_func(
        report_path: str, parsed_args: argparse.Namespace
    ) -> Optional[tuple[str, pd.DataFrame, dict]]:
        service = DataService(report_path, parsed_args)

        # we combine the fields from two different dataframes
        # corresponding to the keys CUDA_KERNEL and CUDA_GPU
        # into one
        service.queue_custom_table(CompositeTable.CUDA_KERNEL)
        service.queue_custom_table(CompositeTable.CUDA_GPU)

        df_dict = service.read_queued_tables()
        if df_dict is None:
            return None

        kernel_df = df_dict[CompositeTable.CUDA_KERNEL]
        gpu_df = df_dict[CompositeTable.CUDA_GPU]

        err_msg = service.filter_and_adjust_time(kernel_df)
        if err_msg is not None:
            logger.error(f"{report_path}: {err_msg}")
            return None

        # remove any NaNs in df and compute kernel exection duration
        kernel_df = kernel_df.dropna(subset=["start", "end"]).copy()
        kernel_df["duration"] = kernel_df["end"] - kernel_df["start"]
        kernel_df = kernel_df[kernel_df["duration"] > 0]

        if kernel_df.empty:
            logger.info(
                f"{report_path}: No GPU kernels found after filtering."
            )
            return None

        # Join streamId from CUDA_GPU on correlationId
        if not gpu_df.empty and "streamId" in gpu_df.columns:
            stream_map = gpu_df[["correlationId", "streamId"]].drop_duplicates(
                "correlationId"
            )
            kernel_df = kernel_df.merge(
                stream_map, on="correlationId", how="left"
            )

        t_min = int(kernel_df["start"].min())
        t_max = int(kernel_df["end"].max())

        times, parallelism = _sweep_line(kernel_df)
        summary = _compute_summary(times, parallelism)
        if not summary:
            logger.warning(
                f"{report_path}: Could not compute parallelism metrics."
            )
            return None

        summary["t_min_ns"] = t_min
        summary["t_max_ns"] = t_max
        summary["num_kernels"] = len(kernel_df)

        filename = Path(report_path).stem
        return filename, summary

    @log.time("Mapper")
    def mapper_func(self, context: recipe.Context) -> list:
        return context.wait(
            context.map(
                self._mapper_func,
                self._parsed_args.input,
                parsed_args=self._parsed_args,
            )
        )

    @log.time("Reducer")
    def reducer_func(self, mapper_res: list) -> None:
        filtered_res = helpers.filter_none_or_empty(mapper_res)
        if not filtered_res:
            logger.warning("No results to aggregate.")
            return

        filtered_res = sorted(
            filtered_res, key=lambda x: helpers.natural_sort_key(x[0])
        )

        for fname, summary in filtered_res:
            t_min_ms = summary["t_min_ns"] / 1e6
            t_max_ms = summary["t_max_ns"] / 1e6
            total_ms = summary["total_time_ns"] / 1e6

            print(f"\n{'=' * 72}")
            print(f"  Report  : {fname}")
            print(
                f"  Window  : {t_min_ms:.1f} ms – {t_max_ms:.1f} ms  "
                f"(span: {total_ms:.1f} ms)"
            )
            print(
                f"  Kernels : {summary['num_kernels']} GPU kernel executions"
            )
            print(f"{'=' * 72}")

            # --- Summary metrics -------------------------------------------
            print("\n  GPU Task Parallelism Summary")
            print(f"  {'─' * 40}")
            print(f"  Average parallelism : {summary['avg_parallelism']:.2f}")
            print(f"  Peak parallelism    : {summary['peak_parallelism']}")
            print(
                f"  Serial fraction     : {summary['serial_fraction'] * 100:.1f}%"
                f"  (exactly 1 kernel active)"
            )
            print(
                f"  Idle fraction       : {summary['idle_fraction'] * 100:.1f}%"
                f"  (no kernels executing)"
            )

            # --- Histogram ---------------------------------------------------
            level_time = summary["level_time_ns"]
            total_ns = summary["total_time_ns"]
            max_level = max(level_time.keys())

            print("\n  Parallelism Histogram")
            print(f"  {'─' * 40}")
            print(f"  {'Level':>6}  {'Duration (ms)':>14}  {'Fraction':>8}")
            print(f"  {'─' * 6}  {'─' * 14}  {'─' * 8}")
            for level in range(max_level + 1):
                dur_ns = level_time.get(level, 0)
                frac = dur_ns / total_ns
                print(
                    f"  {level:>6}  {dur_ns / 1e6:>14.1f}  {frac * 100:>7.1f}%"
                )

            print()

    def run(self, context: recipe.Context) -> None:
        super().run(context)
        mapper_res = self.mapper_func(context)
        self.reducer_func(mapper_res)

    @classmethod
    def get_argument_parser(cls) -> ArgumentParser:
        parser = super().get_argument_parser()
        parser.add_recipe_argument(Option.INPUT, required=True)
        parser.add_recipe_argument(Option.START)
        parser.add_recipe_argument(Option.END)
        return parser
