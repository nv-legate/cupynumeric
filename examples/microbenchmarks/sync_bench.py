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

from _benchmark import MicrobenchmarkSuite, timed_loop
from _benchmark.harness import TimerMode
from _benchmark.sizing import SizeRequest
from _benchmark.timer import Timer

"""
Sync microbenchmark suite.

Tests timer.sync() under different modes to set a baseline for other
benchmarks.
"""


def sync(
    np,
    mode,
    start_mode: TimerMode,
    runs,
    warmup,
    *,
    execution_timer: Timer,
    wall_timer: Timer,
):
    timer: Timer
    match start_mode:
        case TimerMode.EXECUTION:
            timer = execution_timer
        case TimerMode.WALL:
            timer = wall_timer
    return timed_loop(lambda: None, timer, runs, warmup, sync_mode=mode) / runs


class SyncSuite(MicrobenchmarkSuite):
    name = "sync"

    def run_suite(self, _: SizeRequest) -> None:
        self.run_timed(
            sync,
            self.np,
            ["none", "fence", "block"],
            list(TimerMode),
            self.runs,
            self.warmup,
            execution_timer=self.execution_timer,
            wall_timer=self.wall_timer,
        )
