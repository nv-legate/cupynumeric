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
from _benchmark.sizing import SizeRequest

"""
Sync microbenchmark suite.

Tests timer.sync() under different modes to set a baseline for other
benchmarks.
"""


def sync(np, mode, runs, warmup, *, timer):
    return timed_loop(lambda: None, timer, runs, warmup, sync_mode=mode) / runs


class SyncSuite(MicrobenchmarkSuite):
    name = "sync"

    def run_suite(self, _: SizeRequest) -> None:
        self.run_timed(
            sync,
            self.np,
            ["none", "fence", "block"],
            self.runs,
            self.warmup,
            timer=self.timer,
        )
