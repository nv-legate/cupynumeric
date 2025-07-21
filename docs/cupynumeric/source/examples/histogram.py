#!/usr/bin/env python
# Copyright 2024 NVIDIA Corporation
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

import cupy
import numpy
import argparse
import cupynumeric as cpn
import legate.core as lg
from legate.core import broadcast, VariantCode, TaskContext
from legate.core.task import task, InputArray, ReductionArray, ADD
from legate.timing import time

# [histogram-start]
@task(variants=(VariantCode.CPU, VariantCode.GPU,),
      constraints=(broadcast("hist"),))
def histogram_task(ctx: TaskContext, data: InputArray, hist: ReductionArray[ADD], N_bins: int):
    xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
    data_local = xp.asarray(data)
    hist_local = xp.asarray(hist)

    local_hist, _ = xp.histogram(data_local, bins=N_bins)
    hist_local[:] = hist_local + local_hist
# [histogram-end]

def main():
    parser = argparse.ArgumentParser(description="Run Histogram operation.")
    parser.add_argument("--size", type=int, default=1000, help="Size of input arrays")
    args = parser.parse_args()
      
    # [input-section]
    size = args.size
    NUM_BINS = 10

    data = cpn.random.randint(0, NUM_BINS, size=(size,), dtype=cpn.int32)
    hist = cpn.zeros((NUM_BINS,), dtype=cpn.int32)

    histogram_task(data, hist, NUM_BINS)
    # [function-call]
      
    # Benchmark run
    rt = lg.get_legate_runtime()
    rt.issue_execution_fence()
    start = time()

    histogram_task(data, hist, NUM_BINS)

    rt.issue_execution_fence()
    end = time()

    print(f"\nTime elapsed for histogram : {(end - start)/1000:.6f} milliseconds")

if __name__ == "__main__":
    main()
