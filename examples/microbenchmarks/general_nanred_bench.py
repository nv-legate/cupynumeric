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
General NaN Reduction Benchmark

Operations tested:
1. nansum
2. nanmean

on float32, float64
"""

from _benchmark import MicrobenchmarkSuite, microbenchmark, timed_loop


class NanRedSuite(MicrobenchmarkSuite):
    name = "nanred"

    def dtypes(self):
        return ["float32", "float64"]

    @microbenchmark(
        formats={"func": lambda f: f.__name__},
        args_to_arrays=lambda size, dtype: [("input", size, dtype)],
        plan=lambda suite: [
            {"func": func} for func in [suite.np.nansum, suite.np.nanmean]
        ],
    )
    def nan_red(np, func, dtype, size, runs, warmup, *, timer):
        """[np.nansum, np.nanmean](input_with_half_nans)."""

        in_arr = np.ones((size,), dtype=dtype)
        half_sz = size // 2
        in_arr[half_sz:size] = np.dtype(dtype).type(np.nan)

        def operation():
            return func(in_arr)

        return timed_loop(operation, timer, runs, warmup) / runs
