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
General astype Benchmark

Operations tested:
astype()

on float32, float64
"""

from _benchmark import MicrobenchmarkSuite, microbenchmark, timed_loop


class AsTypeSuite(MicrobenchmarkSuite):
    name = "astype"

    def dtypes(self):
        return ["float32", "float64"]

    @microbenchmark(
        args_to_arrays=lambda size, dtype: [
            ("input", size, "int"),
            ("output", size, dtype),
        ]
    )
    def astype(np, dtype, size, runs, warmup, *, timer):
        """np.astype"""

        in_arr = np.random.randint(1, 1000, size=size)
        dtype = np.dtype(dtype)

        def operation():
            out_arr = in_arr.astype(dtype)
            return out_arr

        return timed_loop(operation, timer, runs, warmup) / runs
