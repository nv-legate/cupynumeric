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
General Scalar Reduction Benchmark

Operations tested:
1. sum
2. prod
3. min
4. max
5. argmin
6. argmax

on float32, float64
"""

from _benchmark import (
    MicrobenchmarkSuite,
    microbenchmark,
    random_array,
    timed_loop,
)


def _plan(suite):
    np = suite.np
    return [
        {"func": func}
        for func in [np.sum, np.prod, np.min, np.max, np.argmin, np.argmax]
    ]


class ScalarRedSuite(MicrobenchmarkSuite):
    name = "scalared"

    def dtypes(self) -> list[str]:
        return ["float32", "float64"]

    @microbenchmark(
        formats={"func": lambda f: f.__name__},
        args_to_arrays=lambda size, dtype: [("input", size, dtype)],
        plan=_plan,
    )
    def scalar_red(np, func, dtype, size, runs, warmup, *, timer):
        """[np.sum, np.prod, np.min, np.max, np.argmin, np.argmax]"""

        in_arr = random_array(np, size, dtype)

        def operation():
            return func(in_arr)

        return timed_loop(operation, timer, runs, warmup) / runs
