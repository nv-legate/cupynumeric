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
General Random Generation Benchmark

Operations tested:
1. discrete uniform random.randint()
2. uniform with four bigenerator types, for float32, float64
"""

from typing import Any

from _benchmark import MicrobenchmarkSuite, microbenchmark, timed_loop


def _bitgen_plan(suite):
    bitgen_types = []
    np = suite.np

    match np.__name__:
        case "cupynumeric":
            bitgen_types = [
                np.random.XORWOW,
                np.random.MRG32k3a,
                np.random.PHILOX4_32_10,
            ]
        case "cupy":
            bitgen_types = [
                np.random.XORWOW,
                np.random.MRG32k3a,
                np.random.Philox4x3210,
            ]
        case "numpy":
            bitgen_types = [
                np.random.MT19937,
                np.random.PCG64,
                np.random.PCG64DXSM,
                np.random.Philox,
                np.random.SFC64,
            ]
    return [{"bitgenerator_type": bitgen_type} for bitgen_type in bitgen_types]


class RandomSuite(MicrobenchmarkSuite):
    name = "random"

    def dtypes(self) -> list[str]:
        if self.np.__name__ == "numpy":
            # numpy bitgenerator only generates float64
            return ["float64"]
        return ["float32", "float64"]

    def default_arguments(self) -> dict[str, Any]:
        return {
            **super().default_arguments(),
            "uniform_takes_dtype": self.np.__name__ != "numpy",
        }

    @microbenchmark(args_to_arrays=lambda size: [("output", size, "int")])
    def randint(np, size, runs, warmup, *, timer):
        """random.randint(low, high, size)."""

        def operation():
            return np.random.randint(1, 1000, size=size)

        return timed_loop(operation, timer, runs, warmup) / runs

    @microbenchmark(
        formats={"bitgenerator_type": lambda x: x.__name__},
        # even though cupy provides a dtype argument to
        # Generator.uniform(), memory profiling suggests
        # that enough memory is allocated for float64, even
        # when dtype=float32
        args_to_arrays=lambda size: [("output", size, "float64")],
        plan=_bitgen_plan,
    )
    def bitgenerator(
        np,
        size,
        runs,
        warmup,
        bitgenerator_type,
        dtype,
        *,
        timer,
        uniform_takes_dtype,
    ):
        """bitgenerator operation."""

        bitgen = bitgenerator_type(seed=1729)
        gen = np.random.Generator(bitgen)

        def operation():
            low = 1.414
            high = 3.14
            if uniform_takes_dtype:
                return gen.uniform(low, high, size=size, dtype=dtype)
            else:
                return gen.uniform(low, high, size=size)

        return timed_loop(operation, timer, runs, warmup) / runs
