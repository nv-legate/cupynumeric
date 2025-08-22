#!/usr/bin/env python
# Copyright 2025 NVIDIA Corporation
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

# [code-start]
import cupy
import numpy
import cupynumeric as cpn
from legate.core import TaskContext, VariantCode
from legate.core.task import task, InputArray, OutputArray


@task(variants=(VariantCode.CPU, VariantCode.GPU))
def simple_in_out(
    ctx: TaskContext, in_store: InputArray, out_store: OutputArray
) -> None:
    xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
    in_store = xp.asarray(in_store)
    out_store = xp.asarray(out_store)
    out_store[:] = in_store[:]


def main() -> None:
    in_arr = cpn.array([1, 2, 3], dtype=cpn.int64)
    out_arr = cpn.zeros((3,), dtype=cpn.int64)
    simple_in_out(in_arr, out_arr)
    print(out_arr)


if __name__ == "__main__":
    main()
# [code-end]
