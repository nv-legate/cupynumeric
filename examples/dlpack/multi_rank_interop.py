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

# Legate-managed partitioning with PyTorch local compute.
#
# Legate automatically partitions a 1D array across N CPUs and launches
# one leaf task per shard. Each leaf task wraps its shard as a torch.Tensor
# via DLPack and scales by (task_index + 1).
#
# In a multi-node setup, task_index corresponds to process rank when there
# is one task per node.
#
# Run (legate driver):
#   legate --cpus 1 --gpus 4 --fbmem 1024 --min-gpu-chunk 1 examples/dlpack/multi_rank_interop.py
# Run (python driver):
#   LEGATE_CONFIG="--cpus 1 --gpus 4 --fbmem 1024 --min-gpu-chunk 1" LEGATE_AUTO_CONFIG=0 python examples/dlpack/multi_rank_interop.py

import cupynumeric as cn
import torch
from legate.core import TaskContext, VariantCode, get_legate_runtime
from legate.core.task import task, InputStore, OutputStore


N = 16


@task(variants=(VariantCode.CPU, VariantCode.GPU))
def torch_scale(
    ctx: TaskContext,
    src: InputStore,
    dst: OutputStore,
    scale_factors: OutputStore,
) -> None:
    task_index = ctx.task_index[0]
    num_tasks = ctx.launch_domain.hi[0] + 1

    t_src = torch.from_dlpack(src)
    t_dst = torch.from_dlpack(dst)
    t_sf = torch.from_dlpack(scale_factors)
    print(f"  device: {t_src.device}")

    factor = task_index + 1
    t_dst[:] = t_src * factor
    t_sf[:] = factor

    print(
        f"  task {task_index}/{num_tasks}: shard={t_src.tolist()} "
        f"-> scaled by {factor} -> {t_dst.tolist()}"
    )


if __name__ == "__main__":
    a = cn.arange(N, dtype=cn.float64)
    out = cn.zeros(N, dtype=cn.float64)
    scale_factors = cn.zeros(N, dtype=cn.float64)

    print(f"input:  {a}")
    print("launching index task...")

    torch_scale(a, out, scale_factors)

    get_legate_runtime().issue_execution_fence(block=True)

    print(f"output: {out}")

    expected = a * scale_factors
    ok = bool(cn.allclose(out, expected))
    print(f"expected: {expected}")
    assert ok, "FAIL"
    print("result: pass")
