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

# cuPyNumeric <-> PyTorch interop inside a legate leaf task.
#
# Inside a leaf task, cn.asarray(PhysicalStore) wraps the store as a
# cuPyNumeric array, then torch.from_dlpack() gives a zero-copy tensor.
#
# Run (legate driver):
#   legate --cpus 1 --gpus 1 examples/dlpack/leaf_task_interop.py
# Run (python driver):
#   LEGATE_CONFIG="--cpus 1 --gpus 1" LEGATE_AUTO_CONFIG=0 python examples/dlpack/leaf_task_interop.py

import cupynumeric as cn
import torch
from legate.core import VariantCode
from legate.core.task import task, InputStore, OutputStore


N = 10


@task(variants=(VariantCode.CPU, VariantCode.GPU))
def torch_double_inplace(src: InputStore, dst: OutputStore):
    # cn_src = cn.asarray(src)
    # cn_dst = cn.asarray(dst)

    t_src = torch.from_dlpack(src)
    t_dst = torch.from_dlpack(dst)
    print(f"  device: {t_src.device}")
    t_dst[:] = t_src * 2


if __name__ == "__main__":
    a = cn.arange(N, dtype=cn.float64)
    out = cn.empty(N, dtype=cn.float64)
    print(f"input:    {a}")

    torch_double_inplace(a, out)

    expected = cn.arange(N, dtype=cn.float64) * 2
    ok = bool(cn.allclose(out, expected))
    print(f"output:   {out}")
    print(f"expected: {expected}")
    assert ok, "FAIL"
    print("result: pass")
