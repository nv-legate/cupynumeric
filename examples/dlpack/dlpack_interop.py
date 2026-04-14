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

# DLPack interop between cuPyNumeric and PyTorch.
#
# BiDirectional Test
#   cpn -> torch:  cuPyNumeric owns the buffer, PyTorch writes into it.
#   torch -> cpn:  PyTorch owns the buffer, cuPyNumeric reads from it.
#
# Run (legate driver):
#   legate --cpus 1 --gpus 1 examples/dlpack/dlpack_interop.py
# Run (python driver):
#   LEGATE_CONFIG="--cpus 1 --gpus 1" LEGATE_AUTO_CONFIG=0 python examples/dlpack/dlpack_interop.py

import cupynumeric as cn
import torch


N = 10


def cpn_to_torch():
    src = cn.arange(N, dtype=cn.float64)
    print(f"cn created:    {src}")

    t = torch.from_dlpack(src.__dlpack__())
    print(f"device:        {t.device}")
    t *= 2
    print(f"torch doubled: {t}")

    expected = cn.arange(N, dtype=cn.float64) * 2
    ok = bool(cn.allclose(src, expected))
    print(f"cn readback:   {src}  {'ok' if ok else 'MISMATCH'}")
    return ok


def torch_to_cpn():
    t = torch.arange(N, dtype=torch.float64)
    print(f"torch created: {t}")
    print(f"device:        {t.device}")

    arr = cn.from_dlpack(t)
    t += 100
    print(f"torch += 100:  {t}")

    expected = cn.arange(N, dtype=cn.float64) + 100
    ok = bool(cn.allclose(arr, expected))
    print(f"cn readback:   {arr}  {'ok' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    print("--- cpn -> torch ---")
    p1 = cpn_to_torch()
    print()
    print("--- torch -> cpn ---")
    p2 = torch_to_cpn()
    print()

    assert p1 and p2, "FAIL"
    print("result: pass")
