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

# Raw-pointer interop between cuPyNumeric and PyTorch.
#
# Instead of DLPack, we use data_ptr() / __array_interface__ to get the
# underlying buffer address and wrap it manually on the other side.
#
# Run (legate driver):
#   legate --cpus 1 --gpus 1 examples/dlpack/pointer_interop.py
# Run (python driver):
#   LEGATE_CONFIG="--cpus 1 --gpus 1" LEGATE_AUTO_CONFIG=0 python examples/dlpack/pointer_interop.py

import ctypes
import numpy as np
import cupynumeric as cn
import torch


N = 10


def torch_to_cpn_via_pointer():
    """Get torch's raw pointer, wrap it as a numpy array, hand to cn."""
    t = torch.arange(N, dtype=torch.float64)
    print(f"torch created: {t}")

    ptr = t.data_ptr()
    print(f"torch ptr:     {hex(ptr)}")

    # Build a numpy array that points at torch's buffer (no copy)
    c_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double * N))
    np_view = np.frombuffer(c_ptr.contents, dtype=np.float64)

    arr = cn.asarray(np_view)

    t += 50
    print(f"torch += 50:   {t}")
    print(f"np_view:       {np_view}")
    print(f"cn array:      {arr}")

    ok = bool(cn.allclose(arr, cn.arange(N, dtype=cn.float64) + 50))
    print(f"zero-copy:     {'yes' if ok else 'no (cn may have copied)'}")
    return ok


def cpn_to_torch_via_pointer():
    """Get cn's buffer via np.asarray, extract pointer, wrap in torch."""
    src = cn.arange(N, dtype=cn.float64)
    print(f"cn created:    {src}")

    np_view = np.asarray(src)
    ptr = np_view.ctypes.data
    print(f"np ptr:        {hex(ptr)}")

    # Build a torch tensor from the numpy view
    t = torch.from_numpy(np_view)

    t *= 3
    print(f"torch *= 3:    {t}")
    print(f"np_view:       {np_view}")
    print(f"cn readback:   {src}")

    ok = bool(cn.allclose(src, cn.arange(N, dtype=cn.float64) * 3))
    print(f"zero-copy:     {'yes' if ok else 'no (cn may have copied)'}")
    return ok


if __name__ == "__main__":
    print("--- torch -> cpn via raw pointer ---")
    p1 = torch_to_cpn_via_pointer()
    print()
    print("--- cpn -> torch via raw pointer ---")
    p2 = cpn_to_torch_via_pointer()
    print()

    assert p1 and p2, "FAIL"
    print("result: pass")
