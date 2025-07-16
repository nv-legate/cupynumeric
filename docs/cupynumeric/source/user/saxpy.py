import cupy
import numpy
import argparse
import cupynumeric as cpn
import legate.core as lg
from legate.core import align, VariantCode, TaskContext
from legate.core.task import InputArray, OutputArray, task
from legate.timing import time

@task(variants = (VariantCode.CPU, VariantCode.GPU,),
      constraints = (align("x", "y"),
                     align("y", "z")))
def saxpy_task(ctx: TaskContext, x: InputArray, y: InputArray, z: OutputArray, a: float) -> None:
   xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
   x_local = xp.asarray(x)
   y_local = xp.asarray(y)
   z_local = xp.asarray(z)
   z_local[:] = a * x_local + y_local


parser = argparse.ArgumentParser(description="Run SAXPY operation.")
parser.add_argument("--size", type=int, default=1000, help="Size of input arrays")
args = parser.parse_args()
size = args.size

x_global = cpn.arange(size, dtype=cpn.float32)
y_global = cpn.ones(size, dtype=cpn.float32)
z_global = cpn.zeros(size, dtype=cpn.float32)

rt = lg.get_legate_runtime()

#warm-up run
saxpy_task(x_global, y_global, z_global, 2.0)

rt.issue_execution_fence()
start = time()
saxpy_task(x_global, y_global, z_global, 2.0)
rt.issue_execution_fence()
end = time()

print(f"\nTime elapsed for saxpy: {(end - start)/1000:.6f} milliseconds")
