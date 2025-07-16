import cupy
import numpy
import argparse
import cupynumeric as cpn
import legate.core as lg
from legate.core import align, broadcast, VariantCode, TaskContext
from legate.core.task import InputStore, OutputStore, task
from legate.core.types import complex64
from legate.timing import time

@task(variants = (VariantCode.CPU, VariantCode.GPU,),
      constraints = (align("dst", "src"),
                     broadcast("src", (1, 2))))
def fft2d_batched_gpu(ctx: TaskContext, dst: OutputStore, src: InputStore):
   xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
   cp_src = xp.asarray(src)
   cp_dst = xp.asarray(dst)
   # Apply 2D FFT across axes 1 and 2 for each batch
   cp_dst[:] = xp.fft.fftn(cp_src, axes=(1, 2))

parser = argparse.ArgumentParser(description = "Run FFT operation" )
parser.add_argument("--shape", type=str, default="128,256,256",
                    help="Shape of the array in the format D1,D2,D3")
args = parser.parse_args()
shape = tuple(map(int, args.shape.split(",")))

A_cpn = cpn.zeros(shape, dtype=cpn.complex64)
B_cpn = cpn.random.randint(1, 101, size=shape).astype(cpn.complex64)

rt = lg.get_legate_runtime()

#warm-up run
fft2d_batched_gpu(A_cpn, B_cpn)

rt.issue_execution_fence()
start = time()
fft2d_batched_gpu(A_cpn, B_cpn)
rt.issue_execution_fence()
end = time()

print(f"\nTime elapsed for batched fft: {(end - start)/1000:.6f} milliseconds")
