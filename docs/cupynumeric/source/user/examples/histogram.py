import cupy
import numpy
import argparse
import cupynumeric as cpn
import legate.core as lg
from legate.core import broadcast, VariantCode, TaskContext
from legate.core.task import task, InputArray, ReductionArray, ADD
from legate.timing import time

@task(variants = (VariantCode.CPU, VariantCode.GPU,),
      constraints = (broadcast("hist"),))
def histogram_task(ctx: TaskContext, data: InputArray, hist: ReductionArray[ADD], N_bins: int):
    xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
    data_local = xp.asarray(data)
    hist_local = xp.asarray(hist)

    local_hist,_ = xp.histogram(data_local, bins= N_bins)
    hist_local[:] = hist_local + local_hist

parser = argparse.ArgumentParser(description="Run Histogram operation.")
parser.add_argument("--size", type=int, default=1000, help="Size of input arrays")
args = parser.parse_args()

size = args.size
NUM_BINS = 10

data = cpn.random.randint(0, NUM_BINS, size=(size,), dtype=cpn.int32)
hist = cpn.zeros((NUM_BINS,), dtype=cpn.int32)

rt = lg.get_legate_runtime()

#warm-up run
histogram_task(data, hist, NUM_BINS)

rt.issue_execution_fence()
start = time()
histogram_task(data, hist, NUM_BINS)
rt.issue_execution_fence()
end = time()

print(f"\nTime elapsed for histogram : {(end - start)/1000:.6f} milliseconds")
