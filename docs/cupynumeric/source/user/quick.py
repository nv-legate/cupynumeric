import cupy
import numpy
import cupynumeric as cpn
from legate.core import StoreTarget, PhysicalArray, PhysicalStore, TaskContext, VariantCode
from legate.core.task import task, InputArray, OutputArray

@task(variants = (VariantCode.CPU, VariantCode.GPU))
def foo_in_out(ctx: TaskContext, in_store: InputArray, out_store: OutputArray) -> None:
    xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy  # select CuPy or NumPy depending on variant.
    in_store = xp.asarray(in_store)
    out_store = xp.asarray(out_store)
    out_store[:] = in_store[:]

in_arr = cpn.array([1, 2, 3], dtype=cpn.int64)
out_arr = cpn.zeros((3,), dtype=cpn.int64)
foo_in_out(in_arr, out_arr)

print(out_arr)
