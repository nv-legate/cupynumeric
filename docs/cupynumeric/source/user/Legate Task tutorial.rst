.. _legate-tasks:

====================================
Extend cuPyNumeric with Legate-tasks
====================================

This article assumes familiarity with the basic usage of cuPyNumeric.
Certain scenarios may benefit from building custom functions to integrate with
cuPyNumeric, for example: implementing novel research algorithms, or
leveraging routines from other libraries while still taking advantage of 
transparent scaling. In these cases, Legate tasks can be used to extend
cuPyNumeric by defining scale‑out functions that enable new algorithms
to run seamlessly across distributed resources.

**What is a Legate task?**
==========================

A Legate task is a Python function annotated with the ``@task``
decorator. This decorator informs the Legate runtime that a function
represents a parallel task that can be distributed across multiple CPUs,
GPUs, and nodes. This decorator enables high-performance parallelism
with a simple Python syntax.

Legate tasks are available automatically when using cuPyNumeric. No
additional components are needed after cuPyNumeric is installed. Please refer to the `Distributed Computing with cuPyNumeric`_ for cuPyNumeric installation.

.. _Distributed Computing with cuPyNumeric: https://github.com/NVIDIA/accelerated-computing-hub/blob/main/Accelerated_Python_User_Guide/notebooks/Chapter_11_Distributed_Computing_cuPyNumeric.ipynb

Usage
-----

.. code-block:: python

    @task(
    func: UserFunction | None = None,
    *,
    VariantList variants: VariantList = DEFAULT_VARIANT_LIST,
    constraints: Sequence[ConstraintProxy] | None = None,
    bool throws_exception: bool = False,
    bool register: bool = True,
    )

Parameters
----------

- **func (UserFunction)**, The function to invoke in the task.

- **variants (VariantList, optional)** – The list of variants for which
  func is applicable. Defaults to (VariantCode.CPU,), which means the
  task will run only on the CPU by default. To enable GPU execution, you
  must explicitly include VariantCode.GPU in the list of variants.

- **constraints (Sequence[ConstraintProxy], optional)** – The list of
  constraints which are to be applied to the arguments of func, if any.
  Controls how distributed-memory containers (Legate Logical
  Store/Array) are divided, aligned, or replicated for parallel
  execution across CPUs/GPUs. Defaults to no constraints.

  - *Align* - Ensures that the partitions of multiple arrays/stores are
    aligned along a given dimension.

  - *Broadcast* - Replicates data across all tasks instead of
    partitioning it.

  - Other Legate-supported partitioning constraints such as Image and
    Scale - See
    `Constraints`_ for more information.
.. _Constraints: https://docs.nvidia.com/legate/latest/api/python/generated/legate.core.task.task.html

- **throws_exception (bool, False)** – True if any variant of func
  throws an exception, False otherwise.

Requirements
------------

1. All arguments must have type-hints, without exception.

2. Arguments representing a piece of a Logical Store/Array must be given
   as either InputStore, OutputStore, InputArray, or OutputArray
   (`Arguments`_) cuPyNumeric arrays are backed by Legate Stores, so they are made
   available inside tasks as InputStores or OutputStores. The return
   value of the function must be exactly None. In the future, this
   restriction may be lifted.
.. _Arguments: https://docs.nvidia.com/legate/latest/api/python/generated/legate.core.task.InputStore.html

Quick Example
-------------

.. code-block:: python

        import cupy
        import numpy
        import cupynumeric as cpn
        from legate.core import StoreTarget, PhysicalArray, PhysicalStore, TaskContext, VariantCode
        from legate.core.task import task, InputArray, OutputArray
        
        @task( 
              variants = (VariantCode.CPU, VariantCode.GPU),
             )
        def foo_in_out(ctx: TaskContext, in_store: InputArray, out_store: OutputArray) -> None:
            xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy    
            in_store = xp.asarray(in_store)
            out_store = xp.asarray(out_store)
            out_store[:] = in_store[:]
        
        in_arr = cpn.array([1, 2, 3], dtype=cpn.int64)
        out_arr = cpn.zeros((3,), dtype=cpn.int64)
        foo_in_out(in_arr, out_arr)
        
        print(out_arr)


**Understanding this example -**

The code runs a function that replaces the contents of the
output array with the contents of the input array. cuPyNumeric arrays,
like in_arr and out_arr, are quite similar to NumPy arrays but are
backed by Legate stores for parallel execution across GPUs and CPUs.
When these arrays are passed into the foo_in_out task, they are
automatically converted into Legate-compatible objects such as
InputStore or OutputStore, depending on how they are used in the task.
Legate has built-in datatypes suitable for building richer
distributed data structures, e.g. nullable arrays, but in this tutorial
we exclusively use the simpler Legate Store class, which can only
represent a dense array, but is sufficient to back a cuPyNumeric
ndarray.

The @task decorator specifies both CPU and GPU variants using
VariantCode.CPU and VariantCode.GPU, indicating that the task can be
executed on either device depending on the available resources. Inside
the task, TaskContext provides access to the execution environment,
including inputs, outputs, and the execution target (CPU or GPU). The
method ctx.get_variant_kind() is used to determine the target device,
and based on this, the variable xp is set to either the CuPy for GPU
execution or NumPy for CPU execution. Using xp, the task creates views
of the task-local partitions of the Legate-backed global input and
output arrays as either CuPy or NumPy arrays.

.. _section-1:

SAXPY problem
=============

SAXPY(Single-Precision A·X Plus Y) is a fundamental linear algebra operation that computes the result
of the expression :math:`z = a * x + y`, where :math:`x` and :math:`y` are vectors and :math:`a` is a scalar.
It is a widely used example due to its simplicity and
computational relevance. This example demonstrates how to implement
SAXPY using Legate and cuPyNumeric, with emphasis on leveraging align
constraint for correct and efficient parallel execution. The align
constraint ensures that the input arrays x and y, as well as the output
z, are partitioned consistently. This means that matching elements from
each array are processed together on the same device. As a result, the
element-wise calculation a * x + y can run in parallel correctly,
without needing to move data between different parts of the system.

Main function
--------------

.. code-block:: python

    size = args.size
    
    x_global = cpn.arange(size, dtype=cpn.float32)
    y_global = cpn.ones(size, dtype=cpn.float32)
    z_global = cpn.zeros(size, dtype=cpn.float32)
      
    start = time()
    saxpy_task(x_global, y_global, z_global, 2.0)
    end = time()
    
    print(f"\nTime elapsed for saxpy: {(end - start)/1000:.6f} milliseconds")

For this example, three one-dimensional arrays of default size 1000 are
created. x_global contains values from 0 to 999, y_global is filled with
ones, and z_global is initialized with zeros to store the result. The
saxpy_task function is then called to compute the operation z_global =
2.0 \* x_global + y_global. We can change the size of the arrays
through the ``--size`` command-line argument when running the script.

Task function
-------------

.. code-block:: python

    @task(
       variants = (VariantCode.CPU, VariantCode.GPU,),
       constraints = (
           align("x", "y"),
           align("y", "z"),
       )
    )
    def saxpy_task(ctx: TaskContext, x: InputArray, y: InputArray, z: OutputArray, a: float) -> None:
       xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
       x_local = xp.asarray(x)
       y_local = xp.asarray(y)
       z_local = xp.asarray(z)
       z_local[:] = a * x_local + y_local

The constraint used is align. Align is used to ensure that X, Y , and Z
are partitioned in the same way. This is so that corresponding elements
live together on the same device. For example, imagine there are 4 GPUs,
and the problem size is 1000.

- GPU 1 gets the range 0–249

- GPU 2 gets the range 250–499

- GPU 3 gets the range 500–749

- GPU 4 gets the range 750–999

With the usage of align(“x”, “y”) and align(“y”, “z”) constraints, we
make sure that x[i], y[i], and z[i] are all assigned to the same gpu. If
we want to compute z[2], and GPU 1 handles the calculation for it, x[2]
and y[2] need to be handled in the same GPU in order to get the correct
answer. Given the align constraint, Legate will handle co-location of
corresponding elements across arrays, ensuring correctness.

The saxpy_task function uses TaskContext and its get_variant_kind()
method to determine the execution target (GPU or CPU) and accordingly
create views of the task-local data as NumPy or CuPy arrays. It then performs the SAXPY operation element-wise by computing
z_local[:] = a \* x_local + y_local. This task runs in parallel on the
available hardware (CPU or GPU), enabling efficient computation.

Complete module
---------------

Putting the pieces above together, here is a complete module that
can be run with the ``legate`` command line launcher:

.. code-block:: python

    import cupy
    import numpy
    import argparse
    import cupynumeric as cpn
    import legate.core as lg
    from legate.core import align, VariantCode, TaskContext
    from legate.core.task import InputArray, OutputArray, task
    from legate.timing import time
    
    @task(
       variants = (VariantCode.CPU, VariantCode.GPU,),
       constraints = (
           align("x", "y"),
           align("y", "z"),
       )
    )
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

.. _section-2:

Running on CPU and GPU 
----------------------

In order to run the program, use the legate launcher, and include any
flags necessary like ``--cpus``, ``--gpus``, and more. If you want to run
specifically only on CPU, you must include the flag ``--gpus 0``.
For a complete guide and additional options, see the `Legate documentation`_.

.. _Legate documentation: https://docs.nvidia.com/legate/latest/usage.html

The Legate runtime is used in the main function to control and
synchronize task execution. The get_legate_runtime() function returns
this runtime, which is used to issue commands like execution fences. In
this example, issue_execution_fence() is called before and after the
saxpy_task to ensure accurate time measurement. Since Legate tasks run
asynchronously by default, these fences make the program wait until all
previous tasks have finished, so the measured time reflects only the
actual task execution. This is a common pattern when precise timing,
synchronization, or ordered execution of asynchronous tasks is needed.

GPU initialization has a fixed setup time that can significantly affect
the runtime when processing small arrays. Using a large input (e.g., 100
million elements) ensures that the computation time outweighs the
startup overhead, giving more realistic timing results. Since the first
GPU run may include the setup overhead like compilation or memory
allocation, a warm-up pass helps eliminate these one-time costs from
performance measurements, ensuring more reliable results.

Let’s set the input array size to 100 million elements to better
evaluate the speedup from distributed computing with GPUs.


CPU execution 
~~~~~~~~~~~~~

To run with CPU, use the following command.

.. code-block:: sh

    legate --cpus 1 --gpus 0 ./saxpy.py --size 100000000

This produces the following output:

.. code-block:: text

    Time elapsed for saxpy: 146.303000 milliseconds

GPU execution 
~~~~~~~~~~~~~

To run with GPU, use the following command.

.. code-block:: sh

    legate --gpus 2 ./saxpy.py --size 100000000

This produces the following output:

.. code-block:: text

    Time elapsed for saxpy : 1.949000 milliseconds

Multi-Node execution 
~~~~~~~~~~~~~~~~~~~~
Refer to the Legate documentation on how to run on `multi-node`_. 
Here is an example performed on the `Perlmutter`_ supercomputer.

.. _multi-node: https://docs.nvidia.com/legate/latest/usage.html
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/

To run on multi-node, use the following command.

.. code-block:: sh

    legate --nodes 2 --launcher srun --gpus 4 --ranks-per-node 1 ./saxpy.py --size 100000000

This produces the following output:

.. code-block:: text

    Time elapsed for saxpy : 2.052000 milliseconds

Histogram problem
=================

Histogram computation involves counting how many data points fall into
specific bins, which is useful in tasks like statistical analysis and
image processing. In this example, Legate and CuPy are used to compute a
histogram in parallel, with a key focus on the broadcast constraint.
Broadcasting ensures that the histogram array is not split across
devices, allowing each GPU to access the full array and update it
safely. This prevents partial updates and ensures correct aggregation
using Legate’s reduction mechanism, enabling accurate and efficient
parallel histogram computation.

.. _main-function-1:

Main function
--------------

.. code-block:: python

    size = args.size
    NUM_BINS = 10
    
    data = cpn.random.randint(0, NUM_BINS, size=(size,), dtype=cpn.int32)
    hist = cpn.zeros((NUM_BINS,), dtype=cpn.int32)
        
    start = time()
    histogram_task(data, hist, NUM_BINS)
    end = time()    
    
    print(f"\nTime elapsed: {(end - start)/1000:.6f} milliseconds")

For this example, a one-dimensional array with a default size of 1000
elements is created, filled with random integers ranging from 0 to 9.
Alongside that, an empty hist array of length 10 is prepared to store
counts. The histogram_task function is then called to count the
frequency of each integer in the data array and accumulate these counts
into the hist array. We can change the size of the input array through
the ``--size`` command-line argument when running the script

Task function
-------------

.. code-block:: python

    @task(
        variants = (VariantCode.CPU, VariantCode.GPU,),
        constraints = (
             broadcast("hist"),
        ),
    )
    def histogram_task(ctx: TaskContext, data: InputArray, hist: ReductionArray[ADD], N_bins: int):
        xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
        data_local = xp.asarray(data)
        hist_local = xp.asarray(hist)    
    
        local_hist,_ = xp.histogram(data_local, bins= N_bins)
        hist_local[:] = hist_local + local_hist
    

The histogram_task function uses TaskContext and its get_variant_kind()
method to determine the execution target (GPU or CPU) and accordingly
create views of the task-local data as NumPy or CuPy arrays. It then
computes a local histogram on the partitioned chunk of data using the
specified number of bins and adds this local histogram results to the
global hist array using a reduction mechanism.

The task decorator specifies GPU execution via VariantCode.GPU. The
broadcast constraint on hist ensures that each GPU receives the full
hist array rather than a partitioned slice. This means each local hist
array has the same size as the global hist array. This allows every GPU
task to compute a local histogram on its data chunk and safely add its
results to the global hist array, ensuring correct accumulation of
counts from all distributed data partitions.

In this example, Legate will partition the data array automatically and
distribute chunks of it to different GPUs.

For example, imagine we have 4 GPUs, and the input data size is 1000.
Then:

- GPU 1 might get data[0–249]

- GPU 2 might get data[250–499]

- GPU 3 might get data[500–749]

- GPU 4 might get data[750–999]

Since hist is declared as a ReductionArray[ADD], Legate automatically
merges all the local histograms from all the GPUs by summing them
together at the end of the task execution. This produces the correct
global histogram as the final output.

In short, broadcast makes sure that the full hist array is available on
all devices, and the reduction mechanism handles merging the partial
results into a correct final output.

Complete module
---------------

Putting the pieces above together, here is a complete module that
can be run with the ``legate`` command line launcher:

.. code-block:: python

    import cupy
    import numpy
    import argparse
    import cupynumeric as cpn
    import legate.core as lg
    from legate.core import broadcast, VariantCode, TaskContext
    from legate.core.task import task, InputArray, ReductionArray, ADD
    from legate.timing import time   
    
    @task(
        variants = (VariantCode.CPU, VariantCode.GPU,),
        constraints = (
             broadcast("hist"),
        ),
    )
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

.. _running-on-cpu-and-gpu---guide-1:

Running on CPU and GPU
-----------------------

In order to run the program, use the legate launcher, and include any
flags necessary like ``--cpu``, ``--gpu``, and more. If you want to run
specifically only on CPU, you must add the flag ``--gpus 0``.
For a complete guide and additional options, see the `Legate documentation`_.

.. _Legate documentation: https://docs.nvidia.com/legate/latest/usage.html

Let’s set the size of the input array to 10 million. We’ll also include
a warm-up run before measuring execution time to ensure that one-time
setup costs (like memory allocation or kernel loading) don’t affect the
final performance results.

CPU execution 
~~~~~~~~~~~~~

To run with CPU, use the following command.

.. code-block:: sh

    legate --cpus 1 --gpus 0 ./histogram.py --size 10000000

This produces the following output:

.. code-block:: text

    Time elapsed for histogram: 123.041000 milliseconds

GPU execution 
~~~~~~~~~~~~~

To run with GPU, use the following command.

.. code-block:: sh

    legate --gpus 2 ./histogram.py --size 10000000

This produces the following output:

.. code-block:: text

    Time elapsed for histogram : 3.960000 milliseconds

Multi-Node execution 
~~~~~~~~~~~~~
Refer to the Legate documentation on how to run on `multi-node`_. 
Here is an example performed on the `Perlmutter`_ supercomputer.

.. _multi-node: https://docs.nvidia.com/legate/latest/usage.html
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/

To run with Multi-Node, use the following command.

.. code-block:: sh

    legate --nodes 2 --launcher srun --gpus 4 --ranks-per-node 1 ./histogram.py --size 10000000

This produces the following output:

.. code-block:: text

    Time elapsed for histogram : 4.266000 milliseconds

Simple Matrix Multiplication problem
====================================

We multiply two matrices A (shape (m, k)) and B (shape (k, n)) to
produce C (shape (m, n)), using 3D tiling to enable parallel execution
over blocks of the matrix. This example will introduce basic matrix
multiplication using Legate and CuPy. It emphasizes 3D tiling and
reduction privileges, teaching how to structure tasks for parallel
execution by promoting arrays for consistent partitioning and aligning
the inputs and outputs, and then safely reducing partial results.

.. _main-function-2:

Main function
-------------

.. code-block:: python

    m = args.m
    k = args.k
    n = args.n
    
    A_cpn = cpn.random.randint(1, 101, size=(m, k))
    B_cpn = cpn.random.randint(1, 101, size=(k, n))
    C_cpn = cpn.zeros((m, n))
    
    A_cpn = cpn.broadcast_to(A_cpn[:, :, cpn.newaxis], (m, k, n)) # (m,k,1) -> (m,k,n)
    # The (m, k, n) allows legate to align these stores, so we need the same dimensions
    B_cpn = cpn.broadcast_to(B_cpn[cpn.newaxis, :, :], (m, k, n))
    C_cpn = cpn.broadcast_to(C_cpn[:, cpn.newaxis, :], (m, k, n))
    
    start = time()
    matmul_task(C_cpn, A_cpn, B_cpn)
    end = time()
    
    print(f"\nTime elapsed for matmul: {(end - start)/1000:.6f} seconds")

The important things that this code does are:

- Defines the dimensions of the matrices using the values of m, k, and
  n, which are obtained from command-line arguments.

- Initializes input matrices A and B with random integers and output
  matrix C with zeros.

- Ensures that the inner dimensions of A and B match, which is required
  for valid matrix multiplication.

- Each matrix is promoted to 3D by adding an extra dimension. Because,
  in order to correctly partition the computation, matrices A, B, and C
  should be partitioned in an aligned way. Given the dimension of these
  matrices are A[m,k], B[k,n], and C[m,n], they cannot be aligned
  directly. By adding one dimension to each of them, the dimensions
  become A[m, k, n], B[m, k, n] and C[m, k, n]. The three arrays can now
  be aligned along m, k, and n dimensions, producing the required
  alignment for performing matrix multiplication.

Task function
-------------

.. code-block:: python

    @task(
       variants = (VariantCode.CPU,VariantCode.GPU,),
       constraints = (
          align("C", "A"),
          align("C", "B"),
          ),
       )

- **Variants**: The task can run on either CPU or GPU, depending on the
  available resources at runtime.

- **align(“C”, “A”) / align(“C”, “B”)** : Aligns partitions of A,B, and
  C so that each task instance gets matching chunks of data. If align is
  not used, partitions could be mismatched, leading to errors or even
  incorrect results. For example, if GPU 0 is given block (0:25, 0:38)
  of A and block (0:38, 0:50) of B, then it should be given the correct
  block (0:25, 0:50) of C to update. For example, after promotion to A(m,k,n), B(m,k,n), C(m,k,n), the
  align constraint could produce the partitioning A(0:m/2, 0:k/2,
  0:n/2), B(0:m/2, 0:k/2, 0:n/2), C(0:m/2, 0:k/2, 0:n/2).

.. code-block:: python

    def matmul_task(ctx: TaskContext, C: ReductionArray[ADD], A: InputArray, B: InputArray) -> None:
       xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
       C = xp.asarray(C)[:, 0, :]
       A = xp.asarray(A)[:, :, 0]
       B = xp.asarray(B)[0, :, :]
    
       C += xp.matmul(A,B)


The matmul_task function uses TaskContext to determine if it’s running
on a CPU or GPU, setting xp to NumPy or CuPy accordingly. It then
converts the received task-local data to array views using xp.asarray().
The extra broadcasted dimension introduced earlier is then sliced away
to recover the original 2D shapes of the matrices. Finally performs the
matrix multiplication and accumulates the result into C.

Complete module
---------------

Putting the pieces above together, here is a complete module that
can be run with the ``legate`` command line launcher:

.. code-block:: python

    import cupy
    import numpy
    import argparse
    import cupynumeric as cpn
    import legate.core as lg
    from legate.core import VariantCode, align, TaskContext
    from legate.core.task import task, InputArray, ReductionArray, ADD
    from legate.timing import time
    
    @task(
       variants = (VariantCode.CPU,VariantCode.GPU,),
       constraints = (
          align("C", "A"),
          align("C", "B"),
          ),
       )
    def matmul_task(ctx: TaskContext, C: ReductionArray[ADD], A: InputArray, B: InputArray) -> None:
       xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
       C = xp.asarray(C)[:, 0, :]
       A = xp.asarray(A)[:, :, 0]
       B = xp.asarray(B)[0, :, :]
    
       C += xp.matmul(A,B)
    
    parser= argparse.ArgumentParser(description ="Run Matrix multiplication operation")
    parser.add_argument("-m", type=int, default=50, help="Number of rows in matrix A and C")
    parser.add_argument("-k", type=int, default=75, help="Number of columns in A / rows in B")
    parser.add_argument("-n", type=int, default=100, help="Number of columns in matrix B and C")
    args=parser.parse_args()
    
    m = args.m
    k = args.k
    n = args.n
    
    A_cpn = cpn.random.randint(1, 101, size=(m, k))
    B_cpn = cpn.random.randint(1, 101, size=(k, n))
    C_cpn = cpn.zeros((m, n))
    
    A_cpn = cpn.broadcast_to(A_cpn[:, :, cpn.newaxis], (m, k, n)) #(m,k,1) -> (m,k,n)
    # The (m, k, n) allows legate to align these stores, so we need the same dimensions
    B_cpn = cpn.broadcast_to(B_cpn[cpn.newaxis, :, :], (m, k, n))
    C_cpn = cpn.broadcast_to(C_cpn[:, cpn.newaxis, :], (m, k, n))
    
    rt = lg.get_legate_runtime()
    
    #warm-up run
    matmul_task(C_cpn, A_cpn, B_cpn)
    
    rt.issue_execution_fence()
    start = time()
    matmul_task(C_cpn, A_cpn, B_cpn)
    rt.issue_execution_fence()
    end = time()
    
    print(f"\nTime elapsed for matmul: {(end - start)/1000:.6f} seconds")

.. _section-3:

.. _running-on-cpu-and-gpu---guide-2:

Running on CPU and GPU 
----------------------



In order to run the program, use the legate launcher, and include any
flags necessary like ``--cpu``, ``--gpu``, and more. If you want to run
specifically only on CPU, you must add the flag ``--gpus 0``.
For a complete guide and additional options, see the `Legate documentation`_.

.. _Legate documentation: https://docs.nvidia.com/legate/latest/usage.html

Let's increase the size of the matrix by setting m = 1000, k = 1000, and
n = 1000. We’ll also include a warm-up run before measuring execution
time to ensure that one-time setup costs (like memory allocation or
kernel loading) don’t affect the final performance results.

CPU execution 
~~~~~~~~~~~~~

To run with CPU, use the following command.

.. code-block:: sh

    legate --cpus 1 --gpus 0 ./matmul.py -m 1000 -k 1000 -n 1000

This produces the following output:

.. code-block:: text

    Time elapsed for matmul: 902.748000 milliseconds

GPU execution 
~~~~~~~~~~~~~
To run with GPU, use the following command.

.. code-block:: sh

    legate --gpus 2 ./matmul.py -m 1000 -k 1000 -n 1000

This produces the following output:

.. code-block:: text

    Time elapsed for matmul: 3.076000 milliseconds

Multi-Node execution 
~~~~~~~~~~~~~
Refer to the Legate documentation on how to run on `multi-node`_. 
Here is an example performed on the `Perlmutter`_ supercomputer.

.. _multi-node: https://docs.nvidia.com/legate/latest/usage.html
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/

To run with Multi-Node, use the following command.

.. code-block:: sh

    legate --nodes 2 --launcher srun --gpus 4 --ranks-per-node 1 ./matmul.py -m 1000 -k 1000 -n 1000

This produces the following output:

.. code-block:: text

    Time elapsed for matmul: 3.226000 milliseconds

Fast Fourier Transform problem
==============================

The Fast Fourier Transform (FFT) is an algorithm which is used to
compute the discrete fourier transform of a sequence. It is used to help
break down a complex signal like sound and images, which is instrumental
in image processing, medical imaging, and more. This example
demonstrates how to use Legate and CuPy to perform a batched 2D Fast
Fourier Transform. It highlights how to use align and broadcasting
constraints to control partitioning. Alignment makes sure the input and
output chunks line up correctly while broadcasting keeps part of data
unpartitioned.

.. _main-function-3:

Main function
-------------

.. code-block:: python

    shape = tuple(map(int, args.shape.split(","))) 
    
    A_cpn = cpn.zeros(shape, dtype=cpn.complex64)
    B_cpn = cpn.random.randint(1, 101, size=shape).astype(cpn.complex64)
    
    start = time()
    fft2d_batched_gpu(A_cpn, B_cpn)
    end = time()
    
    print(f"\nTime elapsed for batched fft: {(end - start)/1000:.6f} milliseconds")

For demonstration purposes, a default shape of (128, 256, 256) is used,
representing a batch of 128 two dimensional matrices. Using this shape,
cuPyNumeric arrays are generated, and cast to complex64. B_cpn contains
random values, while A_cpn contains zeros. The fft2d_batched_gpu task is
then launched, by using these two cuPyNumeric arrays. We can change the
shape of the input arrays using the ``--shape`` command-line argument when
running the script

Task function
-------------

.. code-block:: python

    @task(
       variants = (VariantCode.CPU, VariantCode.GPU,),
       constraints = (
           align("dst", "src"),
           broadcast("src", (1, 2)),
       ),
    )
    def fft2d_batched_gpu(ctx: TaskContext, dst: OutputStore, src: InputStore):
       xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
       cp_src = xp.asarray(src)
       cp_dst = xp.asarray(dst)
       # Apply 2D FFT across axes 1 and 2 for each batch
       cp_dst[:] = xp.fft.fftn(cp_src, axes=(1, 2))

The fft2d_batched_gpu function uses TaskContext to detect execution on
GPU and sets xp to CuPy accordingly. It then converts the src and dst
arrays into CuPy arrays as views without copying. Afterwards, it applies
2D FFT for each batch independently. As for the task decorator, it has a
VariantCode.GPU, which means this task is implemented for GPU execution.
As for the align constraint, it ensures that the output and input arrays
are partitioned the same way. This ensures that the corresponding chunks
are processed together. The other constraint broadcast makes sure the
source array is not partitioned along axes 1 and 2. This is important as
this allows each GPU to get full slices along these axes, and makes sure
that you are able to split work along the batch dimension (axis 0).

For example, let's imagine the shape of src is (128, 256, 256). This
means there are 128 independent 2D images, each of size 256×256. If
broadcast is not used, then it might get partitioned like this.

- GPU 0: slices src[0:64, 0:128, :]

- GPU 1: slices src[64:128, 128:256, :]

Now each GPU has partial rows from multiple images, which may lead to
incorrect FFT computations.

But with broadcast("src", (1, 2)), this ensures Legate will partition
only along axis 0, so each GPU gets a full 2D matrix per batch.

- GPU 0: src[0:64, :, :] → 64 full images

- GPU 1: src[64:128, :, :] → remaining 64 full images

Complete module
---------------

Putting the pieces above together, here is a complete module that
can be run with the ``legate`` command line launcher:

.. code-block:: python

    import cupy
    import numpy
    import argparse
    import cupynumeric as cpn
    import legate.core as lg
    from legate.core import align, broadcast, VariantCode, TaskContext
    from legate.core.task import InputStore, OutputStore, task
    from legate.core.types import complex64
    from legate.timing import time
    
    @task(
       variants = (VariantCode.CPU, VariantCode.GPU,),
       constraints = (
           align("dst", "src"),
           broadcast("src", (1, 2)),
       ),
    )
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


.. _running-on-cpu-and-gpu---guide-3:

Running on CPU and GPU 
----------------------

In order to run the program, use the legate launcher, and include any
flags necessary like ``--cpu``, ``--gpu``, and more. If you want to run
specifically only on CPU, you must add the flag ``--gpus 0``.
For a complete guide and additional options, see the `Legate documentation`_.

.. _Legate documentation: https://docs.nvidia.com/legate/latest/usage.html


CPU execution 
~~~~~~~~~~~~~
To run with CPU, use the following command.

.. code-block:: sh

    legate --cpus 1 --gpus 0 ./fft.py

This produces the following output:

.. code-block:: text

    Time elapsed for fft: 173.655000 milliseconds

GPU execution 
~~~~~~~~~~~~~
To run with GPU, use the following command.

.. code-block:: sh

    legate --gpus 2 ./fft.py

This produces the following output:

.. code-block:: text

    Time elapsed for fft: 16.153000 milliseconds

Multi-Node execution 
~~~~~~~~~~~~~~~~~~~~
Refer to the Legate documentation on how to run on `multi-node`_. 
Here is an example performed on the `Perlmutter`_ supercomputer.


.. _multi-node: https://docs.nvidia.com/legate/latest/usage.html
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/

To run with Multi-Node, use the following command.

.. code-block:: sh

    legate --nodes 2 --launcher srun --gpus 4 --ranks-per-node 1 ./fft.py

This produces the following output:

.. code-block:: text

    Time elapsed for fft: 16.443000 milliseconds
