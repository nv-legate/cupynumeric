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

What is a Legate task?
======================

A Legate task is a Python function annotated with the ``@legate.core.task.task``
decorator. This decorator informs the Legate runtime that a function
represents a parallel task that can be distributed across multiple CPUs,
GPUs, and nodes. This decorator enables high-performance parallelism
with a simple Python syntax.

Legate tasks are available automatically when using cuPyNumeric. No
additional components are needed after cuPyNumeric is installed.
Please refer to the `Distributed Computing with cuPyNumeric`_ for cuPyNumeric installation.

.. _Distributed Computing with cuPyNumeric: https://github.com/NVIDIA/accelerated-computing-hub/blob/main/Accelerated_Python_User_Guide/notebooks/Chapter_11_Distributed_Computing_cuPyNumeric.ipynb

Quick example
-------------
Here is an example of defining and invoking a Legate task with a custom function.

.. literalinclude:: examples/quick.py
   :language: python

Understanding this example
~~~~~~~~~~~~~~~~~~~~~~~~~~

The code runs a function that replaces the contents of the
output array with the contents of the input array. cuPyNumeric arrays,
like ``in_arr`` and ``out_arr`` are similar to NumPy arrays but are
backed by Legate stores for parallel execution across CPUs and GPUs.
When these arrays are passed into the ``foo_in_out`` task, they are
automatically converted into Legate-compatible objects such as
``InputStore`` or ``OutputStore``, depending on how they are used in the task.
In Legate, arguments representing a portion of a logical store or array must be
specified as one of the following types: ``InputStore``, ``OutputStore``, ``InputArray``, or
``OutputArray`` (see Arguments_).
Legate has built-in datatypes suitable for building richer
distributed data structures, e.g. nullable arrays, but in this tutorial
we exclusively use the simpler Legate Store class, which can only
represent a dense array. This is sufficient to back a cuPyNumeric
ndarray.

The ``@task`` decorator declares that this task has two variants: ``VariantCode.CPU`` for CPU execution
and ``VariantCode.GPU`` for GPU execution. This means the task can run
on either device, depending on which resources the user selects at runtime.
For more details about Legate task behavior, see legate_task_. Inside
the task, ``TaskContext`` provides access to the execution environment,
including inputs, outputs, and the execution target (CPU or GPU). The
method ``ctx.get_variant_kind()`` is used to determine the target device,
and based on this, the variable ``xp`` is set to either the CuPy for GPU
execution or NumPy for CPU execution. Using ``xp``, the task creates views
of the task-local partitions of the Legate-backed global input and
output arrays as either CuPy or NumPy arrays.

.. _Arguments: https://docs.nvidia.com/legate/latest/api/python/generated/legate.core.task.InputStore.html
.. _legate_task: https://docs.nvidia.com/legate/25.07/api/python/generated/legate.core.task.task.html

SAXPY problem
=============

SAXPY (Single-Precision A·X Plus Y) is a fundamental linear algebra operation that computes the result
of the expression :math:`z = a * x + y`, where :math:`x` and :math:`y` are vectors and :math:`a` is a scalar.
It is a widely used example due to its simplicity and
computational relevance. This example demonstrates how to implement
SAXPY using Legate and cuPyNumeric, with emphasis on leveraging ``align``
constraint for correct and efficient parallel execution. The ``align``
constraint ensures that the input arrays ``x`` and ``y``, as well as the output
``z``, are partitioned consistently. This means that matching elements from
each array are processed together on the same device. As a result, the
element-wise calculation ``a * x + y`` can run in parallel correctly,
without needing to move data between different parts of the system.

Main function
--------------
Let’s take a look at the input and output parameters for this SAXPY example.

.. literalinclude:: examples/saxpy.py
   :language: python
   :lines: 24-29,36,37,39-41

For this example, three one-dimensional arrays of default size 1000 are
created. ``x_global`` contains values from 0 to 999, ``y_global`` is filled with
ones, and ``z_global`` is initialized with zeros to store the result. The
saxpy_task function is then called to compute the operation ``z_global = 2.0 * x_global + y_global``. We can change the size of the arrays
through the ``--size`` command-line argument when running the script.

Task function
-------------
The following example shows how to define a task function that performs the SAXPY operation.

.. literalinclude:: examples/saxpy.py
   :language: python
   :lines: 9-19

The constraint used is ``align``, it is used to ensure that ``x``, ``y`` , and ``z``
are partitioned in the same way. This is so that corresponding elements
live together on the same device. For example, imagine there are 4 GPUs,
and the problem size is 1000.

- GPU 1 gets the range 0–249

- GPU 2 gets the range 250–499

- GPU 3 gets the range 500–749

- GPU 4 gets the range 750–999

With the usage of ``align(“x”, “y”)`` and ``align(“y”, “z”)`` constraints, we ensure that 
the chunks of ``x``, ``y``, and ``z`` are partitioned in the same way across devices. 
This means that each chunk (or window) of data is assigned to the same task, so corresponding 
elements within each chunk are colocated on the same GPU during execution. For example, 
if a chunk covering indices 0–9 is assigned to GPU 1, all data for x[0:10], y[0:10], and z[0:10] 
will be processed together on that device. For further details, see the `Legate alignment constraints documentation`_.

.. _Legate alignment constraints documentation: https://docs.nvidia.com/legate/25.07/api/python/generated/legate.core.task.task.html

The ``saxpy_task`` function uses ``TaskContext`` and its ``get_variant_kind()``
method to determine the execution target (GPU or CPU) and accordingly
create views of the task-local data as NumPy or CuPy arrays. It then performs the SAXPY operation element-wise by computing
``z_local[:] = a * x_local + y_local``. This task runs in parallel on the
available hardware (CPU or GPU), enabling efficient computation.

Complete module
---------------
Putting the pieces above together, here is a complete module that
can be run with the ``legate`` command line launcher:

.. literalinclude:: examples/saxpy.py
   :language: python

The Legate runtime is used in the main function to control and
synchronize task execution. The ``get_legate_runtime()`` function returns
this runtime, which is used to issue commands like execution fences. In
this example, ``issue_execution_fence()`` is called before and after the
``saxpy_task`` to ensure accurate time measurement. Since Legate tasks run
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


Running on CPU and GPU 
----------------------

In order to run the program, use the legate launcher, and include any
flags necessary like ``--cpus``, ``--gpus``, and more. If you want to run
specifically only on CPU, you must include the flag ``--gpus 0``.
For a complete guide and additional options, see the `Legate documentation`_.

.. _Legate documentation: https://docs.nvidia.com/legate/latest/usage.html

Let’s set the input array size to 10 million elements to better
evaluate the speedup from distributed computing with GPUs.


CPU execution 
~~~~~~~~~~~~~

To run with CPU, use the following command.

.. code-block:: sh

    legate --cpus 1 --gpus 0 ./saxpy.py --size 10000000

This produces the following output:

.. code-block:: text

    Time elapsed for saxpy: 14.303000 milliseconds

GPU execution 
~~~~~~~~~~~~~

To run with GPU, use the following command.

.. code-block:: sh

    legate --gpus 2 ./saxpy.py --size 10000000

This produces the following output:

.. code-block:: text

    Time elapsed for saxpy : 1.769000 milliseconds

Multi-Node execution 
~~~~~~~~~~~~~~~~~~~~
Refer to the Legate documentation on how to run on `multi-node`_. 
Here is an example performed on the `Perlmutter`_ supercomputer.

.. _multi-node: https://docs.nvidia.com/legate/latest/usage.html
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/

To run on multi-node, use the following command.

.. code-block:: sh

    legate --nodes 2 --launcher srun --gpus 4 --ranks-per-node 1 ./saxpy.py --size 10000000

This produces the following output:

.. code-block:: text

    Time elapsed for saxpy : 2.052000 milliseconds

Histogram problem
=================

Histogram computation involves counting how many data points fall into
specific bins, This is useful in tasks like statistical analysis and
image processing. In this example, Legate and cuPyNumeric are used to compute a
histogram in parallel, with a key focus on the ``broadcast`` constraint.
Broadcasting ensures that the histogram array is not split across
devices, allowing each GPU to access the full array and update it
safely. This prevents partial updates and ensures correct aggregation
using Legate’s reduction mechanism, enabling accurate and efficient
parallel histogram computation.

.. _main-function-1:

Main function
--------------
Let’s take a quick look at the input and output parameters for this histogram example.

.. literalinclude:: examples/histogram.py
   :language: python
   :lines: 24-29,36,37,39-41

For this example, a one-dimensional array with a default size of 1000
elements is created, filled with random integers ranging from 0 to 9.
Alongside that, an empty ``hist`` array of length 10 is prepared to store
counts. The ``histogram_task`` function is then called to count the
frequency of each integer in the ``data`` array and accumulate these counts
into the ``hist`` array. We can change the size of the input array through
the ``--size`` command-line argument when running the script

Task function
-------------
The following example defines a histogram task function that computes a local histogram and accumulates the results into a global ``hist`` array using a reduction.

.. literalinclude:: examples/histogram.py
   :language: python
   :lines: 9-19

The ``histogram_task`` function uses ``TaskContext`` and its ``get_variant_kind()``
method to determine the execution target (GPU or CPU) and accordingly
create views of the task-local data as NumPy or CuPy arrays. It then
computes a local histogram on the partitioned chunk of data using the
specified number of bins and adds this local histogram results to the
global ``hist`` array using a reduction mechanism.

The task decorator specifies GPU execution via ``VariantCode.GPU``. The
``broadcast`` constraint on ``hist`` ensures that each GPU receives the full
``hist`` array rather than a partitioned slice. This means each local ``hist``
array has the same size as the global ``hist`` array. This allows every GPU
task to compute a local histogram on its data chunk and safely add its
results to the global ``hist`` array, ensuring correct accumulation of
counts from all distributed ``data`` partitions.

In this example, Legate will partition the ``data`` array automatically and
distribute chunks of it to different GPUs.

For example, imagine we have 4 GPUs, and the input data size is 1000.
Then:

- GPU 1 might get data[0–249]

- GPU 2 might get data[250–499]

- GPU 3 might get data[500–749]

- GPU 4 might get data[750–999]

Since hist is declared as a ``ReductionArray[ADD]``, Legate automatically
merges all the local histograms from all the GPUs by summing them
together at the end of the task execution. This produces the correct
global histogram as the final output.

In short, ``broadcast`` makes sure that the full ``hist`` array is available on
all devices, and the reduction mechanism handles merging the partial
results into a correct final output.

Complete module
---------------

Putting the pieces above together, here is a complete module that
can be run with the ``legate`` command line launcher:

.. literalinclude:: examples/histogram.py
   :language: python

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

    Time elapsed for histogram : 3.790000 milliseconds

Multi-Node execution 
~~~~~~~~~~~~~~~~~~~~
Refer to the Legate documentation on how to run on `multi-node`_. 
Here is an example performed on the `Perlmutter`_ supercomputer.

.. _multi-node: https://docs.nvidia.com/legate/latest/usage.html
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/

To run with Multi-Node, use the following command.

.. code-block:: sh

    legate --nodes 2 --launcher srun --gpus 4 --ranks-per-node 1 ./histogram.py --size 10000000

This produces the following output:

.. code-block:: text

    Time elapsed for histogram : 3.716000 milliseconds

Simple matrix multiplication problem
====================================

We multiply two matrices ``A (shape (m, k))`` and ``B (shape (k, n))`` to
produce ``C (shape (m, n))``, using 3D tiling to enable parallel execution
over blocks of the matrix. This example will introduce basic matrix
multiplication using Legate and cuPyNumeric. It emphasizes 3D tiling and
reduction privileges, teaching how to structure tasks for parallel
execution by promoting arrays for consistent partitioning and aligning
the inputs and outputs, and then safely reducing partial results.


Main function
-------------
The following main function prepares input matrices with proper broadcasting, executes the matrix multiplication task, and measures the computation time.

.. literalinclude:: examples/matmul.py
   :language: python
   :lines: 26-39,46,47,49-51

The important things that this code does are:

- Defines the dimensions of the matrices using the values of m, k, and
  n, which are obtained from command-line arguments.

- Initializes input matrices A and B with random integers and output
  matrix C with zeros.

- Ensures that the inner dimensions of A and B match, which is required
  for valid matrix multiplication.

- Each matrix is promoted to 3D by adding an extra dimension. Because,
  in order to correctly partition the computation, matrices ``A``, ``B``, and ``C``
  should be partitioned in an aligned way. Given the dimension of these
  matrices are ``A[m,k]``, ``B[k,n]``, and ``C[m,n]``, they cannot be aligned
  directly. By adding one dimension to each of them, the dimensions
  become ``A[m, k, n]``, ``B[m, k, n]`` and ``C[m, k, n]``. The three arrays can now
  be aligned along ``m``, ``k``, and ``n`` dimensions, producing the required
  alignment for performing matrix multiplication.

Task function
-------------
The following example shows a task function that performs matrix multiplication with aligned partitions across input and output arrays.

.. literalinclude:: examples/matmul.py
   :language: python
   :lines: 9-20

.. code-block:: python

    @task(variants = (VariantCode.CPU,VariantCode.GPU,),
          constraints = (align("C", "A"),
                         align("C", "B")))
    def matmul_task(ctx: TaskContext, C: ReductionArray[ADD], A: InputArray, B: InputArray) -> None:
       xp = cupy if ctx.get_variant_kind() == VariantCode.GPU else numpy
       C = xp.asarray(C)[:, 0, :]
       A = xp.asarray(A)[:, :, 0]
       B = xp.asarray(B)[0, :, :]
    
       C += xp.matmul(A,B)

The task can run on either CPU or GPU, depending on the available resources at runtime.
The alignment constraints ``align(“C”, “A”)`` and ``align(“C”, “B”)`` ensures that partitions of ``A``, ``B``, and
``C`` so that each task instance gets matching chunks of data. If ``align`` is
not used, partitions could be mismatched, leading to errors or even
incorrect results. For example, if GPU 0 is given block (0:25, 0:38)
of ``A`` and block (0:38, 0:50) of ``B``, then it should be given the correct
block (0:25, 0:50) of ``C`` to update. For example, after promotion to ``A(m,k,n)``, ``B(m,k,n)``, ``C(m,k,n)``, the
``align`` constraint could produce the partitioning ``A(0:m/2, 0:k/2,
0:n/2)``, ``B(0:m/2, 0:k/2, 0:n/2)``, ``C(0:m/2, 0:k/2, 0:n/2)``.


The ``matmul_task`` function uses ``TaskContext`` to determine if it’s running
on a CPU or GPU, setting ``xp`` to NumPy or CuPy accordingly. It then
converts the received task-local data to array views using ``xp.asarray()``.
The extra broadcasted dimension introduced earlier is then sliced away
to recover the original 2D shapes of the matrices. Finally performs the
matrix multiplication and accumulates the result into ``C``.

Complete module
---------------

Putting the pieces above together, here is a complete module that
can be run with the ``legate`` command line launcher:

.. literalinclude:: examples/matmul.py
   :language: python

Running on CPU and GPU 
----------------------

In order to run the program, use the legate launcher, and include any
flags necessary like ``--cpu``, ``--gpu``, and more. If you want to run
specifically only on CPU, you must add the flag ``--gpus 0``.
For a complete guide and additional options, see the `Legate documentation`_.

.. _Legate documentation: https://docs.nvidia.com/legate/latest/usage.html

Let's increase the size of the matrix by setting ``m`` = 1000, ``k`` = 1000, and
``n`` = 1000. We’ll also include a warm-up run before measuring execution
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

    Time elapsed for matmul: 2.776000 milliseconds

Multi-Node execution 
~~~~~~~~~~~~~~~~~~~~
Refer to the Legate documentation on how to run on `multi-node`_. 
Here is an example performed on the `Perlmutter`_ supercomputer.

.. _multi-node: https://docs.nvidia.com/legate/latest/usage.html
.. _Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/

To run with Multi-Node, use the following command.

.. code-block:: sh

    legate --nodes 2 --launcher srun --gpus 4 --ranks-per-node 1 ./matmul.py -m 1000 -k 1000 -n 1000

This produces the following output:

.. code-block:: text

    Time elapsed for matmul: 2.926000 milliseconds

Fast Fourier Transform problem
==============================

The Fast Fourier Transform (FFT) is an algorithm which is used to
compute the discrete fourier transform of a sequence. It is used to help
break down a complex signal like sound and images, which is instrumental
in image processing, medical imaging, and more. This example
demonstrates how to use Legate and cuPyNumeric to perform a batched 2D Fast
Fourier Transform. It highlights how to use ``align`` and ``broadcast``
constraints to control partitioning. Alignment makes sure the input and
output chunks line up correctly while broadcasting keeps part of data
unpartitioned.


Main function
-------------

The following code block initializes inputs and performs a GPU-accelerated batched 2D Fast Fourier Transform.

.. literalinclude:: examples/fft.py
   :language: python
   :lines: 25-29,36,37,39-41

For demonstration purposes, a default shape of (128, 256, 256) is used,
representing a batch of 128 two dimensional matrices. Using this shape,
cuPyNumeric arrays are generated, and cast to complex64. ``B_cpn`` contains
random values, while ``A_cpn`` contains zeros. The ``fft2d_batched_gpu`` task is
then launched, by using these two cuPyNumeric arrays. We can change the
shape of the input arrays using the ``--shape`` command-line argument when
running the script

Task function
-------------
The following example defines a task that computes a batched 2D FFT over input data using ``align`` and ``broadcast`` constraints.

.. literalinclude:: examples/fft.py
   :language: python
   :lines: 10-20

The ``fft2d_batched_gpu`` function uses ``TaskContext`` to detect execution on
GPU and sets ``xp`` to CuPy accordingly. It then converts the ``src`` and ``dst``
arrays into CuPy arrays as views without copying. Afterwards, it applies
2D FFT for each batch independently. As for the task decorator, it has a
``VariantCode.GPU``, which means this task is implemented for GPU execution.
As for the ``align`` constraint, it ensures that the output and input arrays
are partitioned the same way. This ensures that the corresponding chunks
are processed together. The other constraint ``broadcast`` makes sure the
source array is not partitioned along axes 1 and 2. This is important as
it allows each GPU to get full slices along these axes, and makes sure
that you are able to split work along the batch dimension (axis 0).

For example, let's imagine the shape of ``src`` is (128, 256, 256). This
means there are 128 independent 2D images, each of size 256×256. If
``broadcast`` is not used, then it might get partitioned like this.

- GPU 0: slices src[0:64, 0:128, :]

- GPU 1: slices src[64:128, 128:256, :]

Now each GPU has partial rows from multiple images, which may lead to
incorrect FFT computations.

But with ``broadcast("src", (1, 2))``, this ensures Legate will partition
only along axis 0, so each GPU gets a full 2D matrix per batch.

- GPU 0: src[0:64, :, :] → 64 full images

- GPU 1: src[64:128, :, :] → remaining 64 full images

Complete module
---------------

Putting the pieces above together, here is a complete module that
can be run with the ``legate`` command line launcher:

.. literalinclude:: examples/fft.py
   :language: python

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

    Time elapsed for fft: 0.573000 milliseconds

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

    Time elapsed for fft: 0.613000 milliseconds
