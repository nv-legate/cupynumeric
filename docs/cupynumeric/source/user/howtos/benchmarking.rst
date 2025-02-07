.. _benchmarking:

Performance Benchmarking
========================

Using Legate timing tools
-------------------------

Use legate's timing API to measure elapsed time, rather than standard Python
timers. cuPyNumeric executes work asynchronously when possible, and a standard
Python timer will only measure the time taken to launch the work, not the time
spent in actual computation.

Make sure warm-up iterations, initialization, I/O, and other one-time
computations are excluded while timing iterative computations.

Here is an example of how to measure elapsed time in milliseconds:

.. code-block:: python

    import cupynumeric as np
    from legate.timing import time

    init() # Initialization step

    # Do few warm-up iterations
    for i in range(n_warmup_iters):
        compute()

    start = time()
    for i in range(niters):
        compute()
    end = time()

    elapsed_millisecs = (end - start)/1000.0

    dump_data() # I/O


Guidelines for performance benchmarks
-------------------------------------

Manual partitioning of data for use with message-passing from Python (say,
using mpi4py package) is discouraged.

Ensure that the problem size is large enough to offset runtime overheads
associated with tasks. A rule of thumb is that the problem size is large
enough for a task granularity of about 1 millisecond (as of release 23.07).

For arrays that are small, or for arrays that operate on a subset of a larger
array, it is recommended that they be merged with similar operations when
possible. For example, in some applications using structured meshes, boundary
conditions are set on a subset of data (at the boundaries only) which typically
tends to be a sequence of very small operations. When possible, boundary
conditions for different variables and different boundaries should be combined.
In general, merging small operations might yield better results.
