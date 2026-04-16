.. _doctor:

cuPyNumeric Doctor
==================

cuPyNumeric Doctor is a built-in diagnostic tool that monitors your code at
runtime and warns you about usage patterns that are known to be inefficient
with cuPyNumeric that are mentioned in :ref:`practices`.
It is designed to help you identify anti-patterns in NumPy
code after porting to cuPyNumeric, and to guide you toward
array-based implementations that scale well across CPUs and GPUs.
It is sufficient to run the program on single processor (CPU or GPU)
using cuPyNumeric Doctor since anti-patterns can be
detected regardless of the scale of execution.

Enabling cuPyNumeric Doctor
---------------------------

Set the ``CUPYNUMERIC_DOCTOR=1`` environment variable when running your script:

.. code-block:: bash

    CUPYNUMERIC_DOCTOR=1 python main.py

Doctor output is printed to stdout at the end of execution. You can also write
it to a file:

.. code-block:: bash

    CUPYNUMERIC_DOCTOR=1 CUPYNUMERIC_DOCTOR_FILENAME=report.txt python main.py

For additional configuration options, see :ref:`settings`.

Output Formats
--------------

The default output format is ``plain`` text. Two additional formats are
available: ``json`` and ``csv``. These are useful when you want to process
Doctor output programmatically.

.. code-block:: bash

    # plain text (default)
    CUPYNUMERIC_DOCTOR=1 python main.py

    # JSON
    CUPYNUMERIC_DOCTOR=1 CUPYNUMERIC_DOCTOR_FORMAT=json python main.py

    # CSV
    CUPYNUMERIC_DOCTOR=1 CUPYNUMERIC_DOCTOR_FORMAT=csv python main.py

To include full Python tracebacks in the output (useful for locating the
issue in deeply nested call stacks):

.. code-block:: bash

    CUPYNUMERIC_DOCTOR=1 CUPYNUMERIC_DOCTOR_TRACEBACK=1 python main.py

End-to-end Workflow
-------------------

This section walks through a complete workflow: starting with a NumPy script,
switching to cuPyNumeric, running Doctor, interpreting the output, and
improving the code.

Step 1: Start with a NumPy Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you have the following script, ``main.py``:

.. code-block:: python

    import numpy as np

    N = 20

    # Initialize arrays
    x = np.zeros((N, N))
    y = np.random.rand(N, N)

    # Anti-pattern 1: element-wise loop with scalar indexing
    for i in range(N):
        for j in range(N):
            x[i, j] = y[i, j] * 2.0

    # Anti-pattern 2: use of Python built-in sum() on an array
    total = sum(sum(x))

    # Anti-pattern 3: use of nonzero() to find and index elements
    indices = np.nonzero(y > 0.5)
    x[indices] = 0.0

    print("Done. total =", total)

This runs correctly with NumPy. However, it contains several patterns that
will not scale well with cuPyNumeric.

Step 2: Switch to cuPyNumeric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace the NumPy import with cuPyNumeric — no other changes needed:

.. code-block:: python

    import cupynumeric as np   # <-- only change

    N = 20

    # Initialize arrays
    x = np.zeros((N, N))
    y = np.random.rand(N, N)

    # Anti-pattern 1: element-wise loop with scalar indexing
    for i in range(N):
        for j in range(N):
            x[i, j] = y[i, j] * 2.0

    # Anti-pattern 2: use of Python built-in sum() on an array
    total = sum(sum(x))

    # Anti-pattern 3: use of nonzero() to find and index elements
    indices = np.nonzero(y > 0.5)
    x[indices] = 0.0

    print("Done. total =", total)

Step 3: Run with cuPyNumeric Doctor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    CUPYNUMERIC_DOCTOR=1 python main.py

Step 4: Interpret the Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Doctor produces a report at the end of the run, grouping all detected issues:

.. code-block:: text

    !!! cuPyNumeric Doctor reported issues !!!

    - issue: multiple scalar item accesses repeated on the same line
      detected on: line 12 of file 'main.py':

        x[i, j] = y[i, j] * 2.0

      refer to: https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-array-based-operations-avoid-loops-with-indexing

    - issue: iterating over a cuPyNumeric array with __iter__ is slow; use vectorized operations instead
      detected on: line 15 of file 'main.py':

        total = sum(sum(x))

      refer to: https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-array-based-operations-avoid-loops-with-indexing

    - issue: multiple scalar item accesses repeated on the same line
      detected on: line 15 of file 'main.py':

        total = sum(sum(x))

      refer to: https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-array-based-operations-avoid-loops-with-indexing

    - issue: use of nonzero can be slow in cuPyNumeric
      detected on: line 18 of file 'main.py':

        indices = np.nonzero(y > 0.5)

      refer to: https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-boolean-masks-avoid-advanced-indexing

    - issue: use of advanced indexing can be slow in cuPyNumeric
      detected on: line 19 of file 'main.py':

        x[indices] = 0.0

      refer to: https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-boolean-masks-avoid-advanced-indexing

Each diagnostic entry contains:

* **issue** — a brief description of the detected anti-pattern
* **detected on** — the exact file and line number, along with the offending
  source line
* **refer to** — a link to the relevant best-practices guidance (when
  available)

Step 5: Fix the Issues
~~~~~~~~~~~~~~~~~~~~~~~

Using the Doctor output and the best-practices documentation from
ref:`practices` as a guide, rewrite the script with array-based operations:

.. code-block:: python

    import cupynumeric as np

    N = 20

    # Initialize arrays
    x = np.zeros((N, N))
    y = np.random.rand(N, N)

    # Fixed: replace element-wise loop with a vectorized operation
    x = y * 2.0

    # Fixed: use np.sum() instead of Python built-in sum()
    total = np.sum(x)

    # Fixed: use a boolean mask instead of nonzero()
    cond = y > 0.5
    x[cond] = 0.0

    print("Done. total =", total)

Step 6: Verify with Doctor
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the updated script with Doctor again to confirm all issues are resolved:

.. code-block:: bash

    CUPYNUMERIC_DOCTOR=1 python main.py

If no anti-patterns remain, Doctor produces no output and the program exits
cleanly. You can then scale the fixed script to multiple GPUs or nodes without
further modification.

Detected Anti-Patterns
----------------------

Doctor checks for compliance with recommendations and most anti-patterns
mentioned in  :ref:`practices`. Note that it is possible that the same source
could trigger multiple warnings, as seen in the example above:

.. code-block:: python

   total = sum(sum(x))

creates multiple warnings due to how the python built-in function ``sum`` is implemented.

.. code-block:: text

    - issue: iterating over a cuPyNumeric array with __iter__ is slow; use vectorized operations instead
      detected on: line 15 of file 'main.py':

        total = sum(sum(x))

      refer to: https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-array-based-operations-avoid-loops-with-indexing

    - issue: multiple scalar item accesses repeated on the same line
      detected on: line 15 of file 'main.py':

        total = sum(sum(x))

      refer to: https://docs.nvidia.com/cupynumeric/latest/user/practices.html#use-array-based-operations-avoid-loops-with-indexing

Following the best practices will typically resolve all the warnings.
