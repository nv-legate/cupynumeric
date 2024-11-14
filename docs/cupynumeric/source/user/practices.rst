.. _practices:

Best practices
==============

General Recommendations
-----------------------

Following the basics of numpy as documented
`here <https://numpy.org/doc/stable/user/basics.html>`_ is highly recommended.
Here we highlight some of the anti-patterns and best practices for cuPyNumeric
to avoid commonly encountered problems related to performance. In general,
array-based computations are recommended.

Availability of each API (e.g., single CPU or Multiple GPUs/Multiple CPUs,
etc.) is noted in the docstring of the API. This would be useful to know while
designing the application since it can impact the scalability.

Guidelines on using cuPyNumeric APIs
----------------------------------

Use cuPyNumeric or NumPy arrays, AVOID native lists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a cuPyNumeric array from data structures native to Python like lists,
tuples, etc., and operate on the cuPyNumeric array, as shown in the example
below. Find more details on this here:

.. https://numpy.org/doc/stable/user/basics.creation.html

.. code-block:: python

    # Not recommended: Performing large-scale computation using lists
    # and other native Python data structures
    x = [1, 2, 3]
    y = []
    for val in x:
        y.append(val + 2)

    # Recommended: Create a cuPyNumeric array and use array-based operations
    y = np.array(x)
    y = x + 2


In the example below, the function ``transform`` is defined to operate on
scalars. But it can also be used on an array to linearly transform its elements,
thus performing an array-based operation.

.. code-block:: python

    import cupynumeric as np

    def transform(input):
        return (input + 3) * 4

    x = np.linspace(start=0, stop=10, num=11)

    # Acceptable options
    y = transform(x)
    # or
    y = (x + 3) * 4

Use array-based operations, AVOID loops with indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use array-based implementations as much as possible, and while doing so, ensure
that some of the best practices given below are followed.

If a component of the array needs to be set/updated, use an array-based
implementation instead of an explicit loop-based implementation.

.. code-block:: python

    # x and y are three-dimensional arrays

    # Not recommended: Naive element-wise implementation
    for i in range(ny):
        for j in range(nx):
            x[0, j, i] = y[3, j, i]

    # Recommended: Array-based implementation
    x[0] = y[3]

The same recommendation applies when the value we are setting to is a scalar
or when values are set conditionally. We first form the condition array
corresponding to the conditional and then use that to update the array,
essentially breaking it down to three steps:

* create the condition array
* update the array corresponding to the `if` statement
* update the array corresponding to the `else` statement while noting that
  the condition is flipped for the `else` statement.

.. code-block:: python

    # x and y are two-dimensional arrays, and we need to update x
    # depending on whether y meets a condition or not.

    # Not recommended: Naive element-wise implementation
    for i in range(ny):
        for j in range(nx):
            if (y[j, i] < tol):
                x[j, i] = const
            else
                x[j, i] = 1.0 - const

    # Recommended: Array-based implementation
    cond = y < tol
    x[cond] = const
    x[~cond] = 1.0 - const


Use boolean masks, AVOID advanced indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Indexing using boolean masks instead of indices is recommended for better
performance. In the example below, indexing the array using a boolean mask
will be faster than using a array with indices derived from ``nonzero`` since
the latter could incur additional communication that might be undesirable for
performance.

.. code-block:: python

    import cupynumeric as np

    # Not recommended: don't use nonzero to get indices
    indices = np.nonzero(h < 0)
    x[indices] = y[indices]

    # Recommended: Use boolean mask to update the array
    cond = h < 0
    x[cond] = y[cond]

Use putmask to update an array based on another array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When an array needs to be updated from another array based on a condition
that they both satisfy, use ``putmask`` for better performance. In this
example, the values of ``x`` are updated to twice the value of ``y`` only when the
condition is met, which can be described using the ``putmask`` API.

.. code-block:: python

    import cupynumeric as np

    # We need to update elements of x from y based on a condition
    cond = y < tol

    # Acceptable
    x[cond] = y[cond] * 2.0

    # Recommended: use putmask to update elements based on a condition
    np.putmask(x, cond, y * 2.0)

Use logic functions, AVOID iterating through a loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting elements of an array that satisfy multiple conditions to a scalar
should be done using logic functions instead of iterating through a loop.
Here is an example:

.. code-block:: python

    # Not recommended: naive element-wise update to update x
    for i in range(ny):
        for j in range(nx):
            if (first_cond and second_cond):
                x[j, i] = const

    # Recommended: Use logical operations.
    x[np.logical_and(first_cond, second_cond)] = const


Refer to the `documentation for other logical operations <https://numpy.org/doc/stable/reference/routines.logic.html#logical-operations>`_.

Use mathematical functions, AVOID element-wise loops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When there are nested element-wise operations, it is recommended that they
are translated to array-based operations using equivalent cuPyNumeric APIs, if
possible. Here is an example:

.. code-block:: python

    import cupynumeric as np

    # Not recommended: Naive element-wise implementation
    for i in range(ny):
        for j in range(nx):
            x[j, i] = max(max(y[j, i], z[j, i]), const)

    # Recommended: Use array-based implementation
    x = np.maximum(np.maximum(y, z), const)


Array Manipulation Routine Pitfalls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _reshape:

Reshape returns a copy instead of view
......................................

It's important to note that in our implementation, ``reshape`` returns a copy
of the array rather than a view like numpy, so this deviation can cause
differences in results, as shown in the example below. This additional copy
can also make it run slower, so we recommend using it as sparingly as possible.

.. code-block:: python

    import cupynumeric as np

    x = np.ones((3,4))
    y = x.reshape((12,))

    y[0] = 42

    assert x[0,0] == 42 # succeeds in NumPy, fails in cuPyNumeric

Stack results in a performance penalty
......................................

There is a performance penalty to stacking arrays using
`hstack <https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy-hstack>`_
or
`vstack <https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy-vstack>`_
because they incur additional copies of data in our implementation.

Faster I/O Routines
~~~~~~~~~~~~~~~~~~~

As of 23.07, we recommend using `h5py <https://github.com/h5py/h5py>`_ to perform I/O.

Guidelines on designing cuPyNumeric applications
----------------------------------------------

Use output arguments to reduce memory allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever possible, use the ``out`` parameter in the APIs, to avoid allocating an
intermediate array in our implementation.

.. code-block:: python

    import cupynumeric as np

    # Acceptable
    x = x + y
    y = x - y
    x = x * y

    # Recommended for better performance
    np.add(x, y, out=x)
    np.subtract(x, y, out=y)
    np.multiply(x, y, out=x)


Vectorize for better performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions with conditionals that operate on scalars might make array-based
operations less straightforward. The general recommendation in such cases is to
apply the three step process mentioned here where we evaluate the conditional
and then apply it for both the ``if`` and ``else`` statements. Here is an
example of what approaches might or might not work. The first and second
options have ``if`` and ``else`` clauses written out as separate array-based
operations while the third option (using the API ``where``) includes them both
in one API.

.. code-block:: python

    # Works with scalars but not NumPy arrays
    def bar(x):
        if x < 0:
            return x + 1
        else:
            return x + 2

    # Not Recommended for arrays
    x = np.array(...)
    y = bar(x) # doesn't work

    # Recommended (1): Use array-based operations
    cond = x < 0
    x[cond] += 1
    x[~cond] += 2

    # Recommended (2): Use array-based operations
    cond = x < 0
    np.add(x, 1, where=cond, out=x)
    np.add(x, 2, where=~cond, out=x)

    # Recommended (3): Use array-based operations
    cond = x < 0
    x = np.where(cond, x + 1, x + 2)


Merge tasks to reduce overhead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended that tasks (e.g., a Python operation like ``z = x + y``,
will be a task) be large enough to execute for at least a millisecond to
mitigate the runtime overheads associated with launching a task. One way to
make the tasks execute for longer is to merge them when possible. This is
especially useful for tasks that are really small, in the order of a few
hundred microseconds or less. Here is an example:

.. code-block:: python

    # x is a 3D array of shape (4, _, _) where only the first three
    # components need to be updated. cond is a 2D bool mask derived from h
    cond = h < 0.0 # h is a two-dimensional array

    # Updating arrays like this is acceptable
    x[0, cond] = const
    x[1, cond] = const
    x[2, cond] = const

    # Making them into one is recommended
    x[0:3, cond] = const


Avoid blocking operations
~~~~~~~~~~~~~~~~~~~~~~~~~

While this might require more invasive application-level changes, it is often
recommended that any blocking operation in an iterative loop is delayed as much
as possible. Blocking can occur when there is data-dependency between execution
of tasks. In the example below, the runtime will be blocked until the result
from ``norm < tolerance`` is available since ``norm`` needs to be fetched from
the processor it is running on to evaluate the conditional.

The current recommended best practice is to design applications such that these
blocking operations are done as sparingly as possible, as permitted by the
computations performed inside the iterative loop. This might manifest in
different ways in applications, so only one illustrative example is provided
here.

.. code-block:: python

    import cupynumeric as np

    # compute() does some computations and returns a multi-dimensional
    # cuPyNumeric array. The application stops after the iterative computation
    # is converged

    # Acceptable: Performing convergence checks every iteration
    for i in range(niterations):
        x_current = compute()
        if i > 0:
            norm = np.linalg.norm(x_current - x_prev)
            if norm < tolerance:
                break
        x_prev = x_current.copy()

    # Recommended: Reduce the frequency of convergence checks
    every_niter = 5
    for i in range(niterations):
        x_current = compute()
        if i > 0 and i%every_niter == 0:
            norm = np.linalg.norm(x_current - x_prev)
            if norm < tolerance:
                break

    # This could potentially be updated one iteration before the
    # convergence check, but that's not done here
    x_prev = x_current.copy()
