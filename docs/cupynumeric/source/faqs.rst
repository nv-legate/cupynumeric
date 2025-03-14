.. _faqs:

Frequently Asked Questions
==========================


What are the different task variants available in Legate?
---------------------------------------------------------

Legate offers three different task variants: CPU, OMP, and GPU. A task variant
determines the type of processor Legate chooses to perform the computations.

What is the difference between Legate and cuPyNumeric?
------------------------------------------------------

Legate is a task-based runtime software stack that enables development of
scalable and composable libraries for distributed and accelerated computing.

cuPyNumeric is one of the foundational libraries built using Legate and aspires
to be a distributed and accelerated drop-in replacement library for NumPy, an
array programming library widely used in scientific computing. cuPyNumeric scales
idiomatic NumPy programs to multiple GPUs and CPUs and seamlessly interoperates
with other Legate libraries.

Check out this `blog post <https://developer.nvidia.com/blog/effortlessly-scale-numpy-from-laptops-to-supercomputers-with-nvidia-cupynumeric/>`_
to learn more about cuPyNumeric.

When to use python vs legate?
-----------------------------

The ``legate`` launcher affords comman line options for configurtion, while
using ``python`` requires configuring via ``LEGATE_CONFIG``. When running
local applications, it is mostly a matter of preference. When running in
multi-node situations, ``legate`` has some additional command line options
that may make usage simpler.

What if I don’t have a GPU?
---------------------------

If you don’t have a GPU, you can either use the CPU or the OMP variant. See
`Resource allocation` for informations on how to use the respective variants.

What does this warning mean?
----------------------------

.. code-block:: text

    RuntimeWarning: cuPyNumeric has not implemented <API> and is falling back to canonical NumPy. You may notice significantly decreased performance for this function call.

This means that the NumPy <API> has not been implemented in cuPyNumeric and that
the Legate runtime is falling back to using NumPy’s implementation which will
be single-threaded execution and can lead to decreased performance for that
function call.

.. code-block:: text

    [0 - 7f0524da9740]    0.000028 {4}{threads}: reservation ('dedicated worker (generic) #1') cannot be satisfied

or

.. code-block:: text

    [0 - 7fe90fa7d740]    0.000029 {4}{threads}: reservation ('utility proc 1d00000000000001') cannot be satisfied

This indicates that the runtime was unable to pin threads onto available cores,
which usually means that the available CPU cores were oversubscribed because
the user has requested more cores than is available.

If the user does not specify which type of processor to run on, legate will use
4 CPUs to execute the program. Legate will also need one core to perform the
dependency analysis and schedule the tasks. If there are fewer than five cores
on the machine, try reducing the number of cores (``--cpus``) passed to legate.

This warning is currently expected on MacOS.

How to determine available memory?
----------------------------------

On Linux, running the following command will display the amount of
available system memory:

.. code-block:: sh

    cat /proc/meminfo | grep MemAvailable

Available GPU memory (for each GPU) can be displayed by running:

.. code-block:: sh

    nvidia-smi --query-gpu memory.free --format=csv

Both of these represent the available amount of memory, which may be shared
with other processes or libraries. You may need to reduce these amounts to
account for these, or to reflect the actual size of your problem more closely.

If you do not have access to run the commands above, then refer to published
machine specs or cluster documentation.

How to handle Out-Of-Memory errors?
-----------------------------------

.. code-block:: text

    [0 - 7fda18f26000]    0.805182 {5}{cunumeric.mapper}: Failed to allocate 8388608 bytes on memory 1e00000000000000 (of kind SYSTEM_MEM) for region requirement(s) 1 of Task cupynumeric::BinaryOpTask[oom.py:24] (UID 18)

The above error indicates that the application ran out of memory during
execution. More granular details on the type of memory, the task that triggered
the error, and what was using up the available memory are provided in the error
message. If possible, try increasing the amount of system memory or framebuffer
memory allocated to the program, or decrease the problem size.

Reducing the ``--eager-alloc-percentage`` to, say, 10 or less can also help
since this reduces the amount of available memory available to the eager memory
pool and will consequently increase the memory reserved for the deferred memory
pool.

Why are the results different from NumPy?
-----------------------------------------

While a majority of the APIs will give the same result as NumPy, some APIs
might be implemented differently from that of NumPy which might lead to
differences in results. One such example is, :ref:`reshape`, which returns a
copy of the array in cuPyNumeric but returns a view in NumPy. Another example
is :ref:`astype` which does *not* return a copy by default, where NumPy does.

Such differences in implementation are noted in the documentation of the
cuPyNumeric APIs, please review them before opening an issue on the
`cuPyNumeric issue tracker <https://github.com/nv-legate/cupynumeric/issues>`_.

Why doesn’t Legate use my GPU?
------------------------------

If you explicitly asked legate to use the GPU but find that the GPU is not
being used, it is possible that your problem size is too small to be run on
GPU and be performant. Either increase your problem size significantly or set
the environment variable ``LEGATE_TEST`` to 1 and run. Setting this environment
variable tells Legate to always use the prescribed resources regardless of the
problem size.

What are the anti-patterns in a NumPy code?
-------------------------------------------

Check out our :ref:`practices` to avoid some of the anti-patterns commonly
encountered in applications.

How do I time the execution of my application?
----------------------------------------------

Check out the :ref:`benchmarking` section for information on how to accurately
measure cuPyNumeric execution.

Why is cuPyNumeric slower than NumPy on my laptop?
--------------------------------------------------

For small problem sizes, cuPyNumeric might be slower than NumPy. We suggest you
increase the problem size and correspondingly increase the resources needed
for the problem size as described in the Usage section. Take a look at our
:ref:`practices` on how to do that.

Why is cuPyNumeric slower than CuPy on my laptop?
-------------------------------------------------

For small problem sizes, cuPyNumeric might be slower than CuPy. We suggest you
increase the problem size and correspondingly increase the resources needed for
the problem size as described in the :ref:`Usage` section. Take a look at
performance :ref:`practices`.

How do I use Jupyter Notebooks?
-------------------------------

See https://docs.nvidia.com/legate/latest/jupyter.html.

How to pass Legion and Realm arguments?
---------------------------------------

See :ref:`advanced`.

What is the version of legate?
------------------------------

Use ``legate-issue`` to know more about the version of Legate, Legion and
several other key packages.

You can also run ``legate –verbose ./script.py <script-options>`` to get
verbose output.

What are the defaults?
----------------------

The default values for several input arguments to Legate are mentioned in
Legate's documentation.

Where I can read more about cuPyNumeric?
----------------------------------------

Check out this `blog post <https://developer.nvidia.com/blog/accelerating-python-applications-with-cupynumeric-and-legate/>`_
or this `tutorial <https://github.com/NVIDIA/accelerated-computing-hub/blob/main/Accelerated_Python_User_Guide/notebooks/Chapter_11_Distributed_Computing_cuPyNumeric.ipynb>`_
to learn more about cuPyNumeric.

Questions?
----------

For technical questions about cuPyNumeric and Legate-based tools, please visit
the `community discussion forum <https://github.com/nv-legate/discussion>`_.

If you have other questions, please contact us at *legate@nvidia.com*.
