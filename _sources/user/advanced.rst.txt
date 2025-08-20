.. _advanced:

Advanced topics
===============

Multi-node execution
--------------------

Using ``legate``
~~~~~~~~~~~~~~~~

cuPyNumeric programs can be run in parallel by using the ``--nodes`` option to
the ``legate`` driver, followed by the number of nodes to be used.
When running on 2+ nodes, a task launcher must be specified.

Legate currently supports using ``mpirun``, ``srun``, and ``jsrun`` as task
launchers for multi-node execution via the ``--launcher`` command like
arguments:

.. code-block:: sh

  legate --launcher srun --nodes 2 script.py <script options>

Using a manual task manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

  mpirun -np N legate script.py <script options>

It is also possible to use "standard python" in place of the ``legate`` driver.

For more details about running multi-node configurations, please see the
`Distributed Launch <https://docs.nvidia.com/legate/latest/manual/usage/running.html#distributed-launch>`_
chapter of the the `Legate documentation <https://docs.nvidia.com/legate/latest/index.html>`_.

Passing Legion and Realm arguments
----------------------------------

It is also possible to pass options to the Legion and Realm runtime directly,
by way of the ``LEGION_DEFAULT_ARGS`` and ``REALM_DEFAULT_ARGS`` environment
variables, for example:

.. code-block:: sh

    LEGION_DEFAULT_ARGS="-ll:cputsc" legate main.py

Using the GASNet networking backend
-----------------------------------

Standard Legate packges come with UCX networking support.
To run cuPyNumeric programs with Legate using GASNet requires installing
additional separate packages.
Please see `How Do I Install Legate with the MPI and GASNet wrappers <https://docs.nvidia.com/legate/latest/gasnet.html#how-do-i-install-legate-with-the-mpi-and-gasnet-wrappers>`_
for full details.

Resource Scoping
----------------

Legate provides APIs for resource scoping that can be used in cuPyNumeric
programs. For example, to restrict a block of code to only run on GPUs, you
can use the following:

.. code-block:: python

    from legate.core import TaskTarget, get_legate_runtime

    machine = get_legate_runtime().get_machine()
    with machine.only(TaskTarget.GPU):
        # code to run only on GPUs

Please see `Machine and Resource Scoping <https://docs.nvidia.com/legate/latest/api/python/machine.html>`_
for full information.
