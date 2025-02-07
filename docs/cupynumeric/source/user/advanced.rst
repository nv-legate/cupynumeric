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

Passing Legion and Realm arguments
----------------------------------

It is also possible to pass options to the Legion and Realm runtime directly,
by way of the ``LEGION_DEFAULT_ARGS`` and ``REALM_DEFAULT_ARGS`` environment
variables, for example:

.. code-block:: sh

    LEGION_DEFAULT_ARGS="-ll:cputsc" legate main.py