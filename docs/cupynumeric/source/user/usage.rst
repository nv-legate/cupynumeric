.. _usage:

Usage
=====

Using cuPyNumeric as a replacement for NumPy is simple. Replace your NumPy import
statement with cuPyNumeric:

.. code-block:: python

  import numpy as np

becomes

.. code-block:: python

  import cupynumeric as np

Then, run the application like you usually do. For example, if you had a script
``main.py`` written in NumPy that adds two vectors,

.. code-block:: python

    import numpy as np
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    z = x + y
    print(z)

change the import statement to use cuPyNumeric like below,

.. code-block:: python

    import cupynumeric as np
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    z = x + y
    print(z)

And run the program, like this

.. code-block:: sh

    python main.py

By default this invocation will use all the hardware resources (e.g. CPU cores,
RAM, GPUs) available on the current machine.

For more information on controlling the resource allocation, running on multiple
nodes etc. see https://docs.nvidia.com/legate/latest/usage.html.

Jupyter
-------

It is possible to run cuPyNumeric code in Jupyter notebooks. You can configure a
Jupyter kernel with desired runtime configuration using the ``legate-jupyter`` command:

.. code-block:: sh

    legate-jupyter --name legate_cpus_2 --cpus 2

Then select this kernel to use in the running notebook.

For more information see
`Running Legate programs with Jupyter Notebook <https://docs.nvidia.com/legate/latest/manual/usage/jupyter.html>`_.
