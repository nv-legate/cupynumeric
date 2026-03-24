Trying Numpy code without changes
=================================

The ``lgpatch`` script (in the same location as the ``legate`` executable) can
help facilitate quick demonstrations of ``cupynumeric`` on existing codebases
that make use of ``numpy``.

To use this tool, invoke it as shown below, with the name of the program to
patch:

.. code-block:: sh

    lgpatch <program> -patch numpy

For example, here is a small ``test.py`` program that imports and uses various
``numpy`` funtions:

.. code-block:: python

    # test.py

    import numpy as np
    input = np.eye(10, dtype=np.float32)
    np.linalg.cholesky(input)

You can invoke ``lgpatch`` to run ``test.py`` using ``cupynumeric`` functions
instead, without any changes to the original source code. Any standard
``cupynumeric`` runtime options may also be used:

.. code-block:: sh

    $ LEGATE_CONFIG="--cpus 4"  lgpatch test.py -patch numpy
