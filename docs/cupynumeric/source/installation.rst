Installation
============

Default conda install
---------------------

cuPyNumeric is available from
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_
on the `legate channel <https://anaconda.org/legate/cupynumeric>`_.
Please make sure you have at least conda version 24.1 installed, then create
a new environment containing cuPyNumeric:

.. code-block:: sh

    conda create -n myenv -c conda-forge -c legate cupynumeric

or install it into an existing environment:

.. code-block:: sh

    conda install -c conda-forge -c legate cupynumeric

Packages with GPU support are available, and will be chosen automatically by
``conda install`` on systems with GPUs.

In an environment without GPUs available, ``conda install`` will by default
choose a CPU-only package. To install a version with GPU support in such an
environment, use environment variable ``CONDA_OVERRIDE_CUDA``:

.. code-block:: sh

    CONDA_OVERRIDE_CUDA="12.2" \
      conda install -c conda-forge -c legate cupynumeric

Once installed, you can verify the installation by running one of the examples
from the cuPyNumeric repository, for instance:

.. code-block:: sh

    $ legate examples/black_scholes.py
    Running black scholes on 10K options...
    Elapsed Time: 129.017 ms

Building from source
---------------------

See :ref:`building cupynumeric from source` for instructions on building
cuPyNumeric manually.

Licenses
--------

This project will download and install additional third-party open source
software projects at install time. Review the license terms of these open
source projects before use.

For license information regarding projects bundled directly, see
:ref:`thirdparty`.