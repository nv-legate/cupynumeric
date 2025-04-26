Installation
============

Installing Conda Packages
-------------------------

cuPyNumeric supports the
`same platforms as Legate <https://docs.nvidia.com/legate/latest/installation.html#support-matrix>`_.

cuPyNumeric is available from
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_
on the `legate channel <https://anaconda.org/legate/cupynumeric>`_.

.. note::
   conda version >= 24.1 required

.. code-block:: bash

   # with a new environment
   $ conda create -n myenv -c conda-forge -c legate cupynumeric

   # =========== OR =========== #

   # into an existing environment
   $ conda install -c conda-forge -c legate cupynumeric

.. important::

   Packages are only offered for Linux (x86_64 and aarch64) supporting Python
   versions 3.11 to 3.13.

Installing PyPI Packages
------------------------

cuPyNumeric is also available from `PyPI
<https://pypi.org/project/nvidia-cupynumeric>`_.  To install, run the following
command:

.. code-block:: bash

   # into existing environment
   $ pip install nvidia-cupynumeric

   # =========== OR =========== #

   # into new environment
   $ python -m venv myenv
   $ source myenv/bin/activate
   $ pip install nvidia-cupynumeric

This will install the latest version of cuPyNumeric and the corresponding
version of `Legate <https://github.com/nv-legate/legate>`_.

The cuPyNumeric package on PyPI is multi-node and multi-rank capable.  Please
check `Legate <https://docs.nvidia.com/legate/latest/networking-wheels.html>`_
documentation to find more details about running on multiple nodes.

.. important::

   Packages are only offered for Linux (x86_64 and aarch64) supporting Python
   versions 3.11 to 3.13.

Verify your Installation
------------------------

You can verify the installation by running one of the
`examples <https://github.com/nv-legate/cunumeric/tree/HEAD/examples>`_.

For instance:

.. code-block:: sh

   $ legate examples/black_scholes.py
   Running black scholes on 10K options...
   Elapsed Time: 129.017 ms

Conda and GPU / CPU Variants
----------------------------

``conda`` automatically installs the right variant for the system:
* CPU variant if no NVIDIA GPU is detected
* GPU variant if an NVIDIA GPU is detected

To override this behavior and force install a version with GPU support, use the
following (with the desired CUDA version):

.. code-block:: sh

   $ CONDA_OVERRIDE_CUDA="12.2" conda install -c conda-forge -c legate cupynumeric


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