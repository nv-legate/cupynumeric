:html_theme.sidebar_secondary.remove:

NVIDIA cuPyNumeric
==================

cuPyNumeric implements the NumPy API on top of the Legate framework,
providing transparent accelerated computing that scales from a single CPU
to a single GPU, and up to multi-node, multi-GPU systems.

For example, you can run `the final example of the Python CFD course`_
completely unmodified on 2048 A100 GPUs in a `DGX SuperPOD`_ and achieve
good weak scaling.

.. toctree::
  :maxdepth: 1
  :caption: Contents:

  installation
  user/index
  examples/index
  api/index
  faqs
  developer/index
  changes/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

.. _NumPy: https://numpy.org/
.. _Legate: https://github.com/nv-legate/legate
.. _DGX SuperPOD: https://www.nvidia.com/en-us/data-center/dgx-superpod/
.. _the final example of the Python CFD course: https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb
