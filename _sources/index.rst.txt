:html_theme.sidebar_secondary.remove:

NVIDIA cuPyNumeric
==================

With cuPyNumeric you can write code productively in Python, using the familiar
`NumPy API`_, and have your program scale with no code changes from single-CPU
computers to multi-node-multi-GPU clusters.

For example, you can run the final example of the `Python CFD course`_
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


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

.. _DGX SuperPOD: https://www.nvidia.com/en-us/data-center/dgx-superpod/
.. _Numpy API: https://numpy.org/doc/stable/reference/
.. _Python CFD course: https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb