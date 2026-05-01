Array API support
=================

.. currentmodule:: cupynumeric

cuPyNumeric exposes the Array API namespace dispatch and inspection hooks used
by Array API consumers.

.. data:: __array_api_version__

   The Array API standard version implemented by cuPyNumeric's Array API
   namespace.

.. autosummary::
   :toctree: generated/

   __array_namespace_info__

.. autoclass:: ArrayNamespaceInfo

   .. rubric:: Methods

   .. autosummary::

      ~ArrayNamespaceInfo.capabilities
      ~ArrayNamespaceInfo.default_device
      ~ArrayNamespaceInfo.default_dtypes
      ~ArrayNamespaceInfo.devices
      ~ArrayNamespaceInfo.dtypes
