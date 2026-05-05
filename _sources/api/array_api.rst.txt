Array API support
=================

.. currentmodule:: cupynumeric

cuPyNumeric exposes the Array API namespace dispatch and inspection hooks used
by Array API consumers.

cuPyNumeric currently exposes ``None`` as its only Array API device token.
``None`` here means Legate-managed placement, not CPU placement. Concrete
device strings such as ``"cpu"`` or ``"cuda:0"`` are rejected by the Array API
hooks.

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
