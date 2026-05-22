Array API support
=================

.. currentmodule:: cupynumeric

cuPyNumeric's Array API support is incremental and is not a conformance claim.
It exposes the namespace dispatch and inspection hooks used by Array API
consumers, along with selected standard function spellings backed by native
cuPyNumeric implementations.

cuPyNumeric currently exposes ``None`` as its only Array API device token.
``None`` here means Legate-managed placement, not CPU placement. Concrete
device strings such as ``"cpu"`` or ``"cuda:0"`` are rejected by the Array API
hooks.

Array API functions with wrapper behavior are listed here. Direct Array API
aliases are listed in the matching functional API sections.

.. data:: __array_api_version__

   The Array API standard version implemented by cuPyNumeric's Array API
   namespace.

.. autosummary::
   :toctree: generated/

   __array_namespace_info__

.. autosummary::
   :toctree: generated/

   astype
   isdtype
   matrix_transpose

.. autoclass:: ArrayNamespaceInfo

   .. rubric:: Methods

   .. autosummary::

      ~ArrayNamespaceInfo.capabilities
      ~ArrayNamespaceInfo.default_device
      ~ArrayNamespaceInfo.default_dtypes
      ~ArrayNamespaceInfo.devices
      ~ArrayNamespaceInfo.dtypes
