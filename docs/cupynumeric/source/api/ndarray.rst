.. currentmodule:: cupynumeric

The N-Dimensional array (:class:`cupynumeric.ndarray`)
======================================================

Constructing arrays
-------------------

New arrays can be constructed using the routines detailed in Array creation
routines, and also by using the low-level ndarray constructor:

.. toctree::
   :maxdepth: 2
   :hidden:

   _ndarray


:meth:`ndarray`

Calculation
-----------

.. autosummary::
   :toctree: generated/

   ndarray.all
   ndarray.any


.. Indexing arrays

.. Internal memory layout of an ndarray

Array Attributes
----------------

Memory Layout
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ndarray.flags
   ndarray.shape
   ndarray.strides
   ndarray.ndim
   ndarray.data
   ndarray.size
   ndarray.itemsize
   ndarray.nbytes
   ndarray.base
   ndarray.ctypes

Data Type
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ndarray.dtype

Other Attributes
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ndarray.T
   ndarray.real
   ndarray.imag
   ndarray.flat

.. Array interface

.. cypes foreign function interface

Array Methods
-------------

Array conversion
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ndarray.item
   ndarray.tolist
   ndarray.itemset
   ndarray.tostring
   ndarray.tobytes
   ndarray.tofile
   ndarray.dump
   ndarray.dumps
   ndarray.astype
   .. ndarray.byteswap
   ndarray.copy
   ndarray.view
   ndarray.getfield
   .. TODO: this is not in the numpy page
   ndarray.setfield
   ndarray.setflags
   ndarray.fill

Shape manipulation
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ndarray.reshape
   .. ndarray.resize
   ndarray.transpose
   ndarray.swapaxes
   ndarray.flatten
   ndarray.ravel
   ndarray.squeeze

Item selection and manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ndarray.take
   ndarray.put
   .. ndarray.repeat
   ndarray.choose
   ndarray.sort
   ndarray.argsort
   ndarray.partition
   ndarray.argpartition
   ndarray.searchsorted
   ndarray.nonzero
   ndarray.compress
   ndarray.diagonal
   ndarray.trace

Calculation
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ndarray.max
   ndarray.argmax
   ndarray.min
   ndarray.argmin
   .. ndarray.ptp
   ndarray.clip
   ndarray.conj
   ndarray.conjugate
   ndarray.dot
   ndarray.flip
   ndarray.round
   .. ndarray.trace
   ndarray.sum
   ndarray.cumsum
   ndarray.mean
   ndarray.var
   .. ndarray.std
   ndarray.prod
   ndarray.cumprod
   ndarray.all
   ndarray.any
   ndarray.unique

Arithmetic, matrix multiplication, and comparison operations
------------------------------------------------------------

.. autosummary::
   :toctree: generated/

   ndarray.__lt__
   ndarray.__le__
   ndarray.__gt__
   ndarray.__ge__
   ndarray.__eq__
   ndarray.__ne__

Truth value of an array(bool()):

.. autosummary::
   :toctree: generated/

   ndarray.__bool__

Unary operations:

.. autosummary::
   :toctree: generated/

   ndarray.__neg__
   ndarray.__pos__
   ndarray.__abs__
   ndarray.__invert__

Arithmetic:

.. autosummary::
   :toctree: generated/

   ndarray.__add__
   ndarray.__sub__
   ndarray.__mul__
   ndarray.__truediv__
   ndarray.__floordiv__
   ndarray.__mod__
   ndarray.__divmod__
   ndarray.__pow__
   ndarray.__lshift__
   ndarray.__rshift__
   ndarray.__and__
   ndarray.__or__
   ndarray.__xor__

Arithmetic, in-place:

.. autosummary::
   :toctree: generated/

   ndarray.__iadd__
   ndarray.__isub__
   ndarray.__imul__
   ndarray.__itruediv__
   ndarray.__ifloordiv__
   ndarray.__imod__
   ndarray.__ipow__
   ndarray.__ilshift__
   ndarray.__irshift__
   ndarray.__iand__
   ndarray.__ior__
   ndarray.__ixor__

Matrix Multiplication:

.. autosummary::
   :toctree: generated/

   ndarray.__matmul__
   ndarray.__imatmul__
   ndarray.__rmatmul__

Special methods
---------------

For standard library functions:

.. autosummary::
   :toctree: generated/

   ndarray.__copy__
   ndarray.__deepcopy__
   ndarray.__index__
   ndarray.__reduce__
   ndarray.__setstate__

Basic customization:

.. autosummary::
   :toctree: generated/

   ndarray.__new__
   ndarray.__array__
   .. ndarray.__array_wrap__

Container customization: (see Indexing)

.. autosummary::
   :toctree: generated/

   ndarray.__len__
   ndarray.__getitem__
   ndarray.__setitem__
   ndarray.__contains__

Conversion;
.. the operations int(), float() and complex(). They work only on arrays that have one element in them and return the appropriate scalar.

.. autosummary::
   :toctree: generated/

   ndarray.__int__
   ndarray.__float__
   ndarray.__complex__

String representations:

.. autosummary::
   :toctree: generated/

   ndarray.__str__
   ndarray.__repr__

.. Utility method for typing:

.. .. autosummary::
..    :toctree: generated/

..    ndarray.__class_getitem__

cuPyNumeric-specific methods
----------------------------

.. autosummary::
   :toctree: generated/

   ndarray.stencil_hint
