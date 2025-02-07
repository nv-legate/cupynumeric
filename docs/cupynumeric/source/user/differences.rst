Differences with Numpy
======================
Supported shapes and datatypes
------------------------------

cuPyNumeric natively supports arrays of dimensionality only up to the maximum
number of dimensions supported by the linked build of Legate.

cuPyNumeric natively supports only numerical datatypes, and doesn't support
extended-precision floats (e.g. `np.float128`).

Trying to use an unsupported number of dimensions or datatype will trigger a
fallback to base NumPy.

Returning a copy instead of a view
----------------------------------

Some functions that return a view in Numpy return a copy in cuPyNumeric. These
include:

* ``np.diag``
* ``ndarray.flat``
* ``np.flip``

Also, reshaping will trigger a copy in more cases than NumPy (operations that
add/remove unitary dimensions, e.g. ``np.ones(3, 4).reshape(3, 1, 4)``, return
a view, but those that do not, e.g. ``np.ones(3,4).reshape(12)``, return a
copy).

Order argument
--------------

The ``order`` argument is generally not implemented, because it doesn't make
sense in a distributed setting (the argument is accepted, but currently
ignored in most methods).

Reductions
----------

Reductions such as ``cumprod``, etc. may return different values due to
floating-point error accumulations happening in different orders when results
are evaluated in a distributed manner. Similar statements apply to the
propagation of ``nan`` and ``inf`` values in a distributed setting.

Scalar return values
--------------------

NumPy will occasionally convert a 0d array to a python-level scalar, but
cuPyNumeric avoids doing that, because in our system an array value can
potentially represent an asynchronous computation. As a result, sometimes
cuPyNumeric will return 0d arrays (possibly deferred), in cases where NumPy
returns a scalar.

Indexing behavior
-----------------

``x[:,True]`` works differently from NumPy. cuPyNumeric broadcasts it up to the
corresponding dimension, whereas NumPy adds a dimension.

Additionally ``[]`` does not work for advanced indexing since ``[]`` is
``float64`` by default.

cuPyNumeric doesn't support non-unit steps on index expressions, e.g. `arr[::2]`.

Duplicate indices on advanced indexing expressions produce undefined behavior.
This is also the case in NumPy but the current NumPy implementation happens
to produce a deterministic result. (See https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments)
