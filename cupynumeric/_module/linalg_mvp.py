# Copyright 2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import re
from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import opt_einsum as oe  # type: ignore [import-untyped]

from .._array.array import ndarray
from .._array.util import (
    add_boilerplate,
    convert_to_cupynumeric_ndarray,
    find_common_type,
)
from .._ufunc.math import multiply
from .._utils.linalg import (
    AxesPairLike,
    inner_modes,
    matmul_modes,
    tensordot_modes,
)
from ..types import NdShape
from .creation_data import copy

if TYPE_CHECKING:
    from .._ufunc.ufunc import CastingKind

_builtin_all = all
_builtin_max = max


@add_boilerplate("a", "b")
def inner(a: ndarray, b: ndarray, out: ndarray | None = None) -> ndarray:
    """
    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : array_like
    out : ndarray, optional
        Output argument. This must have the exact shape that would be returned
        if it was not present. If its dtype is not what would be expected from
        this operation, then the result will be (unsafely) cast to `out`.

    Returns
    -------
    output : ndarray
        If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        ``output.shape = (*a.shape[:-1], *b.shape[:-1])``
        If `out` is given, then it is returned.

    Notes
    -----
    The cuPyNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``inner(ones((1,)), ones((4,)))`` is allowed.

    See Also
    --------
    numpy.inner

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if a.ndim == 0 or b.ndim == 0:
        return multiply(a, b, out=out)
    (a_modes, b_modes, out_modes) = inner_modes(a.ndim, b.ndim)
    return _contract(
        a_modes, b_modes, out_modes, a, b, out=out, casting="unsafe"
    )


@add_boilerplate("a", "b")
def dot(a: ndarray, b: ndarray, out: ndarray | None = None) -> ndarray:
    """
    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using ``a @ b`` is preferred.

    - If either `a` or `b` is 0-D (scalar), it is equivalent to
      :func:`multiply` and using ``cupynumeric.multiply(a, b)`` or ``a * b`` is
      preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of `a` and the second-to-last axis of
      `b`::

        dot(a: ndarray, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : ndarray, optional
        Output argument. This must have the exact shape and dtype that would be
        returned if it was not present.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`. If `out` is given, then it is
        returned.

    Notes
    -----
    The cuPyNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``dot(ones((3,1)), ones((4,5)))`` is allowed.

    Except for the inner-product case, only floating-point types are supported.

    See Also
    --------
    numpy.dot

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.dot(b, out=out)


@add_boilerplate("a", "b")
def matmul(
    a: ndarray,
    b: ndarray,
    /,
    out: ndarray | None = None,
    *,
    casting: CastingKind = "same_kind",
    dtype: np.dtype[Any] | None = None,
) -> ndarray:
    """
    Matrix product of two arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays, scalars not allowed.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that matches the signature `(n,k),(k,m)->(n,m)`.
    casting : ``{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}``, optional
        Controls what kind of data casting may occur.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

        Default is 'same_kind'.
    dtype : data-type, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.

    Returns
    -------
    output : ndarray
        The matrix product of the inputs.
        This is a scalar only when both a, b are 1-d vectors.
        If `out` is given, then it is returned.

    Notes
    -----
    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional
      matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of
      matrices residing in the last two indexes and broadcast accordingly.
    - If the first argument is 1-D, it is promoted to a matrix by
      prepending a 1 to its dimensions. After matrix multiplication
      the prepended 1 is removed.
    - If the second argument is 1-D, it is promoted to a matrix by
      appending a 1 to its dimensions. After matrix multiplication
      the appended 1 is removed.

    ``matmul`` differs from ``dot`` in two important ways:

    - Multiplication by scalars is not allowed, use ``*`` instead.
    - Stacks of matrices are broadcast together as if the matrices
      were elements, respecting the signature ``(n,k),(k,m)->(n,m)``:

      >>> a = ones([9, 5, 7, 4])
      >>> c = ones([9, 5, 4, 3])
      >>> dot(a: ndarray, c).shape
      (9, 5, 7, 9, 5, 3)
      >>> matmul(a: ndarray, c).shape
      (9, 5, 7, 3)
      >>> # n is 7, k is 4, m is 3

    The cuPyNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``matmul(ones((3,1)), ones((4,5)))`` is
    allowed.

    Only floating-point types are supported.

    See Also
    --------
    numpy.matmul

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("Scalars not allowed in matmul")

    (a_modes, b_modes, out_modes) = matmul_modes(a.ndim, b.ndim)

    return _contract(
        a_modes,
        b_modes,
        out_modes,
        a,
        b,
        out=out,
        casting=casting,
        dtype=dtype,
    )


@add_boilerplate("a", "b")
def vdot(a: ndarray, b: ndarray, out: ndarray | None = None) -> ndarray:
    """
    Return the dot product of two vectors.

    The vdot(`a`, `b`) function handles complex numbers differently than
    dot(`a`, `b`).  If the first argument is complex the complex conjugate
    of the first argument is used for the calculation of the dot product.

    Note that `vdot` handles multidimensional arrays differently than `dot`:
    it does *not* perform a matrix product, but flattens input arguments
    to 1-D vectors first. Consequently, it should only be used for vectors.

    Parameters
    ----------
    a : array_like
        If `a` is complex the complex conjugate is taken before calculation
        of the dot product.
    b : array_like
        Second argument to the dot product.
    out : ndarray, optional
        Output argument. This must have the exact shape that would be returned
        if it was not present. If its dtype is not what would be expected from
        this operation, then the result will be (unsafely) cast to `out`.

    Returns
    -------
    output : ndarray
        Dot product of `a` and `b`. If `out` is given, then it is returned.

    Notes
    -----
    The cuPyNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``vdot(ones((1,)), ones((4,)))`` is allowed.

    See Also
    --------
    numpy.vdot

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return inner(a.ravel().conj(), b.ravel(), out=out)


@add_boilerplate("a", "b")
def outer(a: ndarray, b: ndarray, out: ndarray | None = None) -> ndarray:
    """
    Compute the outer product of two vectors.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and ``b = [b0, b1, ..., bN]``,
    the outer product is::

      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) array_like
        First input vector. Input is flattened if not already 1-dimensional.
    b : (N,) array_like
        Second input vector. Input is flattened if not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored. If its dtype is not what would
        be expected from this operation, then the result will be (unsafely)
        cast to `out`.

    Returns
    -------
    output : (M, N) ndarray
        ``output[i, j] = a[i] * b[j]``
        If `out` is given, then it is returned.

    See Also
    --------
    numpy.outer

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return multiply(
        a.ravel()[:, np.newaxis], b.ravel()[np.newaxis, :], out=out
    )


@add_boilerplate("a", "b")
def tensordot(
    a: ndarray, b: ndarray, axes: AxesPairLike = 2, out: ndarray | None = None
) -> ndarray:
    """
    Compute tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : array_like
        Tensors to "dot".

    axes : int or array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.
    out : ndarray, optional
        Output argument. This must have the exact shape that would be returned
        if it was not present. If its dtype is not what would be expected from
        this operation, then the result will be (unsafely) cast to `out`.

    Returns
    -------
    output : ndarray
        The tensor dot product of the inputs. If `out` is given, then it is
        returned.

    Notes
    -----
    The cuPyNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``tensordot(ones((3,1)), ones((1,4)))`` is
    allowed.

    Except for the inner-product case, only floating-point types are supported.

    See Also
    --------
    numpy.tensordot

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    (a_modes, b_modes, out_modes) = tensordot_modes(a.ndim, b.ndim, axes)

    return _contract(
        a_modes, b_modes, out_modes, a, b, out=out, casting="unsafe"
    )


# Trivial multi-tensor contraction strategy: contract in input order
class NullOptimizer(oe.paths.PathOptimizer):  # type: ignore [misc,no-any-unimported] # noqa
    def __call__(
        self,
        inputs: list[set[str]],
        outputs: set[str],
        size_dict: dict[str, int],
        memory_limit: int | None = None,
    ) -> list[tuple[int, int]]:
        return [(0, 1)] + [(0, -1)] * (len(inputs) - 2)


def _maybe_cast_input(
    arr: ndarray, to_dtype: np.dtype[Any], casting: CastingKind
) -> ndarray:
    if arr.dtype == to_dtype:
        return arr
    if not np.can_cast(arr.dtype, to_dtype, casting=casting):
        raise TypeError(
            f"Cannot cast input array of type {arr.dtype} to {to_dtype} with "
            f"casting rule '{casting}'"
        )
    return arr.astype(to_dtype)


# Generalized tensor contraction
def _contract(
    a_modes: list[str],
    b_modes: list[str],
    out_modes: list[str],
    a: ndarray,
    b: ndarray | None = None,
    out: ndarray | None = None,
    casting: CastingKind = "same_kind",
    dtype: np.dtype[Any] | None = None,
) -> ndarray:
    # Sanity checks
    if len(a_modes) != a.ndim:
        raise ValueError(
            f"Expected {len(a_modes)}-d input array but got {a.ndim}-d"
        )

    if b is None:
        if len(b_modes) != 0:
            raise ValueError("Missing input array")
    elif len(b_modes) != b.ndim:
        raise ValueError(
            f"Expected {len(b_modes)}-d input array but got {b.ndim}-d"
        )

    if out is not None and len(out_modes) != out.ndim:
        raise ValueError(
            f"Expected {len(out_modes)}-d output array but got {out.ndim}-d"
        )

    if len(set(out_modes)) != len(out_modes):
        raise ValueError("Duplicate mode labels on output")

    if len(set(out_modes) - set(a_modes) - set(b_modes)) > 0:
        raise ValueError("Unknown mode labels on output")

    makes_view = b is None and len(a_modes) == len(out_modes)
    if dtype is not None and not makes_view:
        c_dtype = dtype
    elif out is not None:
        c_dtype = out.dtype
    elif b is None:
        c_dtype = a.dtype
    else:
        c_dtype = find_common_type(a, b)

    a = _maybe_cast_input(a, c_dtype, casting)

    if b is not None:
        b = _maybe_cast_input(b, c_dtype, casting)

    out_dtype = out.dtype if out is not None else c_dtype

    # Handle duplicate modes on inputs
    c_a_modes = Counter(a_modes)
    for mode, count in c_a_modes.items():
        if count > 1:
            axes = [i for (i, m) in enumerate(a_modes) if m == mode]
            a = a._diag_helper(axes=axes)
            # diagonal is stored on last axis
            a_modes = [m for m in a_modes if m != mode] + [mode]
    c_b_modes = Counter(b_modes)
    for mode, count in c_b_modes.items():
        if count > 1:
            axes = [i for (i, m) in enumerate(b_modes) if m == mode]
            b = b._diag_helper(axes=axes)  # type: ignore [union-attr]
            # diagonal is stored on last axis
            b_modes = [m for m in b_modes if m != mode] + [mode]

    # Drop modes corresponding to singleton dimensions. This handles cases of
    # broadcasting.
    for dim in reversed(range(a.ndim)):
        if a.shape[dim] == 1:
            a = a.squeeze(dim)
            a_modes.pop(dim)
    if b is not None:
        for dim in reversed(range(b.ndim)):
            if b.shape[dim] == 1:
                b = b.squeeze(dim)
                b_modes.pop(dim)

    # Sum-out modes appearing on one argument, and missing from the result
    # TODO: If we supported sum on multiple axes we could do the full sum in a
    # single operation, and avoid intermediates.
    for dim, mode in reversed(list(enumerate(a_modes))):
        if mode not in b_modes and mode not in out_modes:
            a_modes.pop(dim)
            a = a.sum(axis=dim)

    for dim, mode in reversed(list(enumerate(b_modes))):
        if mode not in a_modes and mode not in out_modes:
            b_modes.pop(dim)
            b = b.sum(axis=dim)  # type: ignore [union-attr]

    # Compute extent per mode. No need to consider broadcasting at this stage,
    # since it has been handled above.
    mode2extent: dict[str, int] = {}
    for mode, extent in chain(
        zip(a_modes, a.shape), zip(b_modes, b.shape) if b is not None else []
    ):
        prev_extent = mode2extent.get(mode)
        if prev_extent is not None and extent != prev_extent:
            raise ValueError(
                f"Incompatible sizes between matched dimensions: {extent} vs "
                f"{prev_extent}"
            )
        mode2extent[mode] = extent

    # Any modes appearing only on the result must have originally been present
    # on one of the operands, but got dropped by the broadcast-handling code.
    out_shape = (
        out.shape
        if out is not None
        else tuple(mode2extent.get(mode, 1) for mode in out_modes)
    )
    c_modes = []
    c_shape: NdShape = ()
    c_bloated_shape: NdShape = ()
    for mode, extent in zip(out_modes, out_shape):
        if mode not in a_modes and mode not in b_modes:
            c_bloated_shape += (1,)
        else:
            assert extent > 1
            c_modes.append(mode)
            c_shape += (extent,)
            c_bloated_shape += (extent,)

    # Verify output array has the right shape (input arrays can be broadcasted
    # up to match the output, but not the other way around). There should be no
    # unknown or singleton modes on the result at this point.
    for mode, extent in zip(c_modes, c_shape):
        prev_extent = mode2extent[mode]
        assert prev_extent != 1
        if extent != prev_extent:
            raise ValueError("Wrong shape on output array")

    # Test for fallback to unary case
    if b is not None:
        if len(a_modes) == 0:
            a = a * b
            a_modes = b_modes
            b = None
            b_modes = []
        elif len(b_modes) == 0:
            a = a * b
            b = None

    if b is None:
        # Unary contraction case
        assert len(a_modes) == len(c_modes) and set(a_modes) == set(c_modes)
        if len(a_modes) == 0:
            # NumPy doesn't return a view in this case
            c = copy(a)
        elif a_modes == c_modes:
            c = a
        else:
            # Shuffle input array according to mode labels
            axes = [a_modes.index(mode) for mode in c_modes]
            assert _builtin_all(ax >= 0 for ax in axes)
            c = a.transpose(axes)

    else:
        # Binary contraction case
        # Create result array, if output array can't be directly targeted
        if out is not None and out_dtype == c_dtype and out_shape == c_shape:
            c = out
        else:
            c = ndarray._from_inputs(
                shape=c_shape, dtype=c_dtype, inputs=(a, b)
            )
        # Perform operation
        c._thunk.contract(
            c_modes, a._thunk, a_modes, b._thunk, b_modes, mode2extent
        )

    # Postprocess result before returning
    if out is c:
        # We already decided above to use the output array directly
        return out
    if out_dtype != c_dtype or out_shape != c_bloated_shape:
        # We need to broadcast the result of the contraction or switch types
        # before returning
        if not np.can_cast(c_dtype, out_dtype, casting=casting):
            raise TypeError(
                f"Cannot cast intermediate result array of type {c_dtype} "
                f"into output array of type {out_dtype} with casting rule "
                f"'{casting}'"
            )
        if out is None:
            out = ndarray._from_inputs(
                shape=out_shape, dtype=out_dtype, inputs=(c,)
            )
        out[...] = c.reshape(c_bloated_shape)
        return out
    if out_shape != c_shape:
        # We need to add missing dimensions, but they are all of size 1, so
        # we don't need to broadcast
        assert c_bloated_shape == out_shape
        if out is None:
            return c.reshape(out_shape)
        else:
            out[...] = c.reshape(out_shape)
            return out
    if out is not None:
        # The output and result arrays are fully compatible, but we still
        # need to copy
        out[...] = c
        return out
    return c


def einsum(
    expr: str,
    *operands: ndarray,
    out: ndarray | None = None,
    dtype: np.dtype[Any] | None = None,
    casting: CastingKind = "safe",
    optimize: bool | Literal["greedy", "optimal"] = True,
) -> ndarray:
    """
    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list[array_like]
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    dtype : data-type, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.
    casting : ``{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}``, optional
        Controls what kind of data casting may occur.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : ``{False, True, 'greedy', 'optimal'}``, optional
        Controls if intermediate optimization should occur. If False then
        arrays will be contracted in input order, one at a time. True (the
        default) will use the 'greedy' algorithm. See
        ``cupynumeric.einsum_path`` for more information on the available
        optimization algorithms.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    Notes
    -----
    For most expressions, only floating-point types are supported.

    See Also
    --------
    numpy.einsum

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    operands_list = [convert_to_cupynumeric_ndarray(op) for op in operands]

    if out is not None:
        out = convert_to_cupynumeric_ndarray(out, share=True)

    if optimize is True:
        optimize = "greedy"
    elif optimize is False:
        optimize = NullOptimizer()

    # This call normalizes the expression (adds the output part if it's
    # missing, expands '...') and checks for some errors (mismatch on number
    # of dimensions between operand and expression, wrong number of operands,
    # unknown modes on output, a mode appearing under two different
    # non-singleton extents).
    computed_operands, contractions = oe.contract_path(
        expr, *operands_list, einsum_call=True, optimize=optimize
    )
    for indices, _, sub_expr, _, _ in contractions:
        assert len(indices) == 1 or len(indices) == 2
        a = computed_operands.pop(indices[0])
        b = computed_operands.pop(indices[1]) if len(indices) == 2 else None
        if b is None:
            m = re.match(r"([a-zA-Z]*)->([a-zA-Z]*)", sub_expr)
            if m is None:
                raise NotImplementedError("Non-alphabetic mode labels")
            a_modes = list(m.group(1))
            b_modes = []
            out_modes = list(m.group(2))
        else:
            m = re.match(r"([a-zA-Z]*),([a-zA-Z]*)->([a-zA-Z]*)", sub_expr)
            if m is None:
                raise NotImplementedError("Non-alphabetic mode labels")
            a_modes = list(m.group(1))
            b_modes = list(m.group(2))
            out_modes = list(m.group(3))
        sub_result = _contract(
            a_modes,
            b_modes,
            out_modes,
            a,
            b,
            out=(out if len(computed_operands) == 0 else None),
            casting=casting,
            dtype=dtype,
        )
        computed_operands.append(sub_result)

    assert len(computed_operands) == 1
    return convert_to_cupynumeric_ndarray(computed_operands[0])


def einsum_path(
    expr: str,
    *operands: ndarray,
    optimize: bool | list[Any] | tuple[Any, ...] | str = "greedy",
) -> tuple[list[str | int], str]:
    """
    Evaluates the lowest cost contraction order for an einsum expression by
    considering the creation of intermediate arrays.

    Parameters
    ----------
    expr : str
        Specifies the subscripts for summation.
    *operands : Sequence[array_like]
        These are the arrays for the operation.
    optimize : ``{bool, list, tuple, 'greedy', 'optimal'}``
        Choose the type of path. If a tuple is provided, the second argument is
        assumed to be the maximum intermediate size created. If only a single
        argument is provided the largest input or output array size is used
        as a maximum intermediate size.

        * if a list is given that starts with ``einsum_path``, uses this as the
          contraction path
        * if False no optimization is taken
        * if True defaults to the 'greedy' algorithm
        * 'optimal' An algorithm that combinatorially explores all possible
          ways of contracting the listed tensors and chooses the least costly
          path. Scales exponentially with the number of terms in the
          contraction.
        * 'greedy' An algorithm that chooses the best pair contraction
          at each step. Effectively, this algorithm searches the largest inner,
          Hadamard, and then outer products at each step. Scales cubically with
          the number of terms in the contraction. Equivalent to the 'optimal'
          path for most contractions.

        Default is 'greedy'.

    Returns
    -------
    path : list[tuple[int,...]]
        A list representation of the einsum path.
    string_repr : str
        A printable representation of the einsum path.

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the
    end of the contraction list. This list can then be iterated over until all
    intermediate contractions are complete.

    See Also
    --------
    numpy.einsum_path

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    computed_operands = [convert_to_cupynumeric_ndarray(op) for op in operands]
    memory_limit = _builtin_max(op.size for op in computed_operands)
    if isinstance(optimize, tuple):
        if len(optimize) != 2:
            raise ValueError("einsum_path expects optimize tuples of size 2")
        optimize, memory_limit = optimize
    if optimize is True:
        optimize = "greedy"
    elif optimize is False:
        optimize = [tuple(range(len(computed_operands)))]
    elif optimize in ["greedy", "optimal"]:
        pass
    elif (
        isinstance(optimize, list)
        and len(optimize) > 1
        and optimize[0] == "einsum_path"
    ):
        optimize = optimize[1:]
    else:
        raise ValueError(
            f"einsum_path: unexpected value for optimize: {optimize}"
        )
    path, info = oe.contract_path(
        expr, *computed_operands, optimize=optimize, memory_limit=memory_limit
    )
    return ["einsum_path"] + path, info


@add_boilerplate("a")
def trace(
    a: ndarray,
    offset: int = 0,
    axis1: int | None = None,
    axis2: int | None = None,
    dtype: np.dtype[Any] | None = None,
    out: ndarray | None = None,
) -> ndarray:
    """
    Return the sum along diagonals of the array.

    If a is 2-D, the sum along its diagonal with the given offset is
    returned, i.e., the sum of elements a[i,i+offset] for all i.
    If a has more than two dimensions, then the axes specified by axis1
    and axis2 are used to determine the 2-D sub-arrays whose traces
    are returned. The shape of the resulting array is the same as that
    of a with axis1 and axis2 removed.

    Parameters
    ----------
    a : array_like
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both
        positive and negative. Defaults to 0.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays
        from which the diagonals should be taken. Defaults are the
        first two axes of a.
    dtype : data-type, optional
        Determines the data-type of the returned array and of the
        accumulator where the elements are summed. If dtype has the value
        None and a is of integer type of precision less than the default
        integer precision, then the default integer precision is used.
        Otherwise, the precision is the same as that of a.

    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and
        it must be of the right shape to hold the output.

    Returns
    -------
    sum_along_diagonals : ndarray
        If a is 2-D, the sum along the diagonal is returned. If a has
        larger dimensions, then an array of sums along diagonals is returned.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    See Also
    --------
    numpy.diagonal

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.trace(
        offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
    )
