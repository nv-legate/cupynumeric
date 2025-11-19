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

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..lib.array_utils import normalize_axis_index, normalize_axis_tuple
from ..runtime import runtime
from cupynumeric._utils import is_np2

if is_np2:
    from numpy.lib.array_utils import normalize_axis_index
    from numpy.lib.array_utils import normalize_axis_tuple
else:
    from numpy.core.multiarray import (  # type: ignore
        normalize_axis_index,
    )
    from numpy.core.numeric import (  # type: ignore
        normalize_axis_tuple,
    )

from legate.core import get_machine

from .._array.util import add_boilerplate, convert_to_cupynumeric_ndarray
from .._module import dot, empty_like, eye, matmul, ndarray
from .._module.array_rearrange import flip
from .._module.creation_matrices import diag
from .._module.creation_shape import zeros, zeros_like
from .._module.ssc_searching import where
from .._module.ssc_sorting import argsort
from .._ufunc.math import add, sqrt as _sqrt
from ._exception import LinAlgError

if TYPE_CHECKING:
    import numpy.typing as npt


@add_boilerplate("a")
def cholesky(a: ndarray) -> ndarray:
    """
    Cholesky decomposition.

    Return the Cholesky decomposition, `L * L.H`, of the square matrix `a`,
    where `L` is lower-triangular and .H is the conjugate transpose operator
    (which is the ordinary transpose if `a` is real-valued).  `a` must be
    Hermitian (symmetric if real-valued) and positive-definite. No
    checking is performed to verify whether `a` is Hermitian or not.
    In addition, only the lower-triangular and diagonal elements of `a`
    are used. Only `L` is actually returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.

    Returns
    -------
    L : (..., M, M) array_like
        Upper or lower-triangular Cholesky factor of `a`.  Returns a
        matrix object if `a` is a matrix object.

    Notes
    -----
    The current implementation kills the process when the decomposition fails.

    See Also
    --------
    numpy.linalg.cholesky

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    shape = a.shape
    if len(shape) < 2:
        raise ValueError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )
    elif shape[-1] != shape[-2]:
        raise ValueError("Last 2 dimensions of the array must be square")

    return _thunk_cholesky(a)


@add_boilerplate("a")
def eig(a: ndarray) -> tuple[ndarray, ...]:
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrices for which the eigenvalues and right eigenvectors will be
        computed, at least dimension 2.

    Returns
    -------
    eigenvalues : (…, M) array_like
        The eigenvalues, each repeated according to its multiplicity.
    eigenvectors : (…, M, M) array
        The normalized (unit “length”) eigenvectors, such that the column
        eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue
        eigenvalues[i].

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    Notes
    -----
    Unlike NumPy, cuPyNumeric always returns complex-dtype results, even if the
    imaginary part is zero.

    Multi-GPU/CPU usage is limited to data parallel matrix-wise batching.

    See Also
    --------
    numpy.linalg.eig

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    shape = a.shape
    if len(shape) < 2:
        raise LinAlgError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )
    if shape[-2] != shape[-1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if np.dtype("e") == a.dtype:
        raise TypeError("array type float16 is unsupported in linalg")
    return _thunk_eig(a)


@add_boilerplate("a")
def eigvals(a: ndarray) -> ndarray:
    """
    Compute the eigenvalues of a square array.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrices for which the eigenvalues will be computed, at least
        dimension 2.

    Returns
    -------
    w : (…, M) array_like
        The eigenvalues, each repeated according to its multiplicity.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    Notes
    -----
    Unlike NumPy, cuPyNumeric always returns complex-dtype results, even if the
    imaginary part is zero.

    Multi-GPU/CPU usage is limited to data parallel matrix-wise batching.

    See Also
    --------
    numpy.linalg.eigvals

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    shape = a.shape
    if len(shape) < 2:
        raise LinAlgError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )

    if shape[-2] != shape[-1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if np.dtype("e") == a.dtype:
        raise TypeError("array type float16 is unsupported in linalg")
    return _thunk_eigvals(a)


@add_boilerplate("a")
def eigh(a: ndarray, UPLO: str = "L") -> tuple[ndarray, ...]:
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrices for which the eigenvalues and right eigenvectors will be
        computed, at least dimension 2.
    UPLO {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of a ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    eigenvalues : (…, M) array_like
        The eigenvalues in ascending order, each repeated according to its
        multiplicity.
    eigenvectors : (…, M, M) array
        The normalized (unit “length”) eigenvectors, such that the column
        eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue
        eigenvalues[i].

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    Notes
    -----
    Multi-GPU/CPU usage is limited to data parallel matrix-wise batching.

    See Also
    --------
    numpy.linalg.eigh

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    shape = a.shape
    if len(shape) < 2:
        raise LinAlgError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )
    if shape[-2] != shape[-1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if np.dtype("e") == a.dtype:
        raise TypeError("array type float16 is unsupported in linalg")

    if UPLO == "L":
        uplo_l = True
    elif UPLO == "U":
        uplo_l = False
    else:
        raise ValueError(f"UPLO {UPLO} not supported.")

    return _thunk_eigh(a, uplo_l)


@add_boilerplate("a")
def eigvalsh(a: ndarray, UPLO: str = "L") -> ndarray:
    """
    Compute the eigenvalues of a square array.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrices for which the eigenvalues will be computed, at least
        dimension 2.
    UPLO {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of a ('L', default) or the upper triangular part ('U').
        Irrespective of this value only the real parts of the diagonal will
        be considered in the computation to preserve the notion of a Hermitian
        matrix. It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

    Returns
    -------
    w : (…, M) array_like
        The eigenvalues in ascending order, each repeated according to its
        multiplicity.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    Notes
    -----
    Multi-GPU/CPU usage is limited to data parallel matrix-wise batching.

    See Also
    --------
    numpy.linalg.eigvalsh

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    shape = a.shape
    if len(shape) < 2:
        raise LinAlgError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )
    if shape[-2] != shape[-1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if np.dtype("e") == a.dtype:
        raise TypeError("array type float16 is unsupported in linalg")

    if UPLO == "L":
        uplo_l = True
    elif UPLO == "U":
        uplo_l = False
    else:
        raise ValueError(f"UPLO {UPLO} not supported.")

    return _thunk_eigvalsh(a, uplo_l)


@add_boilerplate("a")
def qr(a: ndarray) -> tuple[ndarray, ...]:
    """
    Compute the qr factorization of a matrix.

    Factor the matrix a as qr, where q is orthonormal
    and r is upper-triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Array like, at least dimension 2.

    Returns
    -------
    q : (M, K) array_like
        A matrix with orthonormal columns. K = min(M, N).
    r : (K, N) array_like
        The uppoer triangular matrix.

    Raises
    ------
    LinAlgError
        If factoring fails.

    Notes
    -----
    Currently does not support the parameter 'mode' from numpy 1.8.

    See Also
    --------
    numpy.linalg.qr

    Availability
    --------
    Single GPU, Single CPU
    """
    shape = a.shape
    if len(shape) < 2:
        raise LinAlgError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )
    if len(shape) > 2:
        raise NotImplementedError(
            "cuPyNumeric does not yet support stacked 2d arrays"
        )
    if np.dtype("e") == a.dtype:
        raise TypeError("array type float16 is unsupported in linalg")
    return _thunk_qr(a)


@add_boilerplate("a", "b")
def solve(a: ndarray, b: ndarray, out: ndarray | None = None) -> ndarray:
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    Computes the "exact" solution, `x`, of the well-determined, i.e., full
    rank, linear matrix equation `ax = b`.

    Parameters
    ----------
    a : (..., M, M) array_like
        Coefficient matrix.
    b : {(M,), (..., M, K)}, array_like
        Ordinate or "dependent variable" values.
    out : {(..., M,), (..., M, K)}, array_like, optional
        An optional output array for the solution

    Returns
    -------
    x : {(..., M,), (..., M, K)} ndarray
        Solution to the system a x = b.  Returned shape is identical to `b`.

    Raises
    ------
    LinAlgError
        If `a` is singular or not square.

    Notes
    ------
    Single matrix multi-GPU usage is limited to cusolverMP. Additional
    multi-GPU/CPU usage is limited to data parallel matrix-wise batching.

    See Also
    --------
    numpy.linalg.solve

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if a.ndim < 2:
        raise LinAlgError(
            f"{a.ndim}-dimensional array given. Array must be at least two-dimensional"
        )
    if b.ndim < 1:
        raise LinAlgError(
            f"{b.ndim}-dimensional array given. Array must be at least one-dimensional"
        )
    if np.dtype("e") in (a.dtype, b.dtype):
        raise TypeError("array type float16 is unsupported in linalg")
    if a.shape[-2] != a.shape[-1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if a.ndim == 2 and a.shape[1] != b.shape[0]:
        if b.ndim == 1:
            raise ValueError(
                "Input operand 1 has a mismatch in its dimension 0, "
                f"with signature (m,m),(m)->(m) (size {b.shape[0]} "
                f"is different from {a.shape[1]})"
            )
        else:
            raise ValueError(
                "Input operand 1 has a mismatch in its dimension 0, "
                f"with signature (m,m),(m,n)->(m,n) (size {b.shape[0]} "
                f"is different from {a.shape[1]})"
            )
    if a.ndim > 2:
        if a.ndim != b.ndim:
            raise ValueError(
                "Batched matrices require signature (...,m,m),(...,m,n)->(...,m,n)"
            )
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(
                "Input operand 1 has a mismatch in its dimension "
                f"{b.ndim - 2}, with signature (...,m,m),(...,m,n)->(...,m,n)"
                f" (size {b.shape[-2]} is different from {a.shape[-1]})"
            )

    if a.size == 0 or b.size == 0:
        return empty_like(b)

    return _thunk_solve(a, b, out)


@add_boilerplate("a")
def svd(a: ndarray, full_matrices: bool = True) -> tuple[ndarray, ...]:
    """
    Singular Value Decomposition.

    Parameters
    ----------
    a : (M, N) array_like
        Array like, at least dimension 2.
    full_matrices : bool, optional
        If True (default), u and vh are of shape (M, M), (N, N).
        If False, the shapes are (M, K) and (K, N), where K = min(M, N).

    Returns
    -------
    u : (M, M) array_like
        Unitary array(s).
    s : (K) array_like
        The singular values, sorted in descending order
    vh : (N, N) array_like
        Unitary array(s).

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    Notes
    -----
    Currently does not support the parameters 'compute_uv' and 'hermitian'.

    See Also
    --------
    numpy.linalg.svd

    Availability
    --------
    Single GPU, Single CPU
    """
    shape = a.shape
    if len(shape) < 2:
        raise LinAlgError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )
    if len(shape) > 2:
        raise NotImplementedError(
            "cuPyNumeric does not yet support stacked 2d arrays"
        )
    if shape[0] < shape[1]:
        raise NotImplementedError("cuPyNumeric only supports M >= N")
    if np.dtype("e") == a.dtype:
        raise TypeError("array type float16 is unsupported in linalg")
    return _thunk_svd(a, full_matrices)


# This implementation is adapted closely from NumPy
@add_boilerplate("a")
def matrix_power(a: ndarray, n: int) -> ndarray:
    """
    Raise a square matrix to the (integer) power `n`.
    For positive integers `n`, the power is computed by repeated matrix
    squarings and matrix multiplications. If ``n == 0``, the identity matrix
    of the same shape as M is returned. If ``n < 0``, the inverse
    is computed and then raised to the ``abs(n)``.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be "powered".
    n : int
        The exponent can be any integer, positive, negative, or zero.

    Returns
    -------
    a**n : (..., M, M) ndarray
        The return value is the same shape and type as `M`;
        if the exponent is positive or zero then the type of the
        elements is the same as those of `M`. If the exponent is
        negative the elements are floating-point.

    See Also
    --------
    numpy.linalg.matrix_power

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Process inputs
    if a.ndim < 2:
        raise LinAlgError(f"Expected at least 2d array, but got {a.ndim}d")
    if a.shape[-2] != a.shape[-1]:
        raise LinAlgError("Last 2 dimensions of the array must be square")
    if not isinstance(n, int):
        raise TypeError("exponent must be an integer")

    # Special cases
    if n == 0:
        a = empty_like(a)
        a[...] = eye(a.shape[-2], dtype=a.dtype)
        return a

    # Invert if necessary
    if n < 0:
        # TODO: Add this once cupynumeric.inv is implemented
        # a = inv(a)
        # n = abs(n)
        raise NotImplementedError("Negative exponent in matrix_power")

    # Fast paths
    if n == 1:
        return a.copy()
    elif n == 2:
        return matmul(a, a)
    elif n == 3:
        return matmul(matmul(a, a), a)

    # Use binary decomposition to reduce the number of matrix multiplications.
    # Here, we iterate over the bits of n, from LSB to MSB, raise `a` to
    # increasing powers of 2, and multiply into the result as needed.
    z: ndarray | None = None
    result: ndarray | None = None
    while n > 0:
        z = a if z is None else matmul(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else matmul(result, z)

    assert result is not None

    return result


# This implementation is adapted closely from NumPy
def multi_dot(
    arrays: Sequence[ndarray], *, out: ndarray | None = None
) -> ndarray:
    """
    Compute the dot product of two or more arrays in a single function call,
    while automatically selecting the fastest evaluation order.
    `multi_dot` chains `dot` and uses optimal parenthesization
    of the matrices.

    Parameters
    ----------
    arrays : Sequence[array_like]
        If the first argument is 1-D it is treated as a row vector.
        If the last argument is 1-D it is treated as a column vector.
        The other arguments must be 2-D.
    out : ndarray, optional
        Output argument. This must have the same shape and dtype that would be
        returned if it was not used.

    Returns
    -------
    output : ndarray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    numpy.linalg.multi_dot

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    arrays = [convert_to_cupynumeric_ndarray(x) for x in arrays]
    if out is not None:
        out = convert_to_cupynumeric_ndarray(out, share=True)

    n = len(arrays)
    # optimization only makes sense for len(arrays) > 2
    if n < 2:
        raise ValueError("multi_dot expects at least two arrays")
    elif n == 2:
        return dot(arrays[0], arrays[1], out=out)

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    # Explicitly convert vectors to 2D arrays to keep the logic of the internal
    # _multi_dot_* functions as simple as possible.
    if arrays[0].ndim == 1:
        arrays[0] = arrays[0][np.newaxis, :]
        if out is not None:
            out = out[np.newaxis, ...]
    if arrays[-1].ndim == 1:
        arrays[-1] = arrays[-1][:, np.newaxis]
        if out is not None:
            out = out[..., np.newaxis]
    for x in arrays:
        if x.ndim != 2:
            raise ValueError("Invalid shape for multi_dot input array")
    if out is not None and out.ndim != 2:
        raise ValueError("Invalid shape for multi_dot output array")

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)

    # return proper shape
    if ndim_first == 1 and ndim_last == 1:
        return result.reshape(())  # scalar
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()  # 1-D
    else:
        return result


def _multi_dot_three(
    A: ndarray, B: ndarray, C: ndarray, out: ndarray | None = None
) -> ndarray:
    """
    Find the best order for three arrays and do the multiplication.
    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)
    else:
        return dot(A, dot(B, C), out=out)


def _multi_dot_matrix_chain_order(
    arrays: Sequence[ndarray],
) -> npt.NDArray[np.int64]:
    """
    Return a `np.array` that encodes the optimal order of mutiplications.
    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.
    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.
        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])
    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = np.zeros((n, n), dtype=np.float64)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = np.empty((n, n), dtype=np.int64)

    for l_ in range(1, n):
        for i in range(n - l_):
            j = i + l_
            m[i, j] = np.inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return s


def _multi_dot(
    arrays: Sequence[ndarray],
    order: npt.NDArray[np.int64],
    i: int,
    j: int,
    out: ndarray | None = None,
) -> ndarray:
    """Actually do the multiplication with the given order."""
    if i == j:
        # the initial call with non-None out should never get here
        assert out is None

        return arrays[i]
    else:
        return dot(
            _multi_dot(arrays, order, i, order[i, j]),
            _multi_dot(arrays, order, order[i, j] + 1, j),
            out=out,
        )


# This implementation is adapted closely from NumPy
@add_boilerplate("x")
def norm(
    x: ndarray,
    ord: str | int | float | None = None,
    axis: int | tuple[int, int] | None = None,
    keepdims: bool = False,
) -> float | ndarray:
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``x.ravel`` will be returned.
    ord : ``{non-zero int, inf, -inf, 'fro', 'nuc'}``, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.
    axis : None or int or tuple[int, int], optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
        is None.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``x.ndim != 2``.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    See Also
    --------
    numpy.linalg.norm

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        ndim = x.ndim
        if (
            (ord is None)
            or (ord in ("f", "fro") and ndim == 2)
            or (ord == 2 and ndim == 1)
        ):
            x = x.ravel()
            if x.dtype.kind == "c":
                x_real = x.real
                x_imag = x.imag
                sqnorm = dot(x_real, x_real) + dot(x_imag, x_imag)
            else:
                sqnorm = dot(x, x)
            ret = _sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim * (1,))
            return ret

    if axis is None:
        computed_axis = tuple(range(x.ndim))
    else:
        computed_axis = normalize_axis_tuple(axis, x.ndim)

    for ax in computed_axis:
        if not isinstance(ax, int):
            raise TypeError(
                "`axis` must be None, an integer or a tuple of integers"
            )

    if len(computed_axis) == 1:
        if ord == np.inf:
            return abs(x).max(axis=computed_axis, keepdims=keepdims)
        elif ord == -np.inf:
            return abs(x).min(axis=computed_axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            return (
                (x != 0)
                .astype(x.dtype)
                .sum(axis=computed_axis, keepdims=keepdims)
            )
        elif ord == 1:
            # special case for speedup
            return add.reduce(abs(x), axis=computed_axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            s = (x.conj() * x).real
            return _sqrt(add.reduce(s, axis=computed_axis, keepdims=keepdims))
        # None of the str-type keywords for ord ("fro", "nuc")
        # are valid for vectors
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = abs(x)
            absx **= ord
            ret = add.reduce(absx, axis=computed_axis, keepdims=keepdims)
            ret **= 1 / ord
            return ret
    elif len(computed_axis) == 2:
        row_axis, col_axis = computed_axis
        row_axis = normalize_axis_index(row_axis, x.ndim)
        col_axis = normalize_axis_index(col_axis, x.ndim)
        if row_axis == col_axis:
            raise ValueError("Duplicate axes given")
        if ord == 2:
            raise NotImplementedError("2-norm requires SVD decomposition")
        elif ord == -2:
            raise NotImplementedError("-2-norm requires SVD decomposition")
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).max(axis=col_axis)
        elif ord == np.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -np.inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in [None, "fro", "f"]:
            squares = (x.conj() * x).real
            ret = _sqrt(squares.sum(axis=col_axis).sum(axis=row_axis))
        elif ord == "nuc":
            raise NotImplementedError(
                "nuclear norm requires SVD decomposition"
            )
        else:
            raise ValueError("Invalid norm order for matrices")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[computed_axis[0]] = 1
            ret_shape[computed_axis[1]] = 1
            ret = ret.reshape(tuple(ret_shape))
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm")


def _thunk_cholesky(a: ndarray) -> ndarray:
    """Cholesky decomposition.

    Return the Cholesky decomposition, `L * L.H`, of the square matrix `a`,
    where `L` is lower-triangular and .H is the conjugate transpose operator
    (which is the ordinary transpose if `a` is real-valued).  `a` must be
    Hermitian (symmetric if real-valued) and positive-definite. No
    checking is performed to verify whether `a` is Hermitian or not.
    In addition, only the lower-triangular and diagonal elements of `a`
    are used. Only `L` is actually returned.

    Parameters
    ----------
    a : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.

    Returns
    -------
    L : (..., M, M) array_like
        Upper or lower-triangular Cholesky factor of `a`.  Returns a
        matrix object if `a` is a matrix object.

    Notes
    -----
    The current implementation kills the process when the decomposition fails.

    See Also
    --------
    numpy.linalg.cholesky

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    input = a
    if input.dtype.kind not in ("f", "c"):
        input = input.astype("float64")
    output = ndarray._from_inputs(
        shape=input.shape, dtype=input.dtype, inputs=(input,)
    )
    output._thunk.cholesky(input._thunk)
    return output


def _thunk_eig(a: ndarray) -> tuple[ndarray, ...]:
    if a.dtype.kind not in ("f", "c"):
        a = a.astype("float64")

    if a.dtype == np.float32:
        complex_dtype = np.dtype(np.complex64)
    elif a.dtype == np.float64:
        complex_dtype = np.dtype(np.complex128)  # type: ignore
    elif a.dtype.kind in ("c"):
        complex_dtype = a.dtype
    else:
        raise TypeError("Eig input not supported (missing a conversion?)")

    if runtime.num_gpus > 0 and not runtime.cusolver_has_geev():
        a = ndarray._from_thunk(runtime.to_eager_array(a._thunk))
        out_ew = ndarray._from_thunk(
            runtime.create_eager_thunk(shape=a.shape[:-1], dtype=complex_dtype)
        )
        out_ev = ndarray._from_thunk(
            runtime.create_eager_thunk(shape=a.shape, dtype=complex_dtype)
        )
    else:
        out_ew = ndarray._from_inputs(
            shape=a.shape[:-1], dtype=complex_dtype, inputs=(a,)
        )
        out_ev = ndarray._from_inputs(
            shape=a.shape, dtype=complex_dtype, inputs=(a,)
        )

    if a.shape[-1] > 0:
        a._thunk.eig(out_ew._thunk, out_ev._thunk)
    return out_ew, out_ev


def _thunk_eigvals(a: ndarray) -> ndarray:
    if a.dtype.kind not in ("f", "c"):
        a = a.astype("float64")

    if a.dtype == np.float32:
        complex_dtype = np.dtype(np.complex64)
    elif a.dtype == np.float64:
        complex_dtype = np.dtype(np.complex128)  # type: ignore
    elif a.dtype.kind in ("c"):
        complex_dtype = a.dtype
    else:
        raise TypeError("Eigvals input not supported (missing a conversion?)")

    if runtime.num_gpus > 0 and not runtime.cusolver_has_geev():
        a = ndarray._from_thunk(runtime.to_eager_array(a._thunk))
        out_ew = ndarray._from_thunk(
            runtime.create_eager_thunk(shape=a.shape[:-1], dtype=complex_dtype)
        )
    else:
        out_ew = ndarray._from_inputs(
            shape=a.shape[:-1], dtype=complex_dtype, inputs=(a,)
        )

    if a.shape[-1] > 0:
        a._thunk.eigvals(out_ew._thunk)
    return out_ew


def _thunk_eigh(a: ndarray, uplo_l: bool) -> tuple[ndarray, ...]:
    if a.dtype.kind not in ("f", "c"):
        a = a.astype("float64")

    if a.dtype == np.complex64:
        real_dtype = np.dtype(np.float32)
    elif a.dtype == np.complex128:
        real_dtype = np.dtype(np.float64)  # type: ignore
    elif a.dtype.kind in ("f"):
        real_dtype = a.dtype
    else:
        raise TypeError("Eigh input not supported (missing a conversion?)")

    out_ew = ndarray._from_inputs(
        shape=a.shape[:-1], dtype=real_dtype, inputs=(a,)
    )
    out_ev = ndarray._from_inputs(shape=a.shape, dtype=a.dtype, inputs=(a,))

    if a.shape[-1] > 0:
        a._thunk.eigh(out_ew._thunk, out_ev._thunk, uplo_l)
    return out_ew, out_ev


def _thunk_eigvalsh(a: ndarray, uplo_l: bool) -> ndarray:
    if a.dtype.kind not in ("f", "c"):
        a = a.astype("float64")

    if a.dtype == np.complex64:
        real_dtype = np.dtype(np.float32)
    elif a.dtype == np.complex128:
        real_dtype = np.dtype(np.float64)  # type: ignore
    elif a.dtype.kind in ("f"):
        real_dtype = a.dtype
    else:
        raise TypeError("Eigvalsh input not supported (missing a conversion?)")

    out_ew = ndarray._from_inputs(
        shape=a.shape[:-1], dtype=real_dtype, inputs=(a,)
    )

    if a.shape[-1] > 0:
        a._thunk.eigvalsh(out_ew._thunk, uplo_l)
    return out_ew


def _thunk_qr(a: ndarray) -> tuple[ndarray, ...]:
    if a.dtype.kind not in ("f", "c"):
        a = a.astype("float64")

    k = min(a.shape[0], a.shape[1])

    out_q = ndarray._from_inputs(
        shape=(a.shape[0], k), dtype=a.dtype, inputs=(a,)
    )
    out_r = ndarray._from_inputs(
        shape=(k, a.shape[1]), dtype=a.dtype, inputs=(a,)
    )

    a._thunk.qr(out_q._thunk, out_r._thunk)
    return out_q, out_r


def _thunk_solve(
    a: ndarray, b: ndarray, output: ndarray | None = None
) -> ndarray:
    if a.dtype.kind not in ("f", "c"):
        a = a.astype("float64")
    if b.dtype.kind not in ("f", "c"):
        b = b.astype("float64")
    if a.dtype != b.dtype:
        dtype = np.result_type(a.dtype, b.dtype)
        a = a.astype(dtype)
        b = b.astype(dtype)

    if output is not None:
        if output.shape != b.shape:
            raise ValueError(
                f"Output shape mismatch: expected {b.shape}, but found {output.shape}"
            )
        elif output.dtype != b.dtype:
            raise TypeError(
                f"Output type mismatch: expected {b.dtype}, but found {output.dtype}"
            )

    expand_b = b.ndim == 1
    if expand_b:
        b = b.reshape((b.shape[0], 1))

    if output is not None:
        out = output.reshape(b.shape)
    else:
        out = ndarray._from_inputs(shape=b.shape, dtype=b.dtype, inputs=(a, b))

    out._thunk.solve(a._thunk, b._thunk)

    if expand_b:
        out = out.reshape((b.shape[0],))

    return out


def _thunk_svd(a: ndarray, full_matrices: bool) -> tuple[ndarray, ...]:
    if a.dtype.kind not in ("f", "c"):
        a = a.astype("float64")

    k = min(a.shape[0], a.shape[1])

    out_u = ndarray._from_inputs(
        shape=(a.shape[0], a.shape[0] if full_matrices else k),
        dtype=a.dtype,
        inputs=(a,),
    )

    real_dtype = a.dtype.type(0).real.dtype

    out_s = ndarray._from_inputs(shape=(k,), dtype=real_dtype, inputs=(a,))
    out_vh = ndarray._from_inputs(
        shape=(a.shape[1] if full_matrices else k, a.shape[1]),
        dtype=a.dtype,
        inputs=(a,),
    )

    a._thunk.svd(out_u._thunk, out_s._thunk, out_vh._thunk)
    return out_u, out_s, out_vh


# helper function to construct rational Pade
# numerator / denominator for expm(A):
#
def make_uv(A: ndarray, b: Any, m: int) -> tuple[ndarray, ndarray]:
    # 1 + floor(m/2):
    #
    k = 1 + m // 2
    n = A.shape[0]

    U = zeros((n, n), dtype=A.dtype)
    V = zeros((n, n), dtype=A.dtype)

    # U := A * ∑_{j=0, k} b_{2j+1} * A^{2j};
    # V := ∑_{j=0, k} b_{2j} * A^{2j};
    #
    A2 = matmul(A, A)
    A2k = eye(n, dtype=A.dtype)
    for j in range(k):
        U = U + b[2 * j + 1] * A2k
        V = V + b[2 * j] * A2k
        A2k = matmul(A2k, A2)

    U = matmul(A, U)

    return (U, V)


@add_boilerplate("a")
def pinv(a: ndarray, rtol: float = 1e-5) -> ndarray:
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be pseudo-inverted.
    rtol : float, optional
        Cutoff for small singular values relative to the largest singular value.
        Singular values less than or equal to ``rtol * largest_singular_value``
        are set to zero. Default: ``1e-5``.

    Returns
    -------
    B : (N, M) ndarray
        The pseudo-inverse of `a`.

    Raises
    ------
    NotImplementedError:
        If the dimension of the array is greater than 2 or if M < N
        for two-dimensional arrays

    LinAlgError:
        If the dimension of the array is less than 2


    See Also
    --------
    numpy.linalg.pinv : Similar function in NumPy

    Availability
    --------
    Single GPU, Single CPU

    Notes
    ------
    - The SVD part of the computation supports only single process execution
    - Does not support batched operations
    - Does not support `rcond` parameter (use `rtol` instead)
    - Does not support `hermitian` parameter (will be supported once
      cupynumeric.svd supports hermitian option)
    """
    shape = a.shape
    if len(shape) < 2:
        raise LinAlgError(
            f"{len(shape)}-dimensional array given. "
            "Array must be at least two-dimensional"
        )
    if len(shape) > 2:
        raise NotImplementedError(
            "cuPyNumeric does not yet support stacked 2d arrays"
        )
    if shape[0] < shape[1]:
        raise NotImplementedError("pinv is not supported for M < N")

    a = a.conj()
    u, s, vt = svd(a, full_matrices=False)
    tol = rtol * s.max(axis=-1, keepdims=True)
    s_inv = where(s > tol, 1.0 / s, 0.0)
    A_pinv = vt.T @ (s_inv[..., np.newaxis] * u.T)

    return A_pinv


class ExpmConstants:
    """
    Aggregates all the necessary expm(A) constants.
    """

    # Pade `b` coefficient generators
    # for both numerator `p(x)` and
    # denominator `q(x)` coefficients
    #
    # dictionary key := `m`, degree of
    # both `p(x)` and `q(x)` for
    # diagonal Pade implementation;
    #
    b_coeff = {
        3: np.array([120, 60, 12, 1], dtype=np.float64),
        5: np.array([30240, 15120, 3360, 420, 30, 1], dtype=np.float64),
        7: np.array(
            [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1],
            dtype=np.float64,
        ),
        9: np.array(
            [
                17643225600,
                8821612800,
                2075673600,
                302702400,
                30270240,
                2162160,
                110880,
                3960,
                90,
                1,
            ],
            dtype=np.float64,
        ),
        13: np.array(
            [
                64764752532480000,
                32382376266240000,
                7771770303897600,
                1187353796428800,
                129060195264000,
                10559470521600,
                670442572800,
                33522128640,
                1323241920,
                40840800,
                960960,
                16380,
                182,
                1,
            ],
            dtype=np.float64,
        ),
    }

    # Pade error control: absolute error tolerance
    # parameter `theta`, also degree `m` dependent:
    #
    theta = {3: 1.5e-2, 5: 2.5e-1, 7: 9.5e-1, 9: 2.1, 13: 5.4}

    # Taylor-18 coefficients
    #
    a01 = 0
    a11 = -0.10036558103014462001
    a21 = -0.00802924648241156960
    a31 = -0.00089213849804572995

    b01 = 0
    b11 = 0.39784974949964507614
    b21 = 1.36783778460411719922
    b31 = 0.49828962252538267755
    b61 = -0.00063789819459472330
    b02 = -10.9676396052962062593
    b12 = 1.68015813878906197182
    b22 = 0.05717798464788655127
    b32 = -0.00698210122488052084
    b62 = 0.00003349750170860705
    b03 = -0.09043168323908105619
    b13 = -0.06764045190713819075
    b23 = 0.06759613017704596460
    b33 = 0.02955525704293155274
    b63 = -0.00001391802575160607
    b04 = 0
    b14 = 0
    b24 = -0.09233646193671185927
    b34 = -0.01693649390020817171
    b64 = -0.00001400867981820361

    # Taylor-18 error control (squaring and scalling decision):
    #
    theta_m = 1.09


def expm_impl(a: ndarray, output: ndarray) -> tuple[int, int]:
    """
    Implements Pade rational aproximant of
    Algorithm 10.20, p.246-247 in
    "Functions of Matrices - Theory and Computation",
    Nicholas J. Higham, SIAM 2008.
    """

    lst_keys = list(ExpmConstants.theta.keys())

    # maximum polynomial degree for [p(x)/q(x)]:
    max_deg = lst_keys[-1]

    # L1 norm of matrix input:
    l1_norm_a = norm(a, 1)

    # loop decides which Pade degree, `m`, to
    # use, starting with the lowest degree
    # up to the one before last degree;
    #
    # if neither satisfies the theta tolerance
    # then exit the loop and proceed by using
    # m=max_deg degree + scaling (to achieve
    # desired tolerance);
    #
    requires_scaling = True
    s = 0
    a_scaled = a

    for m in lst_keys[0:-1]:
        tol_m = ExpmConstants.theta[m]
        b_arr = ExpmConstants.b_coeff[m]
        if l1_norm_a <= tol_m:
            requires_scaling = False
            break

    # at this point scaling + squaring with [max_deg/max_deg]
    # Pade rational approximation is done;
    #
    # using [max_deg/max_deg] Pade with scaling A/(2^s)
    # until || A / (2^s) ||_1 <= tol_13;
    # i.e., s = ceil(log_2(|| A / (2^s) ||_1)):
    #
    if requires_scaling:
        m = max_deg
        tol_m = ExpmConstants.theta[m]
        b_arr = ExpmConstants.b_coeff[m]

        s = np.maximum(1, int(np.ceil(np.log2(l1_norm_a / tol_m))))
        #
        # scale `a` by sfactor = 1.0/2^s = 2^(-s):
        #
        sfactor = np.power(2.0, s)
        #
        # A' <- A / sfactor
        #
        a_scaled = a / sfactor

    # evaluate U, V matrices, via Eq. 10.33 of [1]
    # k = 1 + floor(m/2):
    # U := A * ∑_{j=0, k} b_{2j+1} * A^{2j};
    # V := ∑_{j=0, k} b_{2j} * A^{2j};
    #
    (U, V) = make_uv(a_scaled, b_arr, m)
    A = V - U
    B = V + U

    # independently solve for each column:
    # TODO: can more parallelism be harvested here?
    #       at the very least avoid oversolving by
    #       doing LU / QR factorization once, followed
    #       by `n` backward-forward substitutions;
    #
    output[:] = solve(A, B)

    # if scaling by 1/2^s was done then
    # squaring s times is necessary:
    #
    if requires_scaling:
        for j in range(s):
            output[:] = matmul(output, output)

    return (m, s)


def expm_expl(a: ndarray, output: ndarray) -> tuple[int, int]:
    """
    Implements Taylor expansion, algorithm T_18
    in "Computing the Matrix Exponential with an
    Optimized Taylor Polynomial Approximation",
    Philipp Bader et. al.,
    which minimizes the number of matrix products
    for given number of terms in the expansion.
    """

    tol_m = ExpmConstants.theta_m  # may vary w/ degree, m, in future impls.

    # L1 norm of matrix input:
    l1_norm_a = norm(a, 1)

    requires_scaling = l1_norm_a > tol_m

    s = 0
    A = a
    m = 18

    if requires_scaling:
        s = np.maximum(1, int(np.ceil(np.log2(l1_norm_a / tol_m))))
        #
        # scale `a` by sfactor = 1.0/2^s = 2^(-s):
        #
        sfactor = np.power(2.0, s)
        #
        # A' <- A / sfactor
        #
        A = a / sfactor

    EYE = eye(A.shape[0], dtype=A.dtype)
    A2 = matmul(A, A)
    A3 = matmul(A2, A)
    A6 = matmul(A3, A3)
    B1 = (
        ExpmConstants.a11 * A + ExpmConstants.a21 * A2 + ExpmConstants.a31 * A3
    )
    B2 = (
        ExpmConstants.b11 * A
        + ExpmConstants.b21 * A2
        + ExpmConstants.b31 * A3
        + ExpmConstants.b61 * A6
    )
    B3 = (
        ExpmConstants.b02 * EYE
        + ExpmConstants.b12 * A
        + ExpmConstants.b22 * A2
        + ExpmConstants.b32 * A3
        + ExpmConstants.b62 * A6
    )
    B4 = (
        ExpmConstants.b03 * EYE
        + ExpmConstants.b13 * A
        + ExpmConstants.b23 * A2
        + ExpmConstants.b33 * A3
        + ExpmConstants.b63 * A6
    )
    B5 = (
        ExpmConstants.b24 * A2
        + ExpmConstants.b34 * A3
        + ExpmConstants.b64 * A6
    )

    A9 = B4 + matmul(B1, B5)
    B39 = B3 + A9

    output[:] = B2 + matmul(B39, A9)

    # if scaling by 1/2^s was done then
    # squaring s times is necessary:
    #
    if requires_scaling:
        for j in range(s):
            output[:] = matmul(output, output)

    return (m, s)


@add_boilerplate("a")
def expm(a: ndarray, method: str = "pade") -> ndarray:
    """
    Matrix exponential.

    Returns exp(A) for each (M x M) slice into a multi-dimensional
    array, assumed to be of shape (..., M, M);

    By default Pade (implicit) implementation is used.
    However, explicit Taylor(deg = 18) implementation can be used,
    by supplying additional flag `use_explicit = True`.

    Parameters
    ----------
    a : (..., M, M) array_like
        Input matrix or multi-dimensional array of shape (..., M, M).

    method : String method selector to use explicit ('taylor')
        or implicit ('pade'); default = 'pade'.

    Returns
    -------
    exp(A): matrix exponential of input, or a matrix exponential
        for each slice in the input.

    Notes
    -----
    Implicit Pade implementation is more stable but more computationally
    intensive than explicit Taylor, which is less stable when matrix norm is
    big enough. Also, Taylor can be slightly more performant for matrices of
    small enough norms, but more memory consuming.

    See Also
    --------
    scipy.linalg.expm

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if a.ndim < 2 or a.shape[-1] != a.shape[-2] or a.size <= 1:
        raise ValueError(f"Invalid input shape for expm: {a.shape}")

    output = zeros_like(a)

    m_info = get_machine()
    num_PEs = m_info.count()

    # run implicit (Pade) method by default:
    #
    if method == "pade":
        expm_func = expm_impl
    elif method == "taylor":
        expm_func = expm_expl
    else:
        raise ValueError(f"Method {method} not supported.")

    if num_PEs < 2:
        for idx in np.ndindex(a.shape[:-2]):
            mdeg, s = expm_func(a[idx], output[idx])
    else:
        for idx in np.ndindex(a.shape[:-2]):
            flat_index = np.ravel_multi_index(idx, a.shape[:-2])

            # assign work to multiple GPUs in round-robin way:
            #
            findx = int(flat_index)
            with m_info[findx % num_PEs]:
                mdeg, s = expm_func(a[idx], output[idx])

    return output


@add_boilerplate("a")
def tssvd(a: ndarray) -> tuple[ndarray, ...]:
    """
    Tall-skinny (TS) Singular Value Decomposition.

    Parameters
    ----------
    a : (M, N) array_like
        Array like, dimension 2.

    Returns
    -------
    u : (M, N) array_like
        Unitary array(s).
    s : (N) array_like
        The singular values, sorted in descending order
    vh : (N, N) array_like
        Unitary array(s).

    Raises
    ------
    LinAlgError
        If TS-SVD computation does not converge.

    Notes
    -----
    This routine is only efficient if ``M >> N``. In particular, it assumes that
    an ``(N, N)`` matrix can fit within a single processor memory.

    Implements the algorithm described in [1]_.

    Requires ``a.T @ a`` to not be singular.
    Input matrix must be non-singular.

    See Also
    --------
    numpy.linalg.svd

    Availability
    --------
    Multiple GPUs, Multiple CPUs


    References
    ----------
    .. [1] https://stanford.edu/~rezab/classes/cme323/S22/notes/L17/cme323_lec17.pdf
    """
    if a.ndim != 2 or a.size <= 1:
        raise ValueError(f"Invalid input shape for tssvd: {a.shape}")

    # A.T*A:
    #
    # TODO: Grammian API:
    a2 = a.transpose().conj() @ a

    # eigen-vals, eigen-vecs of A.T*A:
    #
    ew, ev = eigh(a2)

    if any(abs(ew) <= np.finfo(a.dtype).eps):
        raise LinAlgError("Singular matrix. Method cannot be applied.")

    # svals = map sqrt ew
    #
    svals = _sqrt(ew)

    # bring to standard form;
    # i.e., decreasing singular values
    #
    # generate index permutation, pi
    # via sort-by-key decreasingly:
    #
    d_indices = flip(argsort(svals))

    # V.T:
    #
    vt = ev.transpose().conj()

    reciprocal_svals = 1.0 / svals
    Sinv = diag(reciprocal_svals)

    # U = A*V*inv(S):
    #
    u = a @ (ev @ Sinv)

    # re-arrange svals decreasingly:
    #
    svals = svals[d_indices]

    # permute columns of U with pi:
    #
    # u = u[:, d_indices]
    u = u @ eye(u.shape[1])[d_indices].T

    # permute rows of V.T with pi:
    #
    vt = vt[d_indices]

    return u, svals, vt
