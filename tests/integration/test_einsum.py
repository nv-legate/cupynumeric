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

from functools import lru_cache
from itertools import permutations, product

import numpy as np
import pytest
from legate.core.utils import OrderedSet
from utils.comparisons import allclose
from utils.generators import mk_0to1_array, permutes_to
from utils.utils import ONE_MAX_DIM_RANGE

import cupynumeric as num

# Limits for exhaustive expression generation routines
MAX_MODES = 3
MAX_OPERANDS = 2
MAX_OPERAND_DIM = 2
MAX_RESULT_DIM = 2
BASE_DIM_LEN = 10

ORDER = ("C", "F", "A", "K")


def gen_result(used_modes: int):
    for count in range(min(used_modes, MAX_RESULT_DIM) + 1):
        yield from permutations(range(used_modes), count)


def gen_operand(
    used_modes: int, dim_lim: int, mode_lim: int, op: list[int] | None = None
):
    if op is None:
        op = []
    # Yield the operand as constructed thus far
    yield op
    # Grow the operand, if we haven't hit the dimension limit yet
    if len(op) >= dim_lim:
        return
    # If we've hit the limit on distinct modes, only use modes
    # appearing on the same operand
    if len(op) == dim_lim - 1 and len(OrderedSet(op)) >= mode_lim:
        for m in sorted(OrderedSet(op)):
            op.append(m)
            yield from gen_operand(used_modes, dim_lim, mode_lim, op)
            op.pop()
        return
    # Reuse a mode previously encountered in the overall expression
    for m in range(used_modes):
        op.append(m)
        yield from gen_operand(used_modes, dim_lim, mode_lim, op)
        op.pop()
    # Add a new mode (number sequentially)
    if used_modes >= MAX_MODES:
        return
    op.append(used_modes)
    yield from gen_operand(used_modes + 1, dim_lim, mode_lim, op)
    op.pop()


# Exhaustively generate all (normalized) expressions within some limits. These
# limits are set low by default, to keep the unit test running time low.
def gen_expr(
    opers: list[list[int]] | None = None,
    cache: set[tuple[tuple[int]]] | None = None,
):
    if opers is None:
        opers = []
    if cache is None:
        cache = OrderedSet()
    # The goal here is to avoid producing duplicate expressions, up to
    # reordering of operands and alpha-renaming, e.g. the following
    # are considered equivalent (for the purposes of testing):
    #   a,b and a,c
    #   a,b and b,a
    #   ab,bb and aa,ab
    # Don't consider reorderings of arrays
    key = tuple(sorted(tuple(op) for op in opers))
    if key in cache:
        return
    cache.add(key)
    used_modes = max((m + 1 for op in opers for m in op), default=0)
    # Build an expression using the current list of operands
    if len(opers) > 0:
        lhs = ",".join("".join(chr(ord("a") + m) for m in op) for op in opers)
        for result in gen_result(used_modes):
            rhs = "".join(chr(ord("a") + m) for m in result)
            yield lhs + "->" + rhs
    # Add a new operand and recurse
    if len(opers) >= MAX_OPERANDS:
        return
    # Always put the longest operands first
    dim_lim = len(opers[-1]) if len(opers) > 0 else MAX_OPERAND_DIM
    # Between operands of the same length, put those with the most distinct
    # modes first.
    mode_lim = (
        len(OrderedSet(opers[-1])) if len(opers) > 0 else MAX_OPERAND_DIM
    )
    for op in gen_operand(used_modes, dim_lim, mode_lim):
        opers.append(op)
        yield from gen_expr(opers, cache)
        opers.pop()


# Selection of expressions beyond the limits of the exhaustive generation above
LARGE_EXPRS = [
    "ca,da,bc->db",
    "ad,ac,bd->bd",
    "ca,dc,da->ad",
    "ca,dc,ba->bd",
    "ba,dc,ad->ca",
    "ab,da,db->bd",
    "db,dc,ad->ab",
    "bc,ba,db->ba",
    "bc,cd,ab->da",
    "bd,cd,ca->db",
    "cd,cb,ca->ad",
    "adb,bdc,bac->cb",
    "cad,abd,bca->db",
    "cdb,dac,abd->ab",
    "cba,cad,dbc->bc",
    "abc,bda,bcd->bd",
    "dcb,acd,bac->bc",
    "dca,bca,cbd->ad",
    "dba,cbd,cab->cb",
    "cba,adb,dca->cb",
    "cdb,acd,cba->da",
    "bac,cad,cbd->bc",
    "dac,cbd,abc,abd->cb",
    "cab,dcb,cda,bca->ca",
    "dba,bca,cda,adc->dc",
    "acb,bda,dac,acd->db",
    "bcd,cba,adb,bca->cd",
    "cad,dab,cab,acb->ba",
    "abc,abd,acd,cba->ba",
    "cba,cda,bad,acb->db",
    "dca,cba,bdc,bad->cd",
    "cbd,abd,cad,adc->ca",
    "adc,bad,bcd,acb->cb",
]


# Selection of small expressions
SMALL_EXPRS = [
    "->",
    "a->",
    "a->a",
    "a,->",
    "a,->a",
    "a,a->",
    "a,a->a",
    "a,b->ab",
    "ab,ca->a",
    "ab,ca->b",
]


@lru_cache(maxsize=None)
def mk_input_default(lib, shape):
    return [mk_0to1_array(lib, shape)]


@lru_cache(maxsize=None)
def mk_input_that_permutes_to(lib, tgt_shape):
    return [
        mk_0to1_array(lib, src_shape).transpose(axes)
        for (axes, src_shape) in permutes_to(tgt_shape)
    ]


@lru_cache(maxsize=None)
def mk_input_that_broadcasts_to(lib, tgt_shape):
    # If an operand contains the same mode multiple times, then we can't set
    # just one of them to 1. Consider the operation 'aab->ab': (10,10,11),
    # (10,10,1), (1,1,11), (1,1,1) are all acceptable input shapes, but
    # (1,10,11) is not.
    tgt_sizes = list(sorted(OrderedSet(tgt_shape)))
    res = []
    for mask in product([True, False], repeat=len(tgt_sizes)):
        tgt2src_size = {
            d: (d if keep else 1) for (keep, d) in zip(mask, tgt_sizes)
        }
        src_shape = tuple(tgt2src_size[d] for d in tgt_shape)
        res.append(mk_0to1_array(lib, src_shape))
    return res


@lru_cache(maxsize=None)
def mk_typed_input(lib, shape):
    return [
        mk_0to1_array(lib, shape, np.float16),
        mk_0to1_array(lib, shape, np.float32),
        mk_0to1_array(lib, shape, np.complex64),
    ]


# Can't cache these, because they get overwritten by the operation
def mk_typed_output(lib, shape):
    return [lib.zeros(shape, np.float16), lib.zeros(shape, np.complex64)]


def check_np_vs_num(expr, mk_input, mk_output=None, **kwargs):
    lhs, rhs = expr.split("->")
    opers = lhs.split(",")
    in_shapes = [
        tuple(BASE_DIM_LEN + ord(m) - ord("a") for m in op) for op in opers
    ]
    out_shape = tuple(BASE_DIM_LEN + ord(m) - ord("a") for m in rhs)
    for np_inputs, num_inputs in zip(
        product(*(mk_input(np, sh) for sh in in_shapes)),
        product(*(mk_input(num, sh) for sh in in_shapes)),
    ):
        np_res = np.einsum(expr, *np_inputs, **kwargs)
        num_res = num.einsum(expr, *num_inputs, **kwargs)
        rtol = (
            1e-02
            if any(x.dtype == np.float16 for x in np_inputs)
            or kwargs.get("dtype") == np.float16
            else 1e-05
        )
        assert allclose(np_res, num_res, rtol=rtol)
        if mk_output is not None:
            for num_out in mk_output(num, out_shape):
                num.einsum(expr, *num_inputs, out=num_out, **kwargs)
                rtol_out = 1e-02 if num_out.dtype == np.float16 else rtol
                assert allclose(
                    num_out, num_res, rtol=rtol_out, check_dtype=False
                )


@pytest.mark.parametrize("expr", gen_expr())
def test_small(expr):
    check_np_vs_num(expr, mk_input_that_permutes_to)
    check_np_vs_num(expr, mk_input_that_broadcasts_to)


@pytest.mark.parametrize("expr", LARGE_EXPRS)
def test_large(expr):
    check_np_vs_num(expr, mk_input_default)


@pytest.mark.parametrize("expr", SMALL_EXPRS)
@pytest.mark.parametrize("dtype", [None, np.float32])
def test_cast(expr, dtype):
    check_np_vs_num(
        expr, mk_typed_input, mk_typed_output, dtype=dtype, casting="unsafe"
    )


@pytest.mark.parametrize("optimize", [False, "optimal", "greedy", True])
def test_optimize(optimize):
    a = np.random.rand(256, 256)
    b = np.random.rand(256, 256)

    np_res = np.einsum("ik,kj->ij", a, b, optimize=optimize)
    num_res = num.einsum("ik,kj->ij", a, b, optimize=optimize)
    assert allclose(np_res, num_res)


def test_expr_opposite():
    a = np.random.rand(256, 256)
    b = np.random.rand(256, 256)

    expected_exc = ValueError
    with pytest.raises(expected_exc):
        np.einsum("ik,kj=>ij", a, b)
        # Numpy raises ValueError: invalid subscript '=' in einstein
        # sum subscripts string, subscripts must be letters
    with pytest.raises(expected_exc):
        num.einsum("ik,kj=>ij", a, b)
        # cuPyNumeric raises ValueError: Subscripts can only contain one '->'


@pytest.mark.xfail
@pytest.mark.parametrize("order", ORDER)
def test_order(order):
    a = np.random.rand(256, 256)
    b = np.random.rand(256, 256)
    np_res = np.einsum("ik,kj->ij", a, b, order=order)
    num_res = num.einsum("ik,kj->ij", a, b, order=order)
    # cuNmeric raises TypeError: einsum() got an unexpected keyword
    # argument 'order'
    assert allclose(np_res, num_res)


def test_negative() -> None:
    a = np.random.rand(256, 256)
    b = np.random.rand(256, 256)
    msg = r"invalid subscript"
    with pytest.raises(ValueError, match=msg):
        np.einsum("ik,1j->ij", a, b)
    msg = r"Non-alphabetic mode labels"
    with pytest.raises(NotImplementedError, match=msg):
        num.einsum("ik,1j->ij", a, b)


@pytest.mark.parametrize("ndim", ONE_MAX_DIM_RANGE)
def test_large_arrays_high_dimensions(ndim):
    """Test einsum on large arrays with varying dimensions from ONE_MAX_DIM_RANGE."""
    # Create reasonably sized dimensions to avoid memory issues while still testing large arrays
    # For higher dimensions, use smaller per-dimension size
    if ndim <= 2:
        dim_size = 32  # 32x32 for 2D = 1,024 elements
    elif ndim <= 4:
        dim_size = 8  # 8^4 = 4,096 elements
    elif ndim <= 6:
        dim_size = 4  # 4^6 = 4,096 elements
    else:
        dim_size = 2  # 2^n for very high dimensions

    shape = (dim_size,) * ndim

    # Test Case 1: Simple trace-like operation (sum over all dimensions)
    # This creates an einsum expression that contracts all dimensions
    if ndim == 1:
        # For 1D: just sum all elements
        expr = "i->"
        np_a = np.random.rand(*shape).astype(np.float32)
        num_a = num.array(np_a)

        np_result = np.einsum(expr, np_a)
        num_result = num.einsum(expr, num_a)
        assert allclose(np_result, num_result, rtol=1e-5)

    elif ndim == 2:
        # For 2D: test matrix multiplication and trace
        np_a = np.random.rand(*shape).astype(np.float32)
        np_b = np.random.rand(*shape).astype(np.float32)
        num_a = num.array(np_a)
        num_b = num.array(np_b)

        # Matrix multiplication: ik,kj->ij
        np_result = np.einsum("ik,kj->ij", np_a, np_b)
        num_result = num.einsum("ik,kj->ij", num_a, num_b)
        assert allclose(np_result, num_result, rtol=1e-5)

        # Trace: ii->
        np_result = np.einsum("ii->", np_a)
        num_result = num.einsum("ii->", num_a)
        assert allclose(np_result, num_result, rtol=1e-5)

    else:
        # For higher dimensions: test tensor contraction
        # Create two tensors with same shape
        np_a = np.random.rand(*shape).astype(np.float32)
        np_b = np.random.rand(*shape).astype(np.float32)
        num_a = num.array(np_a)
        num_b = num.array(np_b)

        # Create einsum expression for element-wise multiplication and sum
        # e.g., for 3D: "ijk,ijk->" (sum of element-wise product)
        indices = "".join(chr(ord("i") + i) for i in range(ndim))
        expr = f"{indices},{indices}->"

        np_result = np.einsum(expr, np_a, np_b)
        num_result = num.einsum(expr, num_a, num_b)
        assert allclose(np_result, num_result, rtol=1e-5)

        # Test partial contraction - contract first dimension only
        # e.g., for 3D: "ijk,ijk->jk"
        if ndim >= 2:
            remaining_indices = indices[1:]
            expr_partial = f"{indices},{indices}->{remaining_indices}"

            np_result_partial = np.einsum(expr_partial, np_a, np_b)
            num_result_partial = num.einsum(expr_partial, num_a, num_b)
            assert allclose(np_result_partial, num_result_partial, rtol=1e-5)

    # Test Case 2: Single array operations (sum over specific dimensions)
    np_a = np.random.rand(*shape).astype(np.float32)
    num_a = num.array(np_a)

    # Sum over first dimension
    if ndim >= 2:
        indices = "".join(chr(ord("i") + i) for i in range(ndim))
        remaining_indices = indices[1:]  # Remove first index
        sum_expr = f"{indices}->{remaining_indices}"

        np_sum_result = np.einsum(sum_expr, np_a)
        num_sum_result = num.einsum(sum_expr, num_a)
        assert allclose(np_sum_result, num_sum_result, rtol=1e-5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
