#!/usr/bin/env python
"""
Test unary operations inside leaf tasks as inline execution.
"""

import numpy as np
from legate.core.task import task, InputStore, OutputStore
from legate.core import VariantCode
import cupynumeric as cn


@task(variants=(VariantCode.CPU, VariantCode.GPU, VariantCode.OMP))
def unary_negation(inp: InputStore, out: OutputStore):
    arr_in = cn.asarray(inp)
    arr_out = cn.asarray(out)
    cn.negative(arr_in, out=arr_out)


@task(variants=(VariantCode.CPU, VariantCode.GPU, VariantCode.OMP))
def unary_sqrt(inp: InputStore, out: OutputStore):
    arr_in = cn.asarray(inp)
    arr_out = cn.asarray(out)
    cn.sqrt(arr_in, out=arr_out)


@task(variants=(VariantCode.CPU, VariantCode.GPU, VariantCode.OMP))
def unary_exp(inp: InputStore, out: OutputStore):
    arr_in = cn.asarray(inp)
    arr_out = cn.asarray(out)
    cn.exp(arr_in, out=arr_out)


def test_negation():
    data = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)
    expected = -data

    a = cn.array(data)
    b = cn.empty_like(a)

    unary_negation(a, b)

    result = np.array(b)
    np.testing.assert_allclose(result, expected)


def test_sqrt():
    data = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float32)
    expected = np.sqrt(data)

    a = cn.array(data)
    b = cn.empty_like(a)

    unary_sqrt(a, b)

    result = np.array(b)
    np.testing.assert_allclose(result, expected)


def test_exp():
    data = np.array([0.0, 1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    expected = np.exp(data)

    a = cn.array(data)
    b = cn.empty_like(a)

    unary_exp(a, b)

    result = np.array(b)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
