#!/usr/bin/env python
"""
Test binary operations inside leaf tasks as inline execution.
"""

from typing import Callable

import numpy as np
from legate.core.task import task, InputStore, OutputStore
from legate.core import VariantCode
import cupynumeric as cn


@task(variants=(VariantCode.CPU, VariantCode.GPU, VariantCode.OMP))
def binary_add(a: InputStore, b: InputStore, out: OutputStore):
    arr_a = cn.asarray(a)
    arr_b = cn.asarray(b)
    arr_out = cn.asarray(out)
    cn.add(arr_a, arr_b, out=arr_out)


@task(variants=(VariantCode.CPU, VariantCode.GPU, VariantCode.OMP))
def binary_subtract(a: InputStore, b: InputStore, out: OutputStore):
    arr_a = cn.asarray(a)
    arr_b = cn.asarray(b)
    arr_out = cn.asarray(out)
    cn.subtract(arr_a, arr_b, out=arr_out)


@task(variants=(VariantCode.CPU, VariantCode.GPU, VariantCode.OMP))
def binary_multiply(a: InputStore, b: InputStore, out: OutputStore):
    arr_a = cn.asarray(a)
    arr_b = cn.asarray(b)
    arr_out = cn.asarray(out)
    cn.multiply(arr_a, arr_b, out=arr_out)


@task(variants=(VariantCode.CPU, VariantCode.GPU, VariantCode.OMP))
def binary_divide(a: InputStore, b: InputStore, out: OutputStore):
    arr_a = cn.asarray(a)
    arr_b = cn.asarray(b)
    arr_out = cn.asarray(out)
    cn.divide(arr_a, arr_b, out=arr_out)


def _make_binary_test(task_func: Callable, expected_func: Callable) -> None:
    a_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b_data = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)

    a = cn.array(a_data)
    b = cn.array(b_data)
    out = cn.empty_like(a)

    task_func(a, b, out)

    result = np.array(out)
    expected = expected_func(a_data, b_data)
    np.testing.assert_allclose(result, expected)


def test_add():
    _make_binary_test(binary_add, lambda a, b: a + b)


def test_subtract():
    _make_binary_test(binary_subtract, lambda a, b: a - b)


def test_multiply():
    _make_binary_test(binary_multiply, lambda a, b: a * b)


def test_divide():
    _make_binary_test(binary_divide, lambda a, b: a / b)
