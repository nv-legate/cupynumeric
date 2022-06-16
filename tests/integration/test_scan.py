# Copyright 2021-2022 NVIDIA Corporation
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


import numpy as np
import pytest
import cunumeric as num

np.random.seed(12345)


def _gen_array(n0, shape, dt, axis, outtype):
    # range 1-100, avoiding zeros to ensure correct testing for int prod case
    A = (99 * np.random.random(shape) + 1).astype(dt)
    if n0 == "first_half":
        # second element along all axes is a NAN
        if len(shape) == 1:
            A[1] = np.nan
        elif len(shape) == 2:
            A[1,1] = np.nan
        elif len(shape) == 3:
            A[1,1,1] = np.nan
    elif n0 == "second_half":
        # second from last element along all axes is a NAN
        if len(shape) == 1:
            A[shape[0]-2] = np.nan
        elif len(shape) == 2:
            A[shape[0]-2,shape[1]-2] = np.nan
        elif len(shape) == 3:
            A[shape[0]-2,shape[1]-2,shape[2]-2] = np.nan
    if outtype is None:
        B = None
        C = None
    else:
        if axis is None:
            B = np.zeros(shape=A.size, dtype=outtype)
            C = np.zeros(shape=A.size, dtype=outtype)
        else:
            B = np.zeros(shape=shape, dtype=outtype)
            C = np.zeros(shape=shape, dtype=outtype)
    return A, B, C


def _run_tests(op, n0, shape, dt, axis, out0, outtype):
    if n0 is None:
        str_n0 = "None"
    else:
        str_n0 = n0
    if axis is None:
        str_axis = "None"
    else:
        str_axis = str(axis)
    print("Running test: " + op + ", shape: " + str(shape) + ", nan location: " + str_n0 + ", axis: " + str_axis + ", in type: " + str(dt) + ", out type: " + str(outtype) + ", output array not provided: " + str(out0))
    if out0 == True:
        A, B, C = _gen_array(n0, shape, dt, axis, None)
        if op == "cumsum":
            B = num.cumsum(A, out = None, axis=axis, dtype=outtype)
            C = np.cumsum(A, out = None, axis=axis, dtype=outtype)
        elif op == "cumprod":
            B = num.cumprod(A, out = None, axis=axis, dtype=outtype)
            C = np.cumprod(A, out = None, axis=axis, dtype=outtype)
        elif op == "nancumsum":
            B = num.nancumsum(A, out = None, axis=axis, dtype=outtype)
            C = np.nancumsum(A, out = None, axis=axis, dtype=outtype)
        elif op == "nancumprod":
            B = num.nancumprod(A, out = None, axis=axis, dtype=outtype)
            C = np.nancumprod(A, out = None, axis=axis, dtype=outtype)
    else:
        A, B, C = _gen_array(n0, shape, dt, axis, outtype)
        if op == "cumsum":
            num.cumsum(A, out = B, axis=axis, dtype=outtype)
            np.cumsum(A, out = C, axis=axis, dtype=outtype)
        elif op == "cumprod":
            num.cumprod(A, out = B, axis=axis, dtype=outtype)
            np.cumprod(A, out = C, axis=axis, dtype=outtype)
        elif op == "nancumsum":
            num.nancumsum(A, out = B, axis=axis, dtype=outtype)
            np.nancumsum(A, out = C, axis=axis, dtype=outtype)
        elif op == "nancumprod":
            num.nancumprod(A, out = B, axis=axis, dtype=outtype)
            np.nancumprod(A, out = C, axis=axis, dtype=outtype)

    print("Checking result...")
    if np.allclose(B, C, equal_nan=True):
        print("PASS!")
    else:
        print("FAIL!")
        print("INPUT    : " + str(A))
        print("CUNUMERIC: " + str(B))
        print("NUMPY    : " + str(C))
        assert False


def test_scan():
    ops = ["cumsum",
           "cumprod",
           "nancumsum",
           "nancumprod",]
    n0s = ["first_half",
          "second_half",]
    shapes = [[10000],
              [1000,1000],
              [100,100,100],]
    int_types = [np.int32,
                 np.int64,]
    float_types = [np.float32,
                   np.float64,]
    axes = [None,
             0,]
    out0s = [True,
            False,]
    for op in ops:
        for shape in shapes:
            for axis in axes:
                for out0 in out0s:
                    for outtype in int_types:
                        for dt in int_types:
                            _run_tests(op, None, shape, dt, axis, out0, outtype)
                        for dt in float_types:
                            for n0 in n0s:
                                _run_tests(op, n0, shape, dt, axis, out0, outtype)
                    for outtype in float_types:
                        for dt in int_types:
                            _run_tests(op, None, shape, dt, axis, out0, outtype)
                        for dt in float_types:
                            for n0 in n0s:
                                _run_tests(op, n0, shape, dt, axis, out0, outtype)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
