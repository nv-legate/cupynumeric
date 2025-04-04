# Copyright 2023 NVIDIA Corporation
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
import re
import typing

import pytest
from numba.types import Tuple, UniTuple, float32, float64, int32, int64

from cunumeric.numba_utils import compile_ptx_soa


@pytest.fixture
def addsub() -> typing.Callable:
    def addsub(x, y):
        return x + y, x - y

    return addsub


@pytest.fixture
def addsubmul() -> typing.Callable:
    def addsubmul(a, x, y):
        return a * (x + y), a * (x - y)

    return addsubmul


@pytest.fixture
def addsubconst() -> typing.Callable:
    def addsubconst(x, y):
        return x + y, x - y, 3

    return addsubconst


def param_pattern(
    function_name: str, param_index: int, param_type: str, reg_prefix: str
) -> str:
    """
    A helper function to generate patterns for recognizing parameter references
    in PTX functions - an example of a parameter reference looks like:

        ld.param.u64 %rd1, [addsubconst_param_0];

    These usually appear at the beginning of a function.
    """
    return (
        rf"ld\.param\.{param_type}"
        rf"\s+%{reg_prefix}[0-9]+,\s+"
        rf"\[{function_name}_param_"
        rf"{param_index}\]"
    )


def test_soa(addsub) -> None:
    # A basic test of compilation with an SoA interface

    # Compile for two int32 inputs and two int32 outputs
    signature = UniTuple(int32, 2)(int32, int32)
    ptx, resty = compile_ptx_soa(addsub, signature, device=True)

    # The function definition should use the name of the Python function
    fn_def_pattern = r"\.visible\s+\.func\s+addsub"
    assert re.search(fn_def_pattern, ptx)

    # The return type should match that of the signature's return type
    assert resty == signature.return_type

    # The function should have 4 parameters (numbered 0 to 3)
    assert re.search("addsub_param_3", ptx)
    assert not re.search("addsub_param_4", ptx)

    # The first two parameters should be treated as pointers (u64 values)
    assert re.search(param_pattern("addsub", 0, "u64", "rd"), ptx)
    assert re.search(param_pattern("addsub", 1, "u64", "rd"), ptx)

    # The remaining two parameters should be treated as 32 bit integers
    assert re.search(param_pattern("addsub", 2, "u32", "r"), ptx)
    assert re.search(param_pattern("addsub", 3, "u32", "r"), ptx)


def test_soa_fn_name(addsub) -> None:
    # Ensure that when a wrapper function name is specified, it is used in the
    # PTX.
    signature = UniTuple(int32, 2)(int32, int32)
    abi_info = {"abi_name": "addsub_soa"}
    ptx, resty = compile_ptx_soa(
        addsub, signature, device=True, abi_info=abi_info
    )
    fn_def_pattern = r"\.visible\s+\.func\s+addsub_soa"
    assert re.search(fn_def_pattern, ptx)


def test_soa_arg_types(addsub) -> None:
    # Ensure that specifying a different argument type is reflected
    # appropriately in the generated PTX
    signature = UniTuple(int32, 2)(int32, int64)
    ptx, resty = compile_ptx_soa(addsub, signature, device=True)

    # The final two parameters should now be a 32- and a 64-bit values
    # respectively. Note that the load of the last parameter may be an
    # instruction with a 32-bit destination type that effectively chops off the
    # upper 32 bits, so we cannot test for a load of a 64-bit value, which
    # would look like:
    #
    #    ld.param.u64 	%rd2, [addsub_param_3];
    #
    # but instead we'd potentially get
    #
    #    ld.param.u32 	%r2, [addsub_param_3];
    #
    # So we test the bit width of the parameters only:
    assert re.search(r".param\s+.b32\s+addsub_param_2", ptx)
    assert re.search(r".param\s+.b64\s+addsub_param_3", ptx)


def test_soa_more_args(addsubmul) -> None:
    # A test with three arguments, but only two return values

    signature = UniTuple(int32, 2)(int32, int32, int32)
    ptx, resty = compile_ptx_soa(addsubmul, signature, device=True)

    # The function should have 5 parameters (numbered 0 to 4)
    assert re.search("addsubmul_param_4", ptx)
    assert not re.search("addsubmul_param_5", ptx)

    # The first two parameters should be treated as pointers (u64 values)
    assert re.search(param_pattern("addsubmul", 0, "u64", "rd"), ptx)
    assert re.search(param_pattern("addsubmul", 1, "u64", "rd"), ptx)

    # The remaining three parameters should be treated as 32 bit integers
    assert re.search(param_pattern("addsubmul", 2, "u32", "r"), ptx)
    assert re.search(param_pattern("addsubmul", 3, "u32", "r"), ptx)
    assert re.search(param_pattern("addsubmul", 4, "u32", "r"), ptx)


def test_soa_more_returns(addsubconst) -> None:
    # Test with two arguments and three return values

    signature = UniTuple(int32, 3)(int32, int32)
    ptx, resty = compile_ptx_soa(addsubconst, signature, device=True)

    # The function should have 5 parameters (numbered 0 to 4)
    assert re.search("addsubconst_param_4", ptx)
    assert not re.search("addsubconst_param_5", ptx)

    # The first three parameters should be treated as pointers (u64 values)
    assert re.search(param_pattern("addsubconst", 0, "u64", "rd"), ptx)
    assert re.search(param_pattern("addsubconst", 1, "u64", "rd"), ptx)
    assert re.search(param_pattern("addsubconst", 2, "u64", "rd"), ptx)

    # The remaining two parameters should be treated as 32 bit integers
    assert re.search(param_pattern("addsubconst", 3, "u32", "r"), ptx)
    assert re.search(param_pattern("addsubconst", 4, "u32", "r"), ptx)


def test_soa_varying_types(addsub) -> None:
    # Argument types differ from each other and the return type

    signature = UniTuple(float64, 2)(int32, float32)
    ptx, resty = compile_ptx_soa(addsub, signature, device=True)

    # The first two parameters should be treated as pointers (u64 values)
    assert re.search(param_pattern("addsub", 0, "u64", "rd"), ptx)
    assert re.search(param_pattern("addsub", 1, "u64", "rd"), ptx)

    # The remaining two parameters should be a 32-bit integer and a 32-bit
    # float
    assert re.search(param_pattern("addsub", 2, "u32", "r"), ptx)
    assert re.search(param_pattern("addsub", 3, "f32", "f"), ptx)

    # There should be a 64-bit floating point store for the result
    assert re.search(r"st\.f64", ptx)


def test_soa_heterogeneous_return_type(addsubconst) -> None:
    # Test with return values of different types

    signature = Tuple((float32, float64, int32))(float32, float32)
    ptx, resty = compile_ptx_soa(addsubconst, signature, device=True)

    # There should be 32- and 64-bit floating point, and 32-bit integer stores
    # for the result
    assert re.search(r"st\.f32", ptx)
    assert re.search(r"st\.f64", ptx)
    assert re.search(r"st\.u32", ptx)


# Test of one return value

# Test of not putting device in

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
