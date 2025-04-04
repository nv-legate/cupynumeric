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
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from llvmlite import ir
from llvmlite.ir.builder import IRBuilder
from llvmlite.ir.instructions import Ret
from llvmlite.ir.types import FunctionType
from llvmlite.ir.values import Value
from numba import types
from numba.core import sigutils
from numba.core.base import BaseContext
from numba.core.callconv import BaseCallConv
from numba.core.codegen import CodeLibrary
from numba.core.compiler_lock import global_compiler_lock
from numba.core.funcdesc import FunctionDescriptor
from numba.core.typing.templates import Signature
from numba.cuda.codegen import CUDACodeLibrary
from numba.cuda.compiler import compile_cuda


class SoACallConv(BaseCallConv):
    """
    Calling convention where returned values are stored through pointers
    provided as arguments.

    - If the return type is a scalar, the first argument is a pointer to the
      return type.
    - If the return type is a tuple of length N, then the first N arguments are
      pointers to each of the elements of the tuple.

    In equivalent C, the prototype of a function with this calling convention
    would take the following form:

        void <func_name>(<Tuple item 1>*, ..., <Tuple item N>*,
                         <Python arguments... >);
    """

    def _make_call_helper(self, builder: Any) -> None:
        # Call helpers are used for the exception implementation. This is not
        # needed when only wrapping functions.
        msg = "Python exceptions are unsupported when returning in SoA form"
        raise NotImplementedError(msg)

    def return_value(self, builder: IRBuilder, retval: Value) -> Ret:
        return builder.ret(retval)

    def return_user_exc(
        self,
        builder: IRBuilder,
        exc: Any,
        exc_args: Any = None,
        loc: Any = None,
        func_name: Any = None,
    ) -> None:
        msg = "Python exceptions are unsupported when returning in SoA form"
        raise NotImplementedError(msg)

    def return_status_propagate(self, builder: IRBuilder, status: Any) -> None:
        msg = "Return status is unsupported when returning in SoA form"
        raise NotImplementedError(msg)

    def get_function_type(
        self, restype: types.Type, argtypes: Iterable[types.Type]
    ) -> FunctionType:
        """
        Get the LLVM IR Function type for *restype* and *argtypes*.
        """
        arginfo = self._get_arg_packer(argtypes)
        be_argtypes = list(arginfo.argument_types)
        if isinstance(restype, types.BaseTuple):
            return_types = [self.get_return_type(t) for t in restype.types]
        else:
            return_types = [self.get_return_type(restype)]
        fnty = ir.FunctionType(ir.VoidType(), return_types + be_argtypes)
        return fnty

    def decorate_function(
        self,
        fn: Callable[[Any], Any],
        args: Iterable[str],
        fe_argtypes: List[types.Type],
        noalias: bool = False,
    ) -> None:
        """
        Set names and attributes of function arguments.
        """
        raise NotImplementedError("Function decoration not used for SoA ABI")

    def get_arguments(
        self, func: ir.Function, restype: types.Type
    ) -> Tuple[ir.Argument, ...]:
        """
        Get the Python-level arguments of LLVM *func*.
        """
        if isinstance(restype, types.BaseTuple):
            n_returns = len(restype.types)
        else:
            n_returns = 1

        return func.args[n_returns:]

    def call_function(
        self,
        builder: ir.IRBuilder,
        callee: ir.Function,
        resty: types.Type,
        argtys: Iterable[types.Type],
        args: Iterable[ir.Value],
        attrs: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[ir.Value, ir.Value]:
        """
        Call the Numba-compiled *callee*.
        """
        raise NotImplementedError("Can't call SoA return function directly")


def soa_wrap_function(
    context: BaseContext,
    lib: CodeLibrary,
    fndesc: FunctionDescriptor,
    nvvm_options: Dict[str, Union[int, str, None]],
    wrapper_name: str,
) -> CUDACodeLibrary:
    """
    Wrap a Numba ABI function such that it returns tuple values into SoA
    arguments.
    """
    new_library = lib.codegen.create_library(
        f"{lib.name}_function_",
        entry_name=wrapper_name,
        nvvm_options=nvvm_options,
    )
    library = cast(CUDACodeLibrary, new_library)
    library.add_linking_library(lib)

    # Determine the caller (C ABI) and wrapper (Numba ABI) function types
    argtypes = fndesc.argtypes
    restype = fndesc.restype
    soa_call_conv = SoACallConv(context)
    wrapperty = soa_call_conv.get_function_type(restype, argtypes)
    calleety = context.call_conv.get_function_type(restype, argtypes)

    # Create a new module and declare the callee
    wrapper_module = context.create_module("cuda.soa.wrapper")
    callee = ir.Function(wrapper_module, calleety, fndesc.llvm_func_name)

    # Define the caller - populate it with a call to the callee and return
    # its return value

    wrapper = ir.Function(wrapper_module, wrapperty, wrapper_name)
    builder = ir.IRBuilder(wrapper.append_basic_block(""))

    arginfo = context.get_arg_packer(argtypes)
    wrapper_args = soa_call_conv.get_arguments(wrapper, restype)
    callargs = arginfo.as_arguments(builder, wrapper_args)
    # We get (status, return_value), but we ignore the status since we
    # can't propagate it through the SoA ABI anyway
    _, return_value = context.call_conv.call_function(
        builder, callee, restype, argtypes, callargs
    )

    if isinstance(restype, types.BaseTuple):
        for i in range(len(restype.types)):
            val = builder.extract_value(return_value, i)
            builder.store(val, wrapper.args[i])
    else:
        builder.store(return_value, wrapper.args[0])
    builder.ret_void()

    library.add_ir_module(wrapper_module)
    library.finalize()
    return library


@global_compiler_lock
def compile_ptx_soa(
    pyfunc: Callable[..., Any],
    sig: Union[Tuple[types.Type], str, Signature],
    debug: bool = False,
    lineinfo: bool = False,
    device: bool = False,
    fastmath: bool = False,
    cc: Optional[Tuple[int, int]] = None,
    opt: bool = True,
    abi_info: Optional[Dict[str, str]] = None,
) -> Tuple[str, types.Type]:
    # This is just a copy of Numba's compile_ptx, with a modification to return
    # values as SoA and some simplifications to keep it short
    if not device:
        raise NotImplementedError(
            "Only device functions can be compiled for the SoA ABI"
        )

    nvvm_options: Dict[str, Union[int, str, None]] = {
        "fastmath": fastmath,
        "opt": 3 if opt else 0,
    }

    # Use the Python function name as the function name in PTX if it is not
    # specified - otherwise, use the specified name.
    if abi_info:
        wrapper_name = abi_info["abi_name"]
    else:
        wrapper_name = pyfunc.__name__

    args, return_type = sigutils.normalize_signature(sig)

    # Default to Compute Capability 5.0 if not specified
    cc = cc or (5, 0)

    cres = compile_cuda(
        pyfunc,
        return_type,
        args,
        debug=debug,
        lineinfo=lineinfo,
        fastmath=fastmath,
        nvvm_options=nvvm_options,
        cc=cc,
    )

    lib = soa_wrap_function(
        cres.target_context,
        cres.library,
        cres.fndesc,
        nvvm_options,
        wrapper_name,
    )

    ptx = lib.get_asm_str(cc=cc)
    resty = cres.signature.return_type

    return ptx, resty
