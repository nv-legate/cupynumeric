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

from typing import Iterable, Optional, Tuple

from llvmlite.ir.builder import IRBuilder
from llvmlite.ir.types import FunctionType, PointerType, Type as LLType
from llvmlite.ir.values import Function, Value
from numba.core.base import BaseContext
from numba.core.datamodel import ArgPacker
from numba.core.types import Type

class BaseCallConv:
    def __init__(self, context: BaseContext): ...
    def _get_arg_packer(self, argtypes: Iterable[Type]) -> ArgPacker: ...
    def get_return_type(self, ty: Type) -> LLType: ...
    def get_function_type(
        self, restype: Type, argtypes: Iterable[Type]
    ) -> FunctionType: ...
    def call_function(
        self,
        builder: IRBuilder,
        callee: Function,
        resty: Type,
        argtys: Iterable[Type],
        args: Iterable[Value],
        attrs: Optional[Tuple[str, ...]] = None,
    ) -> Tuple[Value, Value]: ...
