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

from typing import Sequence

from llvmlite import ir
from numba.core.callconv import BaseCallConv
from numba.core.datamodel import ArgPacker
from numba.core.types import Type

class BaseContext:
    call_conv: BaseCallConv

    def create_module(self, name: str) -> ir.Module: ...
    def get_arg_packer(self, fe_args: Sequence[Type]) -> ArgPacker: ...
