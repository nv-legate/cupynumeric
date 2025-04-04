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

from typing import Iterable, Optional, Union

from llvmlite.ir.instructions import Instruction, Ret
from llvmlite.ir.values import Block, Value

class IRBuilder:
    def __init__(self, block: Optional[Block]): ...
    def ret(self, return_value: Value) -> Ret: ...
    def extract_value(
        self,
        agg: Value,
        idx: Union[Iterable[int], int],
        name: str = "",
    ) -> Instruction: ...
    def store(
        self, value: Value, ptr: Value, align: Optional[int] = None
    ) -> Instruction: ...
    def ret_void(self) -> Ret: ...
