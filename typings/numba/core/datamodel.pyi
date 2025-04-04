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

from typing import Sequence, Tuple

from llvmlite.ir import Argument
from llvmlite.ir.builder import IRBuilder
from llvmlite.ir.types import Type
from llvmlite.ir.values import Value

class ArgPacker:
    argument_types: Sequence[Type]

    def as_arguments(
        self, builder: IRBuilder, values: Tuple[Argument, ...]
    ) -> Tuple[Value]: ...
