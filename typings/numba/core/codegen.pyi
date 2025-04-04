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

from typing import Any

from llvmlite import ir

class CodeLibrary:
    codegen: "Codegen"
    name: str

    def add_linking_library(self, library: CodeLibrary) -> None: ...
    def add_ir_module(self, module: ir.Module) -> None: ...
    def finalize(self) -> None: ...

class Codegen:
    name: str

    def create_library(self, name: str, **kwargs: Any) -> CodeLibrary: ...
