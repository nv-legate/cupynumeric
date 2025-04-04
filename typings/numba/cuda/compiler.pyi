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

from typing import Any, Callable, Dict, Optional, Tuple, Union

from numba.core.compiler import CompileResult
from numba.core.types import Type

def compile_ptx(
    pyfunc: Callable[[Any], Any],
    args: Any,
    debug: bool = False,
    lineinfo: bool = False,
    device: bool = False,
    fastmath: bool = False,
    cc: Optional[Any] = None,
    opt: bool = True,
) -> tuple[Any]: ...
def compile_cuda(
    pyfunc: Callable[[Any], Any],
    return_type: Type,
    args: Tuple[Type, ...],
    debug: bool = False,
    lineinfo: bool = False,
    inline: bool = False,
    fastmath: bool = False,
    nvvm_options: Optional[Dict[str, Optional[Union[str, int]]]] = None,
    cc: Optional[Tuple[int, int]] = None,
) -> CompileResult: ...
