# copyright 2026 nvidia corporation
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
#

import os

import importlib.util
from typing import TextIO
from legate.util.has_started import runtime_has_started

HAVE_RICH: bool = importlib.util.find_spec("rich") is not None


def use_rich(stream: TextIO, *, start_runtime: bool = False) -> bool:
    if not (HAVE_RICH and stream.isatty()):
        return False
    use_rich = os.environ.get("LEGATE_BENCHMARK_USE_RICH", "1")
    if use_rich == "0":
        return False
    if start_runtime or runtime_has_started():
        from legate.core import get_legate_runtime

        runtime = get_legate_runtime()
        machine = runtime.get_machine()
        nodes = machine.get_node_range()
        num_nodes = nodes[1] - nodes[0]
        if num_nodes == 1:
            return True
        limit_stdout = os.environ.get("LEGATE_LIMIT_STDOUT", "0")
        if limit_stdout == "1":
            return True
        return False
    # no chance of more than one rank
    return True
