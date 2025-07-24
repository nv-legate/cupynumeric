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
#
from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst.directives import choice
from sphinx.application import Sphinx
from sphinx.util.logging import getLogger

from . import PARALLEL_SAFE, SphinxParallelSpec
from ._comparison_config import GROUPED_CONFIGS, NUMPY_CONFIGS
from ._comparison_util import generate_section
from ._cupynumeric_directive import CupynumericDirective
from ._templates import COMPARISON_TABLE

log = getLogger(__name__)


class ComparisonTable(CupynumericDirective):
    has_content = False
    required_arguments = 0
    optional_arguments = 1

    option_spec = {"sections": lambda x: choice(x, ("numpy", "grouped"))}

    def run(self) -> list[nodes.Node]:
        if self.options.get("sections", "numpy") == "numpy":
            section_configs = NUMPY_CONFIGS
        else:
            section_configs = GROUPED_CONFIGS

        sections = [generate_section(config) for config in section_configs]

        rst_text = COMPARISON_TABLE.render(sections=sections)
        log.debug(rst_text)

        return self.parse(rst_text, "<comparison-table>")


def setup(app: Sphinx) -> SphinxParallelSpec:
    app.add_directive_to_domain("py", "comparison-table", ComparisonTable)
    return PARALLEL_SAFE
