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

from typing import Any

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.logging import getLogger

import cupynumeric as cn

from .._utils.coverage import is_implemented
from . import PARALLEL_SAFE, SphinxParallelSpec
from ._cupynumeric_directive import CupynumericDirective

log = getLogger(__name__)


def _filter(x: Any) -> bool:
    return (
        callable(x)
        and is_implemented(x)
        and (x.__name__.startswith("__") or not x.__name__.startswith("_"))
    )


namespaces = (cn, cn.fft, cn.linalg, cn.random)


class ImplementedIndex(CupynumericDirective):
    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        refs: list[str] = []
        for ns in namespaces:
            refs += [
                f"* :obj:`{ns.__name__}.{x.__name__}`"
                for n, x in ns.__dict__.items()
                if _filter(x)
            ]
        refs += [
            f"* :obj:`cupynumeric.ndarray.{x.__name__}`"
            for x in cn.ndarray.__dict__.values()
            if _filter(x)
        ]

        rst_text = "\n".join(sorted(set(refs)))

        log.debug(rst_text)

        return self.parse(rst_text, "<implemented-index>")


def setup(app: Sphinx) -> SphinxParallelSpec:
    app.add_directive_to_domain("py", "implemented-index", ImplementedIndex)
    return PARALLEL_SAFE
