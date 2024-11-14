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
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.errors import NoUri
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.logging import get_node_location, getLogger

from . import PARALLEL_SAFE, SphinxParallelSpec

log = getLogger(__name__)

SKIP = (
    "cupynumeric.cast",
    "cupynumeric.ndarray.__array_function__",
    "cupynumeric.ndarray.__array_ufunc__",
    "cupynumeric.ndarray.__format__",
    "cupynumeric.ndarray.__hash__",
    "cupynumeric.ndarray.__iter__",
    "cupynumeric.ndarray.__radd__",
    "cupynumeric.ndarray.__rand__",
    "cupynumeric.ndarray.__rdivmod__",
    "cupynumeric.ndarray.__reduce_ex__",
    "cupynumeric.ndarray.__rfloordiv__",
    "cupynumeric.ndarray.__rmod__",
    "cupynumeric.ndarray.__rmul__",
    "cupynumeric.ndarray.__ror__",
    "cupynumeric.ndarray.__rpow__",
    "cupynumeric.ndarray.__rsub__",
    "cupynumeric.ndarray.__rtruediv__",
    "cupynumeric.ndarray.__rxor__",
    "cupynumeric.ndarray.__sizeof__",
)

MISSING: list[tuple[str, str]] = []


class MissingRefs(SphinxPostTransform):
    default_priority = 5

    def run(self, **kwargs: Any) -> None:
        for node in self.document.findall(addnodes.pending_xref):
            self._check_target(node)

    def _check_target(self, node: Any) -> None:
        target = node["reftarget"]

        if not target.startswith("cupynumeric.") or target in SKIP:
            return

        domain = self.env.domains[node["refdomain"]]

        assert self.app.builder is not None

        try:
            uri = domain.resolve_xref(
                self.env,
                node.get("refdoc", self.env.docname),
                self.app.builder,
                node["reftype"],
                target,
                node,
                nodes.TextElement("", ""),
            )
        except NoUri:
            uri = None

        if uri is None:
            loc = get_node_location(node)
            log.warning(
                f"cuPyNumeric reference missing a target: {loc}: {target}",
                type="ref",
            )


def setup(app: Sphinx) -> SphinxParallelSpec:
    app.add_post_transform(MissingRefs)
    return PARALLEL_SAFE
