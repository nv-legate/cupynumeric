# Copyright 2022 NVIDIA Corporation
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

from dataclasses import dataclass
from types import ModuleType

from docutils.parsers.rst.directives import choice
from sphinx.util.logging import getLogger

from ..coverage import is_implemented, is_multi, is_single
from ._comparison_config import (
    GROUPED_CONFIGS,
    MISSING_NP_REFS,
    NUMPY_CONFIGS,
    SKIP,
)
from ._cunumeric_directive import CunumericDirective
from ._templates import COMPARISON_TABLE

log = getLogger(__name__)


@dataclass(frozen=True)
class ItemDetail:
    name: str
    implemented: bool
    np_ref: str
    lg_ref: str
    single: str
    multi: str


@dataclass(frozen=True)
class SectionDetail:
    title: str
    np_count: int
    lg_count: int
    items: list[ItemDetail]


def _npref(name, obj):
    if isinstance(obj, ModuleType):
        full_name = f"{obj.__name__}.{name}"
    else:
        full_name = f"numpy.{obj.__name__}.{name}"

    role = "meth" if "ndarray" in full_name else "obj"

    if full_name in MISSING_NP_REFS:
        return f"``{full_name}``"
    return f":{role}:`{full_name}`"


def _lgref(name, obj, implemented):
    if not implemented:
        return "-"

    if isinstance(obj, ModuleType):
        full_name = f"{obj.__name__}.{name}"
    else:
        full_name = f"cunumeric.{obj.__name__}.{name}"

    role = "meth" if "ndarray" in full_name else "obj"

    return f":{role}:`{full_name}`"


def _filter_names(obj, types=None):
    names = (n for n in dir(obj))  # every name in the module or class
    names = (n for n in names if n not in SKIP)  # except the ones we skip
    names = (n for n in names if not n.startswith("_"))  # or any private names
    if types:
        # optionally filtered by type
        names = (n for n in names if isinstance(getattr(obj, n), types))
    return names


def _get_item(name, np_obj, lg_obj):
    lg_attr = getattr(lg_obj, name)

    implemented = is_implemented(lg_attr)

    if implemented:
        single = "YES" if is_single(lg_attr) else "NO"
        multi = "YES" if is_multi(lg_attr) else "NO"
    else:
        single = multi = ""

    return ItemDetail(
        name=name,
        implemented=implemented,
        np_ref=_npref(name, np_obj),
        lg_ref=_lgref(name, lg_obj, implemented),
        single=single,
        multi=multi,
    )


def _get_namespaces(attr):
    import numpy

    import cunumeric

    if attr is None:
        return numpy, cunumeric

    return getattr(numpy, attr), getattr(cunumeric, attr)


def _generate_section(config):
    np_obj, lg_obj = _get_namespaces(config.attr)

    if config.names:
        names = config.names
    else:
        names = _filter_names(np_obj, config.types)

    items = [_get_item(name, np_obj, lg_obj) for name in names]

    return SectionDetail(
        title=config.title,
        np_count=len(items),
        lg_count=len([item for item in items if item.implemented]),
        items=sorted(items, key=lambda x: x.name),
    )


class ComparisonTable(CunumericDirective):

    has_content = False
    required_arguments = 0
    optional_arguments = 1

    option_spec = {
        "sections": lambda x: choice(x, ("numpy", "grouped")),
    }

    def run(self):
        if self.options.get("sections", "numpy") == "numpy":
            section_configs = NUMPY_CONFIGS
        else:
            section_configs = GROUPED_CONFIGS

        sections = [_generate_section(config) for config in section_configs]

        rst_text = COMPARISON_TABLE.render(sections=sections)
        log.debug(rst_text)

        return self.parse(rst_text, "<comparison-table>")


def setup(app):
    app.add_directive_to_domain("py", "comparison-table", ComparisonTable)
