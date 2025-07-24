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

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from .._utils.coverage import (
    GPUSupport,
    is_implemented,
    is_multi,
    is_single,
    is_wrapped,
)
from ._comparison_config import MISSING_NP_REFS, SKIP

if TYPE_CHECKING:
    from ._comparison_config import SectionConfig


def _support_symbol(support: GPUSupport) -> str:
    YES = "\u2713"
    NO = "\u274c"
    PARTIAL = "\U0001f7e1"

    match support:
        case GPUSupport.YES:
            return YES
        case GPUSupport.NO:
            return NO
        case GPUSupport.PARTIAL:
            return PARTIAL


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


def _npref(name: str, obj: Any) -> str:
    if isinstance(obj, ModuleType):
        full_name = f"{obj.__name__}.{name}"
    else:
        full_name = f"numpy.{obj.__name__}.{name}"

    role = "meth" if "ndarray" in full_name else "obj"

    if full_name in MISSING_NP_REFS:
        return f"``{full_name}``"
    return f":{role}:`{full_name}`"


def _lgref(name: str, obj: Any, implemented: bool) -> str:
    if not implemented:
        return "-"

    if isinstance(obj, ModuleType):
        full_name = f"{obj.__name__}.{name}"
    else:
        full_name = f"cupynumeric.{obj.__name__}.{name}"

    role = "meth" if "ndarray" in full_name else "obj"

    return f":{role}:`{full_name}`"


def filter_wrapped_names(
    obj: Any, *, skip: Iterable[str] = ()
) -> Iterator[str]:
    names = (n for n in dir(obj))  # every name in the module or class
    names = (
        n for n in names if is_wrapped(getattr(obj, n))
    )  # that is wrapped
    names = (n for n in names if n not in skip)  # except the ones we skip
    names = (n for n in names if not n.startswith("_"))  # or any private names
    return names


def filter_type_names(obj: Any, *, skip: Iterable[str] = ()) -> Iterator[str]:
    names = (n for n in dir(obj))  # every name in the module or class
    names = (
        n for n in names if isinstance(getattr(obj, n), type)
    )  # that is a type (class, dtype, etc)
    names = (n for n in names if n not in skip)  # except the ones we skip
    names = (n for n in names if not n.startswith("_"))  # or any private names
    return names


def get_item(name: str, np_obj: Any, lg_obj: Any) -> ItemDetail:
    lg_attr = getattr(lg_obj, name)

    if implemented := is_implemented(lg_attr):
        single = _support_symbol(is_single(lg_attr))
        multi = _support_symbol(is_multi(lg_attr))
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


def get_namespaces(attr: str | None) -> tuple[Any, Any]:
    import numpy

    import cupynumeric

    if attr is None:
        return numpy, cupynumeric

    return getattr(numpy, attr), getattr(cupynumeric, attr)


def generate_section(config: SectionConfig) -> SectionDetail:
    np_obj, lg_obj = get_namespaces(config.attr)

    names: Iterable[str]

    if config.names:
        names = set(config.names)
    else:
        wrapped_names = filter_wrapped_names(lg_obj, skip=SKIP)
        type_names = filter_type_names(lg_obj, skip=SKIP)
        names = set(wrapped_names) | set(type_names)

    # we can omit anything that isn't in np namespace to begin with
    names = {n for n in names if n in dir(np_obj)}

    items = [get_item(name, np_obj, lg_obj) for name in names]

    return SectionDetail(
        title=config.title,
        np_count=len(items),
        lg_count=len([item for item in items if item.implemented]),
        items=sorted(items, key=lambda x: x.name),
    )
