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

import csv
import sys
from dataclasses import astuple, dataclass

from cupynumeric._sphinxext._comparison_config import GROUPED_CONFIGS
from cupynumeric._sphinxext._comparison_util import filter_names
from cupynumeric.coverage import is_implemented


@dataclass
class Row:
    group: str
    np_count: int = 0
    lg_count: int = 0
    cp_count: int = 0


def get_namespaces(attr):
    import cupy
    import numpy

    import cupynumeric

    if attr is None:
        return numpy, cupynumeric, cupy

    return (
        getattr(numpy, attr),
        getattr(cupynumeric, attr),
        getattr(cupy, attr),
    )


def write_rows(rows):
    headers = ("group", "numpy", "cupynumeric", "cupy")
    writer = csv.writer(sys.stdout)
    writer.writerow(headers)
    for row in rows:
        writer.writerow(astuple(row))


def generate_row(config):
    np_obj, lg_obj, cp_obj = get_namespaces(config.attr)

    if config.names:
        names = config.names
    else:
        names = filter_names(np_obj, config.types)

    row = Row(group=config.title)

    for name in names:
        row.np_count += 1

        if is_lg(name, lg_obj):
            row.lg_count += 1

        if is_cp(name, cp_obj):
            row.cp_count += 1

    return row


def is_lg(name, obj):
    return is_implemented(getattr(obj, name))


def is_cp(name, obj):
    return getattr(obj, name, None) is not None


def main():
    rows = [generate_row(config) for config in GROUPED_CONFIGS]

    write_rows(rows)


if __name__ == "__main__":
    main()
