#!/bin/bash

# Copyright 2026 NVIDIA Corporation
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

set -e

legate_root=$(python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())') || {
    echo "legate not installed" >&2
    exit 1
}
if [[ -z "${legate_root}" ]]; then
    echo "legate not installed" >&2
    exit 1
fi
echo "Using Legate at ${legate_root}"

cupynumeric_root=$(python -c 'import cupynumeric.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())') || {
    echo "cupynumeric not installed" >&2
    exit 1
}
if [[ -z "${cupynumeric_root}" ]]; then
    echo "cupynumeric not installed" >&2
    exit 1
fi
echo "Using cuPyNumeric at ${cupynumeric_root}"
cmake -S . -B build -D legate_ROOT="$legate_root" -D cupynumeric_ROOT="$cupynumeric_root" -D CMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel 8
