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

import traceback
from types import FrameType


def find_last_user_stacklevel() -> int:
    stacklevel = 1
    for frame, _ in traceback.walk_stack(None):
        if not frame.f_globals["__name__"].startswith("cupynumeric"):
            break
        stacklevel += 1
    return stacklevel


def get_line_number_from_frame(frame: FrameType) -> str:
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def find_last_user_frames(top_only: bool = True) -> str:
    for last, _ in traceback.walk_stack(None):
        if "__name__" not in last.f_globals:
            continue
        name = last.f_globals["__name__"]
        if not any(name.startswith(pkg) for pkg in ("cupynumeric", "legate")):
            break

    if top_only:
        return get_line_number_from_frame(last)

    frames: list[FrameType] = []
    curr: FrameType | None = last
    while curr is not None:
        if "legion_top.py" in curr.f_code.co_filename:
            break
        frames.append(curr)
        curr = curr.f_back
    return "|".join(get_line_number_from_frame(f) for f in frames)
