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

from types import FrameType


def find_last_user_stacklevel_and_frame() -> tuple[int, FrameType | None]:
    # We want the caller of the public API from this file to see itself as
    # stacklevel 1. So discount the first two hops up the stack, for this
    # helper function, and the public API frontend function.
    stacklevel = -1
    frame: FrameType | None = None
    try:
        raise Exception()
    except Exception as exc:
        if exc.__traceback__ is not None:
            frame = exc.__traceback__.tb_frame
        while frame is not None:
            if (name := frame.f_globals.get("__name__")) is not None:
                if not any(
                    name.startswith(pkg + ".")
                    for pkg in ("cupynumeric", "legate")
                ):
                    return stacklevel, frame
            stacklevel += 1
            frame = frame.f_back
    return stacklevel, None


def find_last_user_stacklevel() -> int:
    stacklevel, _ = find_last_user_stacklevel_and_frame()
    return stacklevel


def get_line_number_from_frame(frame: FrameType) -> str:
    return f"{frame.f_code.co_filename}:{frame.f_lineno}"


def find_last_user_frame() -> FrameType | None:
    _, last = find_last_user_stacklevel_and_frame()
    return last


def find_last_user_frames() -> list[FrameType]:
    _, last = find_last_user_stacklevel_and_frame()

    frames: list[FrameType] = []
    curr: FrameType | None = last
    while curr is not None:
        frames.append(curr)
        curr = curr.f_back

    return frames


def find_last_user_line_numbers(top_only: bool = True) -> str:
    if top_only:
        frame = find_last_user_frame()
        return get_line_number_from_frame(frame) if frame else "unkown"

    frames = find_last_user_frames()
    return "|".join(get_line_number_from_frame(f) for f in frames)
