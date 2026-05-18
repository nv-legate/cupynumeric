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
#
from __future__ import annotations

from functools import cache, wraps
from typing import Callable, ParamSpec, TypeVar

from legate.core import get_legate_runtime, track_provenance, ProfileRange


__all__ = ("profiling_wrapper", "ProfileRange")

ProfileName = str | Callable[..., str]
P = ParamSpec("P")
R = TypeVar("R")


@cache
def _profiling_enabled() -> bool:
    """Check if Legate profiling is enabled via --profile flag"""
    return get_legate_runtime().config().profile  # type: ignore[no-any-return]


def profiling_wrapper(
    func: Callable[P, R], name: ProfileName
) -> Callable[P, R]:
    """
    Wrap a function with profiling and provenance tracking.

    Parameters
    ----------
    func : callable
        The function to wrap
    name : str or callable
        Fully qualified name (e.g., "cupynumeric.matmul"), or a callable that
        receives the wrapped function's arguments and returns the name.

    Returns
    -------
    wrapped : callable
        Function with profiling instrumentation
    """

    if not _profiling_enabled():

        @wraps(func)
        @track_provenance()
        def _provenance_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return _provenance_wrapper

    @wraps(func)
    @track_provenance()
    def _profiling_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        profile_name = name(*args, **kwargs) if callable(name) else name
        # Execute within profiling range
        with ProfileRange(profile_name):
            return func(*args, **kwargs)

    return _profiling_wrapper
