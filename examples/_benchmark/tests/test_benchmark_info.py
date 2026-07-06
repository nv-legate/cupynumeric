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

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import cupynumeric as num

from _benchmark.info import (
    _TIME,
    INFO,
    BenchmarkInfo,
    _format_package,
    benchmark_info,
    create_benchmark_info,
    get_benchmark_info,
)


# ---------------------------------------------------------------------------
# BenchmarkInfo dataclass
# ---------------------------------------------------------------------------


def _make_info(**overrides: Any) -> Any:
    defaults: dict[str, Any] = {
        "name": "bench",
        "input_names": {"x": "X"},
        "output_names": _TIME,
        "formats": {},
        "returns_time": 0,
    }
    defaults.update(overrides)
    return BenchmarkInfo(**defaults)


def test_benchmark_info_replace_returns_new_instance() -> None:
    original = _make_info()
    updated = original.replace(name="renamed")
    assert updated is not original
    assert updated.name == "renamed"
    assert original.name == "bench"


def test_benchmark_info_replace_preserves_other_fields() -> None:
    original = _make_info(returns_time=2)
    updated = original.replace(name="other")
    assert updated.input_names == original.input_names
    assert updated.output_names == original.output_names
    assert updated.formats == original.formats
    assert updated.returns_time == 2


# ---------------------------------------------------------------------------
# _format_package
# ---------------------------------------------------------------------------


def test_format_package_numpy() -> None:
    assert _format_package(np) == "numpy"


def test_format_package_other_module() -> None:
    fake = SimpleNamespace(__name__="cupy")
    assert _format_package(fake) == "cupy"  # type: ignore[arg-type]


def test_format_package_cupynumeric_appends_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = MagicMock()
    machine.preferred_target.name = "mockdevice"
    monkeypatch.setattr("legate.core.get_machine", lambda: machine)

    assert _format_package(num) == "cupynumeric.mockdevice"


# ---------------------------------------------------------------------------
# create_benchmark_info
# ---------------------------------------------------------------------------


def test_create_benchmark_info_uses_function_name() -> None:
    def my_bench(x: int) -> float:
        return 0.0

    info = create_benchmark_info(my_bench)
    assert info.name == "my_bench"


def test_create_benchmark_info_explicit_name_overrides() -> None:
    def my_bench() -> float:
        return 0.0

    info = create_benchmark_info(my_bench, name="custom")
    assert info.name == "custom"


def test_create_benchmark_info_requires_name_for_anonymous() -> None:
    class Callable:
        def __call__(self) -> float:
            return 0.0

    obj = Callable()
    # An instance has no __name__ attribute by default.
    with pytest.raises(RuntimeError, match="has no __name__"):
        create_benchmark_info(obj)


def test_create_benchmark_info_default_output_names_is_time() -> None:
    def f() -> float:
        return 0.0

    info = create_benchmark_info(f)
    assert info.output_names == _TIME


def test_create_benchmark_info_requires_output_names_when_returns_time_nonzero() -> (
    None
):
    def f() -> float:
        return 0.0

    with pytest.raises(RuntimeError, match="Unable to determine output_names"):
        create_benchmark_info(f, returns_time=-1)


def test_create_benchmark_info_returns_time_nonzero_with_output_names() -> (
    None
):
    def f() -> float:
        return 0.0

    info = create_benchmark_info(f, output_names="result", returns_time=-1)
    assert info.output_names == "result"
    assert info.returns_time == -1


def test_create_benchmark_info_np_param_added_as_array_package() -> None:
    def f(np: Any) -> float:
        return 0.0

    info = create_benchmark_info(f)
    assert info.input_names == {"np": "array package"}
    assert info.formats == {"array package": _format_package}


def test_create_benchmark_info_no_np_param_has_no_columns() -> None:
    def f(x: int) -> float:
        return 0.0

    info = create_benchmark_info(f)
    assert info.input_names == {}
    assert info.formats == {}


def test_create_benchmark_info_input_names_override_np_label() -> None:
    def f(np: Any) -> float:
        return 0.0

    info = create_benchmark_info(f, input_names={"np": "package"})
    assert info.input_names == {"np": "package"}


def test_create_benchmark_info_input_names_extend_with_np() -> None:
    def f(np: Any, n: int) -> float:
        return 0.0

    info = create_benchmark_info(f, input_names={"n": "size"})
    assert info.input_names == {"np": "array package", "n": "size"}


def test_create_benchmark_info_user_formats_override_defaults() -> None:
    def fmt(_: Any) -> str:
        return "custom"

    def f(np: Any) -> float:
        return 0.0

    info = create_benchmark_info(f, formats={"array package": fmt})
    assert info.formats["array package"] is fmt


def test_create_benchmark_info_rejects_unknown_input_name() -> None:
    def f(x: int) -> float:
        return 0.0

    with pytest.raises(RuntimeError, match="missing is not a parameter"):
        create_benchmark_info(f, input_names={"missing": "label"})


def test_create_benchmark_info_rejects_keyword_only_parameter() -> None:
    def f(*, x: int = 1) -> float:
        return 0.0

    with pytest.raises(RuntimeError, match="x is not positional"):
        create_benchmark_info(f, input_names={"x": "X"})


def test_create_benchmark_info_accepts_positional_only_parameter() -> None:
    def f(x: int, /) -> float:
        return 0.0

    info = create_benchmark_info(f, input_names={"x": "X"})
    assert info.input_names == {"x": "X"}


def test_create_benchmark_info_tuple_output_names() -> None:
    def f() -> tuple[float, int]:
        return 0.0, 0

    info = create_benchmark_info(
        f, output_names=("time", "iters"), returns_time=0
    )
    assert info.output_names == ("time", "iters")


# ---------------------------------------------------------------------------
# benchmark_info decorator
# ---------------------------------------------------------------------------


def test_benchmark_info_decorator_attaches_info_attribute() -> None:
    @benchmark_info()
    def f() -> float:
        return 0.0

    assert hasattr(f, INFO)
    attached = getattr(f, INFO)
    assert isinstance(attached, BenchmarkInfo)
    assert attached.name == "f"


def test_benchmark_info_decorator_returns_same_function() -> None:
    def original() -> float:
        return 0.0

    decorated = benchmark_info()(original)
    assert decorated is original


def test_benchmark_info_decorator_returns_time_true_means_zero() -> None:
    @benchmark_info(returns_time=True)
    def f() -> float:
        return 0.0

    assert getattr(f, INFO).returns_time == 0


def test_benchmark_info_decorator_returns_time_false_means_negative_one() -> (
    None
):
    @benchmark_info(returns_time=False, output_names="out")
    def f() -> float:
        return 0.0

    assert getattr(f, INFO).returns_time == -1


def test_benchmark_info_decorator_returns_time_integer_passes_through() -> (
    None
):
    @benchmark_info(returns_time=2, output_names=("a", "b", "t"))
    def f() -> tuple[int, int, float]:
        return 1, 2, 3.0

    assert getattr(f, INFO).returns_time == 2


def test_benchmark_info_decorator_propagates_name_and_outputs() -> None:
    @benchmark_info(name="explicit", output_names="value", returns_time=False)
    def f() -> float:
        return 0.0

    attached = getattr(f, INFO)
    assert attached.name == "explicit"
    assert attached.output_names == "value"


def test_benchmark_info_decorator_propagates_input_names_and_formats() -> None:
    def fmt(_: Any) -> str:
        return "x"

    @benchmark_info(input_names={"n": "size"}, formats={"size": fmt})
    def f(n: int) -> float:
        return 0.0

    attached = getattr(f, INFO)
    assert attached.input_names == {"n": "size"}
    assert attached.formats == {"size": fmt}


# ---------------------------------------------------------------------------
# get_benchmark_info
# ---------------------------------------------------------------------------


def test_get_benchmark_info_returns_attached_info() -> None:
    @benchmark_info(name="x")
    def f() -> float:
        return 0.0

    assert get_benchmark_info(f) is getattr(f, INFO)


def test_get_benchmark_info_generates_default_when_missing() -> None:
    def f(x: int) -> float:
        return 0.0

    info = get_benchmark_info(f)
    assert isinstance(info, BenchmarkInfo)
    assert info.name == "f"
    assert info.output_names == _TIME


def test_get_benchmark_info_default_does_not_attach_attribute() -> None:
    def f() -> float:
        return 0.0

    get_benchmark_info(f)
    assert not hasattr(f, INFO)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
