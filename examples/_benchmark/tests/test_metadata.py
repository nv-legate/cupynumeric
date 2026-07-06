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

import json
import sys
from subprocess import CalledProcessError
from types import SimpleNamespace
from typing import Any

import pytest
from legate.util.settings import SettingBase

from _benchmark.harness import (
    FAILED_TO_DETECT,
    _conda_list,
    _cupy_package_details,
    _cupynumeric_settings_info,
    _try_conda,
    _try_version,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    # Each of these helpers is wrapped in functools.cache; clear before every
    # test so monkeypatched dependencies are actually exercised.
    _cupynumeric_settings_info.cache_clear()
    _conda_list.cache_clear()
    _cupy_package_details.cache_clear()


# ---------------------------------------------------------------------------
# _cupynumeric_settings_info
# ---------------------------------------------------------------------------


class _FakeSetting(SettingBase[Any]):
    def __init__(self, name: str, value: Any) -> None:
        super().__init__(name=name)
        self._value = value

    def __call__(self) -> Any:
        return self._value


def _install_fake_settings_module(
    monkeypatch: pytest.MonkeyPatch, settings: Any
) -> None:
    fake_module = SimpleNamespace(settings=settings)
    monkeypatch.setitem(sys.modules, "cupynumeric.settings", fake_module)


def test_cupynumeric_settings_info_collects_setting_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        foo=_FakeSetting("foo", 1), bar=_FakeSetting("bar", "hello")
    )
    _install_fake_settings_module(monkeypatch, settings)
    assert _cupynumeric_settings_info() == {"foo": 1, "bar": "hello"}


def test_cupynumeric_settings_info_skips_non_setting_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        real=_FakeSetting("real", "ok"), also_a_number=42, also_a_str="ignore"
    )
    _install_fake_settings_module(monkeypatch, settings)
    assert _cupynumeric_settings_info() == {"real": "ok"}


def test_cupynumeric_settings_info_uses_setting_name_as_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(attribute_name=_FakeSetting("display_name", 7))
    _install_fake_settings_module(monkeypatch, settings)
    # Result is keyed by the setting's .name, not by the host attribute.
    assert _cupynumeric_settings_info() == {"display_name": 7}


def test_cupynumeric_settings_info_returns_empty_when_no_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_settings_module(monkeypatch, SimpleNamespace())
    assert _cupynumeric_settings_info() == {}


def test_cupynumeric_settings_info_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    class _Counting(SettingBase[Any]):
        def __init__(self) -> None:
            super().__init__(name="x")

        def __call__(self) -> Any:
            nonlocal call_count
            call_count += 1
            return call_count

    _install_fake_settings_module(monkeypatch, SimpleNamespace(x=_Counting()))
    first = _cupynumeric_settings_info()
    second = _cupynumeric_settings_info()
    assert first == {"x": 1}
    assert second is first
    assert call_count == 1


# ---------------------------------------------------------------------------
# _conda_list
# ---------------------------------------------------------------------------


def _encode_conda_list(packages: list[dict[str, str]]) -> bytes:
    return json.dumps(packages).encode("utf-8")


def test_conda_list_success_returns_dict_keyed_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _encode_conda_list(
        [
            {
                "name": "numpy",
                "version": "1.26.0",
                "channel": "conda-forge",
                "build_string": "py311h0",
            },
            {
                "name": "scipy",
                "version": "1.11.4",
                "channel": "pypi",
                "build_string": "py3",
            },
        ]
    )
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: output
    )
    result = _conda_list()
    assert isinstance(result, dict)
    assert set(result.keys()) == {"numpy", "scipy"}


def test_conda_list_entry_format_aligns_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Max version width = 6 ("12.0.0"); max build width = 7 ("py3_h_0").
    output = _encode_conda_list(
        [
            {
                "name": "a",
                "version": "1.0",
                "channel": "c1",
                "build_string": "b1",
            },
            {
                "name": "b",
                "version": "12.0.0",
                "channel": "c2",
                "build_string": "py3_h_0",
            },
        ]
    )
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: output
    )
    result = _conda_list()
    assert isinstance(result, dict)
    assert result["a"] == f"{'1.0':6}  {'b1':7}  c1"
    assert result["b"] == f"{'12.0.0':6}  {'py3_h_0':7}  c2"


def test_conda_list_invokes_expected_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    def fake(*args: Any, **_kwargs: Any) -> bytes:
        captured.append(args)
        return _encode_conda_list(
            [
                {
                    "name": "x",
                    "version": "1",
                    "channel": "c",
                    "build_string": "b",
                }
            ]
        )

    monkeypatch.setattr("_benchmark.harness.check_output", fake)
    _conda_list()
    assert captured == [(["conda", "list", "--json"],)]


def test_conda_list_called_process_error_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(*_a: Any, **_k: Any) -> bytes:
        raise CalledProcessError(returncode=1, cmd=["conda"])

    monkeypatch.setattr("_benchmark.harness.check_output", raise_err)
    assert _conda_list() == FAILED_TO_DETECT


def test_conda_list_missing_key_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Missing "channel" and "build_string" -> KeyError caught.
    output = _encode_conda_list([{"name": "x", "version": "1"}])
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: output
    )
    assert _conda_list() == FAILED_TO_DETECT


def test_conda_list_file_not_found_returns_conda_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(*_a: Any, **_k: Any) -> bytes:
        raise FileNotFoundError()

    monkeypatch.setattr("_benchmark.harness.check_output", raise_err)
    assert _conda_list() == "(conda missing)"


def test_conda_list_empty_output_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Falsy output skips the body and the else-clause returns FAILED_TO_DETECT.
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: b""
    )
    assert _conda_list() == FAILED_TO_DETECT


def test_conda_list_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0

    def fake(*_a: Any, **_k: Any) -> bytes:
        nonlocal call_count
        call_count += 1
        return _encode_conda_list(
            [
                {
                    "name": "x",
                    "version": "1",
                    "channel": "c",
                    "build_string": "b",
                }
            ]
        )

    monkeypatch.setattr("_benchmark.harness.check_output", fake)
    _conda_list()
    _conda_list()
    assert call_count == 1


# ---------------------------------------------------------------------------
# _try_version
# ---------------------------------------------------------------------------


def test_try_version_returns_attribute_as_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mod = SimpleNamespace(__version__="1.2.3")
    monkeypatch.setattr(
        "_benchmark.harness.importlib.import_module", lambda _name: fake_mod
    )
    assert _try_version("anything", "__version__") == "1.2.3"


def test_try_version_stringifies_non_string_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mod = SimpleNamespace(VERSION=(1, 2, 3))
    monkeypatch.setattr(
        "_benchmark.harness.importlib.import_module", lambda _name: fake_mod
    )
    assert _try_version("x", "VERSION") == "(1, 2, 3)"


def test_try_version_falsy_module_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "_benchmark.harness.importlib.import_module", lambda _name: None
    )
    assert _try_version("x", "__version__") == FAILED_TO_DETECT


def test_try_version_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_err(_name: str) -> Any:
        raise ModuleNotFoundError("absent")

    monkeypatch.setattr(
        "_benchmark.harness.importlib.import_module", raise_err
    )
    assert _try_version("absent", "__version__") == FAILED_TO_DETECT


def test_try_version_import_error_strips_parenthesized_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(_name: str) -> Any:
        raise ImportError("boom (/path/to/file.so)")

    monkeypatch.setattr(
        "_benchmark.harness.importlib.import_module", raise_err
    )
    assert _try_version("x", "__version__") == "(ImportError: boom)"


def test_try_version_import_error_without_paren_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(_name: str) -> Any:
        raise ImportError("clean message")

    monkeypatch.setattr(
        "_benchmark.harness.importlib.import_module", raise_err
    )
    assert _try_version("x", "v") == "(ImportError: clean message)"


def test_try_version_generic_exception_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(_name: str) -> Any:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "_benchmark.harness.importlib.import_module", raise_err
    )
    assert _try_version("x", "v") == "(Exception on import: kaboom)"


def test_try_version_missing_attribute_falls_into_generic_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # AttributeError from getattr() is not ModuleNotFoundError/ImportError,
    # so it lands in the generic Exception handler.
    fake_mod = SimpleNamespace()
    monkeypatch.setattr(
        "_benchmark.harness.importlib.import_module", lambda _name: fake_mod
    )
    result = _try_version("x", "__version__")
    assert result.startswith("(Exception on import:")


# ---------------------------------------------------------------------------
# _try_conda
# ---------------------------------------------------------------------------


def test_try_conda_returns_dist_and_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = json.dumps(
        [{"dist_name": "cupy-13.0.0", "channel": "conda-forge"}]
    ).encode("utf-8")
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: output
    )
    assert _try_conda("cupy") == "cupy-13.0.0 (conda-forge)"


def test_try_conda_uses_first_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    output = json.dumps(
        [
            {"dist_name": "first", "channel": "c1"},
            {"dist_name": "second", "channel": "c2"},
        ]
    ).encode("utf-8")
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: output
    )
    assert _try_conda("p") == "first (c1)"


def test_try_conda_invokes_command_with_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    def fake(*args: Any, **_kwargs: Any) -> bytes:
        captured.append(args)
        return json.dumps([{"dist_name": "d", "channel": "c"}]).encode("utf-8")

    monkeypatch.setattr("_benchmark.harness.check_output", fake)
    _try_conda("somepkg")
    assert captured == [(["conda", "list", "somepkg", "--json"],)]


def test_try_conda_called_process_error_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(*_a: Any, **_k: Any) -> bytes:
        raise CalledProcessError(returncode=1, cmd=["conda"])

    monkeypatch.setattr("_benchmark.harness.check_output", raise_err)
    assert _try_conda("anything") == FAILED_TO_DETECT


def test_try_conda_empty_array_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # IndexError from [0] on an empty list is caught.
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: b"[]"
    )
    assert _try_conda("nope") == FAILED_TO_DETECT


def test_try_conda_missing_key_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = json.dumps([{"dist_name": "x"}]).encode("utf-8")
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: output
    )
    assert _try_conda("x") == FAILED_TO_DETECT


def test_try_conda_file_not_found_returns_conda_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(*_a: Any, **_k: Any) -> bytes:
        raise FileNotFoundError()

    monkeypatch.setattr("_benchmark.harness.check_output", raise_err)
    assert _try_conda("anything") == "(conda missing)"


def test_try_conda_empty_output_returns_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "_benchmark.harness.check_output", lambda *_a, **_k: b""
    )
    assert _try_conda("anything") == FAILED_TO_DETECT


# ---------------------------------------------------------------------------
# _cupy_package_details
# ---------------------------------------------------------------------------


def test_cupy_package_details_returns_try_conda_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Any] = []

    def fake(*args: Any, **_kwargs: Any) -> bytes:
        captured.append(args)
        return json.dumps(
            [{"dist_name": "cupy-13.0.0", "channel": "conda-forge"}]
        ).encode("utf-8")

    monkeypatch.setattr("_benchmark.harness.check_output", fake)
    assert _cupy_package_details() == "cupy-13.0.0 (conda-forge)"
    assert captured == [(["conda", "list", "cupy", "--json"],)]


def test_cupy_package_details_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(*_a: Any, **_k: Any) -> bytes:
        raise FileNotFoundError()

    monkeypatch.setattr("_benchmark.harness.check_output", raise_err)
    assert _cupy_package_details() == "(conda missing)"


def test_cupy_package_details_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake(*_a: Any, **_k: Any) -> bytes:
        nonlocal call_count
        call_count += 1
        return json.dumps(
            [{"dist_name": "cupy-13.0.0", "channel": "conda-forge"}]
        ).encode("utf-8")

    monkeypatch.setattr("_benchmark.harness.check_output", fake)
    _cupy_package_details()
    _cupy_package_details()
    assert call_count == 1


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
