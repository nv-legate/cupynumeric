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

import importlib.util
import json
import re
import shutil
import sys

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_WHEEL_DIR = Path("scripts") / "build" / "python"


def _load_helper(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUMP_VERSIONS = _load_helper(
    "bump_versions",
    _REPO_ROOT / "scripts" / "release" / "util" / "bump_versions.py",
)
RENOVATE_HELPER = _load_helper(
    "renovate_helper",
    _REPO_ROOT / "scripts" / "maint" / "renovate_update_legate_version.py",
)


def _wheel_pyproject(root: Path, wheel_package: str) -> Path:
    return root / _WHEEL_DIR / wheel_package / "pyproject.toml"


def _copy_real_wheel_pyprojects(root: Path, wheel_package_deps) -> None:
    for wheel_package, _ in wheel_package_deps:
        path = _wheel_pyproject(root, wheel_package)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_wheel_pyproject(_REPO_ROOT, wheel_package), path)


def _assert_wheel_pins(root: Path, wheel_package_deps) -> None:
    for wheel_package, dependency_package in wheel_package_deps:
        pyproject = _wheel_pyproject(root, wheel_package)
        assert (
            f'"{dependency_package}==26.7.*,>=0.0.0a0"'
            in pyproject.read_text()
        )


def test_release_helper_updates_real_wheel_pyprojects(tmp_path: Path) -> None:
    _copy_real_wheel_pyprojects(tmp_path, BUMP_VERSIONS._WHEEL_PACKAGE_DEPS)
    ctx = SimpleNamespace(
        cupynumeric_dir=tmp_path,
        dry_run=False,
        version_after_this="26.07",
        vprint=lambda *args, **kwargs: None,
    )

    BUMP_VERSIONS.bump_pyproject_dependency(ctx)

    _assert_wheel_pins(tmp_path, BUMP_VERSIONS._WHEEL_PACKAGE_DEPS)


def test_renovate_helper_updates_version_metadata_and_wheel_pyprojects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_real_wheel_pyprojects(tmp_path, RENOVATE_HELPER._WHEEL_PACKAGE_DEPS)
    versions_json = tmp_path / "cmake" / "versions.json"
    versions_json.parent.mkdir(parents=True)
    versions_json.write_text(
        json.dumps(
            {
                "packages": {
                    "legate": {
                        "git_tag": "0" * 40,
                        "git_url": (
                            "git@github.com:nv-legate/legate.internal.git"
                        ),
                        "version": "26.05.00",
                    }
                }
            }
        )
    )

    monkeypatch.setattr(RENOVATE_HELPER, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        RENOVATE_HELPER, "_fetch_version", lambda repo, git_sha: "26.07.00"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["renovate_update_legate_version.py", "--new-version", "1" * 40],
    )

    assert RENOVATE_HELPER.main() == 0

    versions = json.loads(versions_json.read_text())
    assert versions["packages"]["legate"]["git_tag"] == "1" * 40
    assert versions["packages"]["legate"]["version"] == "26.07.00"
    _assert_wheel_pins(tmp_path, RENOVATE_HELPER._WHEEL_PACKAGE_DEPS)


def test_renovate_file_filters_cover_helper_write_set() -> None:
    config = (_REPO_ROOT / "scripts" / "maint" / "renovate.json5").read_text()
    match = re.search(
        r'description: "Update Legate version metadata after git ref updates",'
        r".*?fileFilters: \[(?P<filters>.*?)\]",
        config,
        flags=re.DOTALL,
    )
    assert match is not None
    allowed_paths = set(re.findall(r'"([^"]+)"', match.group("filters")))

    write_set = {"cmake/versions.json"}
    write_set.update(
        str(_WHEEL_DIR / wheel_package / "pyproject.toml")
        for wheel_package, _ in RENOVATE_HELPER._WHEEL_PACKAGE_DEPS
    )
    assert write_set <= allowed_paths


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
