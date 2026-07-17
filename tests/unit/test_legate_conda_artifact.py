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
import sys

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_helper() -> ModuleType:
    path = (
        _REPO_ROOT
        / "continuous_integration"
        / "scripts"
        / "tools"
        / "legate_conda_artifact.py"
    )
    spec = importlib.util.spec_from_file_location(
        "legate_conda_artifact", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = _load_helper()


@pytest.fixture
def versions() -> dict[str, Any]:
    return {
        "packages": {
            "legate": {
                "artifact_name": (
                    "<<platform>>-<<build_type>>-<<repo>>-"
                    "python<<python_version>>-<<target_device>>-release-"
                    "with_tests-<<network>>-cuda<<cuda_major>>-<<git_tag>>"
                ),
                "artifact_workflow": "ci-gh.yml",
                "cpu_artifact_cuda_major": "14",
                "git_tag": "1" * 40,
                "nightly_workflow": "ci-gh-nightly-release.yml",
                "org": "nv-legate",
                "repo": "legate.internal",
            }
        }
    }


@pytest.mark.parametrize(
    ("build_type", "target_device", "cuda_version", "workflow", "cuda_major"),
    (
        ("ci", "gpu", "12.9.0", "ci-gh.yml", "12"),
        ("nightly", "cpu", "12.9.0", "ci-gh-nightly-release.yml", "14"),
    ),
)
def test_resolve_metadata(
    versions: dict[str, Any],
    build_type: str,
    target_device: str,
    cuda_version: str,
    workflow: str,
    cuda_major: str,
) -> None:
    metadata = HELPER.resolve_metadata(
        versions,
        build_type=build_type,
        platform="linux",
        python_version="3.14",
        target_device=target_device,
        cuda_version=cuda_version,
        network="ucx",
    )

    assert metadata == {
        "name": (
            f"linux-{build_type}-legate.internal-python3.14-{target_device}-"
            f"release-with_tests-ucx-cuda{cuda_major}-{'1' * 40}"
        ),
        "repository": "nv-legate/legate.internal",
        "sha": "1" * 40,
        "workflow": workflow,
    }


def test_resolve_metadata_rejects_unknown_build_type(
    versions: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="unsupported build type: debug"):
        HELPER.resolve_metadata(
            versions,
            build_type="debug",
            platform="linux",
            python_version="3.14",
            target_device="gpu",
            cuda_version="13.0.0",
            network="ucx",
        )


def test_resolve_metadata_rejects_unknown_template_token(
    versions: dict[str, Any],
) -> None:
    versions["packages"]["legate"]["artifact_name"] = "<<unknown>>"

    with pytest.raises(ValueError, match="unresolved artifact-name template"):
        HELPER.resolve_metadata(
            versions,
            build_type="ci",
            platform="linux",
            python_version="3.14",
            target_device="gpu",
            cuda_version="13.0.0",
            network="ucx",
        )


def test_current_manifest_resolves_exact_ci_artifact() -> None:
    with (_REPO_ROOT / "cmake" / "versions.json").open() as versions_file:
        versions = json.load(versions_file)

    metadata = HELPER.resolve_metadata(
        versions,
        build_type="ci",
        platform="linux",
        python_version="3.14",
        target_device="gpu",
        cuda_version="13.0.0",
        network="ucx",
    )

    assert metadata["name"] == (
        "linux-ci-legate.internal-python3.14-gpu-release-with_tests-ucx-"
        f"cuda13-{versions['packages']['legate']['git_tag']}"
    )


def test_current_manifest_resolves_exact_cpu_artifact() -> None:
    with (_REPO_ROOT / "cmake" / "versions.json").open() as versions_file:
        versions = json.load(versions_file)

    metadata = HELPER.resolve_metadata(
        versions,
        build_type="nightly",
        platform="linux",
        python_version="3.14",
        target_device="cpu",
        cuda_version="12.9.0",
        network="ucx",
    )

    assert metadata["name"] == (
        "linux-nightly-legate.internal-python3.14-cpu-"
        "release-with_tests-ucx-"
        f"cuda13-{versions['packages']['legate']['git_tag']}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
