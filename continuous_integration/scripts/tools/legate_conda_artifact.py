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

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any


def resolve_metadata(
    versions: dict[str, Any],
    *,
    build_type: str,
    platform: str,
    python_version: str,
    target_device: str,
    cuda_version: str,
    network: str,
) -> dict[str, str]:
    legate = versions["packages"]["legate"]
    workflow_keys = {"ci": "artifact_workflow", "nightly": "nightly_workflow"}
    try:
        workflow_key = workflow_keys[build_type]
    except KeyError as ex:
        raise ValueError(f"unsupported build type: {build_type}") from ex

    cuda_major = (
        str(legate["cpu_artifact_cuda_major"])
        if target_device == "cpu"
        else cuda_version.split(".")[0]
    )
    replacements = {
        "build_type": build_type,
        "cuda_major": cuda_major,
        "git_tag": str(legate["git_tag"]),
        "network": network,
        "platform": platform,
        "python_version": python_version,
        "repo": str(legate["repo"]),
        "target_device": target_device,
    }

    artifact_name = str(legate["artifact_name"])
    for key, value in replacements.items():
        artifact_name = artifact_name.replace(f"<<{key}>>", value)

    if "<<" in artifact_name or ">>" in artifact_name:
        raise ValueError(f"unresolved artifact-name template: {artifact_name}")

    return {
        "name": artifact_name,
        "repository": f"{legate['org']}/{legate['repo']}",
        "sha": str(legate["git_tag"]),
        "workflow": str(legate[workflow_key]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("versions", type=Path)
    parser.add_argument(
        "--build-type", choices=("ci", "nightly"), required=True
    )
    parser.add_argument("--platform", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument(
        "--target-device", choices=("cpu", "gpu"), required=True
    )
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--network", default="ucx")
    args = parser.parse_args()

    with args.versions.open() as versions_file:
        versions = json.load(versions_file)

    print(
        json.dumps(
            resolve_metadata(
                versions,
                build_type=args.build_type,
                platform=args.platform,
                python_version=args.python_version,
                target_device=args.target_device,
                cuda_version=args.cuda_version,
                network=args.network,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
