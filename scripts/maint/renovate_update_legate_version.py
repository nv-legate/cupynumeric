#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib import error, request

_GIT_URL_RE = re.compile(
    r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$"
)
_VERSION_RE = re.compile(r"^\d+(?:\.\d+){1,3}(?:\.dev)?$")
_VERSION_PATH = "VERSION"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _github_token() -> str | None:
    return (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("RENOVATE_TOKEN")
        or os.environ.get("GH_TOKEN")
    )


def _repo_from_git_url(git_url: str) -> str:
    match = _GIT_URL_RE.search(git_url)
    if not match:
        msg = f"Unsupported git_url format: {git_url!r}"
        raise ValueError(msg)
    return f"{match.group('owner')}/{match.group('repo')}"


def _fetch_version(repo: str, git_sha: str) -> str:
    url = (
        f"https://api.github.com/repos/{repo}/contents/{_VERSION_PATH}"
        f"?ref={git_sha}"
    )
    headers = {"Accept": "application/vnd.github.raw"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"token {token}"
    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        msg = f"Failed to fetch {repo}@{git_sha}:{_VERSION_PATH} ({exc})"
        raise ValueError(msg) from exc
    except error.URLError as exc:
        msg = f"Failed to reach GitHub API for {repo}: {exc}"
        raise ValueError(msg) from exc
    version = raw.strip()
    if not version:
        raise ValueError(f"Empty VERSION file in {repo}@{git_sha}")
    if not _VERSION_RE.match(version):
        raise ValueError(f"Unexpected VERSION format: {version!r}")
    if version.endswith(".dev"):
        version = version[:-4]
    return version


def _pyproject_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Expected at least major.minor in version, got {version}"
        )
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric version {version}") from exc
    return f"{major}.{minor}"


def _update_versions_json(
    path: Path, *, git_sha: str, legate_version: str
) -> bool:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        data = json.load(handle)

    packages = data.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("Missing 'packages' section in versions.json")
    legate_cfg = packages.get("legate")
    if not isinstance(legate_cfg, dict):
        raise ValueError("Missing 'legate' entry in versions.json")

    updated = False
    if legate_cfg.get("git_tag") != git_sha:
        legate_cfg["git_tag"] = git_sha
        updated = True
    if legate_cfg.get("version") != legate_version:
        legate_cfg["version"] = legate_version
        updated = True

    if not updated:
        return False
    with path.open(mode="w") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")
    return True


def _update_pyproject(path: Path, *, legate_version: str) -> bool:
    if not path.is_file():
        raise FileNotFoundError(path)
    contents = path.read_text(encoding="utf-8")
    new_version = _pyproject_version(legate_version)
    pattern = re.compile(r"legate==(?P<version>\d+\.\d+)\.\*,>=0\.0\.0a0")
    if not pattern.search(contents):
        raise ValueError(
            "Failed to locate legate dependency pin in pyproject.toml"
        )
    replacement = f"legate=={new_version}.*,>=0.0.0a0"
    new_contents = pattern.sub(replacement, contents, count=1)
    if new_contents == contents:
        return False
    path.write_text(new_contents, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update Legate version metadata (Renovate helper)."
    )
    parser.add_argument(
        "--new-version",
        required=True,
        help="New Legate git SHA from Renovate.",
    )
    args = parser.parse_args()

    root = _repo_root()
    versions_json = root / "cmake" / "versions.json"

    try:
        with versions_json.open() as handle:
            data = json.load(handle)
        packages = data.get("packages", {})
        legate_cfg = packages.get("legate", {})
        if not isinstance(legate_cfg, dict):
            raise ValueError("Missing 'legate' entry in versions.json")
        git_url = legate_cfg.get("git_url")
        if not isinstance(git_url, str):
            raise ValueError("Missing git_url for Legate in versions.json")
        repo = _repo_from_git_url(git_url)
        legate_version = _fetch_version(repo, args.new_version)

        _update_versions_json(
            versions_json,
            git_sha=args.new_version,
            legate_version=legate_version,
        )

        pyproject = (
            root
            / "scripts"
            / "build"
            / "python"
            / "cupynumeric"
            / "pyproject.toml"
        )
        _update_pyproject(pyproject, legate_version=legate_version)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
