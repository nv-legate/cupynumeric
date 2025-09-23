# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


def bump_version_file(ctx: Context) -> None:
    version_file = ctx.cupynumeric_dir / "VERSION"
    ctx.vprint(f"Opening {version_file}")
    full_version = ctx.to_full_version(
        ctx.version_after_this, extra_zeros=True
    )
    if ctx.dry_run:
        return
    version_file.write_text(f"{full_version}\n")
    ctx.vprint(f"Updated {version_file} -> {full_version}")


def _pyproject_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"Expected at least major.minor in version, got {version}"
        )
    major, minor = parts[0], parts[1]
    try:
        major_int = int(major)
        minor_int = int(minor)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric version {version}") from exc
    return f"{major_int}.{minor_int}"


def bump_cross_repo_dependencies(ctx: Context) -> None:
    versions_json = ctx.cupynumeric_dir / "cmake" / "versions.json"
    ctx.vprint(f"Opening {versions_json}")
    if not versions_json.is_file():
        raise FileNotFoundError(versions_json)

    with versions_json.open() as fd:
        data = json.load(fd)

    packages = data.setdefault("packages", {})
    if "legate" not in packages:
        raise ValueError(
            "Failed to find 'legate' package entry in versions.json"
        )

    legate_cfg: dict[str, object] = packages["legate"]
    updated = False

    new_version = ctx.to_full_version(ctx.version_after_this, extra_zeros=True)
    if legate_cfg.get("version") != new_version:
        legate_cfg["version"] = new_version
        updated = True

    if ctx.legate_git_tag is not None:
        if legate_cfg.get("git_tag") != ctx.legate_git_tag:
            legate_cfg["git_tag"] = ctx.legate_git_tag
            updated = True
    else:
        ctx.print(
            "--legate-git-tag not provided; leaving cmake/versions.json git_tag unchanged."
        )

    if not updated:
        ctx.vprint("cmake/versions.json already up to date")
        return

    if ctx.dry_run:
        return

    with versions_json.open(mode="w") as fd:
        json.dump(data, fd, indent=4, sort_keys=True)

    ctx.vprint(f"Updated {versions_json}")


def bump_pyproject_dependency(ctx: Context) -> None:
    pyproject = (
        ctx.cupynumeric_dir
        / "scripts"
        / "build"
        / "python"
        / "cupynumeric"
        / "pyproject.toml"
    )
    ctx.vprint(f"Opening {pyproject}")
    if not pyproject.is_file():
        raise FileNotFoundError(pyproject)

    contents = pyproject.read_text()
    new_version = _pyproject_version(ctx.version_after_this)
    needle = '"legate=='
    start = contents.find(needle)
    if start == -1:
        raise ValueError(
            "Failed to find legate dependency pin in pyproject.toml"
        )

    end = contents.find(',>=0.0.0a0"', start)
    if end == -1:
        raise ValueError(
            "Failed to locate legate dependency suffix in pyproject.toml"
        )

    new_contents = (
        contents[: start + len(needle)]
        + f'{new_version}.*,>=0.0.0a0"'
        + contents[end + len(',>=0.0.0a0"') :]
    )

    if new_contents == contents:
        ctx.vprint("pyproject.toml already up to date")
        return

    if ctx.dry_run:
        return

    pyproject.write_text(new_contents)
    ctx.vprint(f"Updated {pyproject}")
