# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


_WHEEL_PACKAGE_DEPS = (
    ("cupynumeric", "legate"),
    ("cupynumeric-cu12", "legate-cu12"),
)


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


def _update_pyproject_dependency(
    path: Path, *, package: str, version: str, dry_run: bool = False
) -> bool:
    if not path.is_file():
        raise FileNotFoundError(path)

    contents = path.read_text()
    pattern = re.compile(
        rf'"{re.escape(package)}==(?P<version>\d+\.\d+)\.\*,>=0\.0\.0a0"'
    )
    if not pattern.search(contents):
        raise ValueError(
            f"Failed to locate {package} dependency pin in {path}"
        )

    new_contents = pattern.sub(
        f'"{package}=={version}.*,>=0.0.0a0"', contents, count=1
    )
    if new_contents == contents:
        return False

    if dry_run:
        return True

    path.write_text(new_contents)
    return True


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
    new_version = _pyproject_version(ctx.version_after_this)

    for wheel_package, dependency_package in _WHEEL_PACKAGE_DEPS:
        pyproject = (
            ctx.cupynumeric_dir
            / "scripts"
            / "build"
            / "python"
            / wheel_package
            / "pyproject.toml"
        )
        ctx.vprint(f"Opening {pyproject}")

        updated = _update_pyproject_dependency(
            pyproject,
            package=dependency_package,
            version=new_version,
            dry_run=ctx.dry_run,
        )

        if updated:
            action = "Would update" if ctx.dry_run else "Updated"
            ctx.vprint(f"{action} {pyproject}")
        else:
            ctx.vprint(f"{pyproject} already up to date")
