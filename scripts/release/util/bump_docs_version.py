# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from .context import Context


class SwitcherData(TypedDict, total=False):
    name: str
    preferred: bool
    url: str
    version: str


DEFAULT_CHANGELOG = """\
..
  SPDX-FileCopyrightText: Copyright (c) 2025-{current_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

Changes: {version}
====================

.. rubric:: Highlights

.. rubric:: Improvements

.. rubric:: Bug Fixes

.. rubric:: Deprecations

""".strip()


DEFAULT_INDEX = """\
..
  SPDX-FileCopyrightText: Copyright (c) 2025-{current_year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

Change Log
==========

.. toctree::
   :maxdepth: 1

   dev

""".strip()


def rotate_switcher(ctx: Context) -> None:
    switcher_json = (
        ctx.cupynumeric_dir / "docs" / "cupynumeric" / "switcher.json"
    )
    ctx.vprint(f"Opening {switcher_json}")
    if not switcher_json.is_file():
        raise FileNotFoundError(switcher_json)

    with switcher_json.open() as fd:
        data: list[SwitcherData] = json.load(fd)

    for entry in data:
        if entry.get("preferred", False):
            last_release = entry
            break
    else:  # pragma: no cover - malformed switcher
        raise ValueError("Failed to find preferred release in switcher.json")

    dev_index: int | None = None
    dev_entry = None
    for idx, entry in enumerate(data):
        if entry.get("version") == "dev":
            dev_index = idx
            dev_entry = entry
            break

    if (
        dev_entry is None or dev_index is None
    ):  # pragma: no cover - malformed switcher
        raise ValueError("Failed to find dev entry in switcher.json")

    expected_keys = {"name", "url", "version"}
    if not set(last_release).issuperset(expected_keys):
        raise ValueError(
            f"Switcher entry missing expected keys {expected_keys}: {last_release}"
        )

    # Remove the preferred flag from the previous release so the new entry can
    # become the preferred version.
    if "preferred" in last_release:
        preferred = last_release.pop("preferred")
        if preferred is not True:
            raise ValueError(
                "Expected the preferred release to have 'preferred': true"
            )

    last_version = last_release["version"].strip()
    new_version = ctx.version_after_this
    if last_version == new_version:
        ctx.vprint(f"Switcher already points to {new_version}")
        return

    data.pop(dev_index)
    new_release: SwitcherData = {
        "name": new_version,
        "preferred": True,
        "url": last_release["url"].replace(last_version, new_version),
        "version": new_version,
    }

    data.append(new_release)
    data.append(dev_entry)

    if not ctx.dry_run:
        with switcher_json.open(mode="w") as fd:
            json.dump(data, fd, indent=4, sort_keys=True)
    ctx.vprint(f"Updated {switcher_json} to {new_version}")


def _changes_dir(ctx: Context) -> Path:
    changes = (
        ctx.cupynumeric_dir / "docs" / "cupynumeric" / "source" / "changes"
    )
    if not changes.is_dir() and not ctx.dry_run:
        changes.mkdir(parents=True, exist_ok=True)
    return changes


def _ensure_index(changes_dir: Path, ctx: Context) -> None:
    index = changes_dir / "index.rst"
    if index.is_file():
        return

    if not ctx.dry_run:
        text = DEFAULT_INDEX.format(current_year=datetime.now().year)
        index.write_text(text)
        ctx.run_cmd(["git", "add", str(index)])
    ctx.vprint(f"Created {index}")


def _rotate_log_file(changes_dir: Path, ctx: Context) -> Path:
    ver_file = ctx.version_after_this.replace(".", "")
    new_log = changes_dir / f"{ver_file}.rst"

    if new_log.is_file():
        ctx.vprint(f"Changelog already exists: {new_log}")
        return new_log

    header = DEFAULT_CHANGELOG.format(
        version=ctx.version_after_this, current_year=datetime.now().year
    )
    if not ctx.dry_run:
        new_log.write_text(header)
        ctx.run_cmd(["git", "add", str(new_log)])
    ctx.vprint(f"Wrote new log to {new_log}")
    return new_log


def _update_symlink(changes_dir: Path, new_log: Path, ctx: Context) -> None:
    dev_link = changes_dir / "dev.rst"

    def create_link() -> None:
        if ctx.dry_run:
            return
        if dev_link.exists() or dev_link.is_symlink():
            dev_link.unlink()
        dev_link.symlink_to(new_log.name)
        ctx.run_cmd(["git", "add", str(dev_link)])
        ctx.vprint(f"Created symlink {dev_link} -> {new_log.name}")

    if not dev_link.exists():
        create_link()
        return

    if not dev_link.is_symlink():
        raise ValueError(f"Expected {dev_link} to be a symlink")

    if dev_link.readlink().resolve() == new_log.resolve():
        ctx.vprint(f"dev symlink already up to date: {dev_link}")
        return

    create_link()


def update_changelog(ctx: Context) -> None:
    changes_dir = _changes_dir(ctx)
    _ensure_index(changes_dir, ctx)
    new_log = _rotate_log_file(changes_dir, ctx)
    _update_symlink(changes_dir, new_log, ctx)
