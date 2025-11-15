# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .context import Context


class SwitcherData(TypedDict, total=False):
    name: str
    preferred: bool
    url: str
    version: str


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
