#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re
import sys
import json
import argparse
from pathlib import Path

# Capture the full upper-bound token to avoid partial matches (e.g., "1!2.0").
_UPPER_BOUND_RE = re.compile(r"(<=?)([^,|\s]+)")
_KEY_RE_TEMPLATE = r"^\s*{dep}_version:\s*$"
_LIST_ITEM_RE = re.compile(
    r"^(?P<indent>\s*)-\s*(?P<quote>[\"'])(?P<value>.*?)(?P=quote)(?P<suffix>\s*(?:#.*)?)$"
)
_MARKER_RE = re.compile(
    r"^\s*# renovate-conda-bound:\s+dep=(?P<dep>\S+)\s+"
    r"newVersion=(?P<new_version>\S+)\s+currentValue=(?P<current_value>.+?)\s*$"
)
_STRICT_VERSION_RE = re.compile(r"^\d+(?:\.\d+)*$")
_DATA_FILE_ENV = "RENOVATE_POST_UPGRADE_COMMAND_DATA_FILE"


def _debug_enabled() -> bool:
    return (
        os.environ.get("RENOVATE_HELPER_DEBUG") == "1"
        or os.environ.get("LOG_LEVEL", "").lower() == "debug"
    )


def _debug(message: str) -> None:
    if _debug_enabled():
        sys.stderr.write(f"DEBUG: {message}\n")


def _load_post_upgrade_data() -> object | None:
    data_path = os.environ.get(_DATA_FILE_ENV)
    if not data_path:
        return None

    path = Path(data_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"Post-upgrade data file not found: {path}"
        raise ValueError(msg) from exc
    except OSError as exc:
        msg = f"Unable to read post-upgrade data file {path}: {exc}"
        raise ValueError(msg) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in post-upgrade data file {path}: {exc}"
        raise ValueError(msg) from exc


def _iter_upgrade_entries(data: object | None) -> list[dict[str, object]]:
    if data is None:
        return []
    if isinstance(data, dict):
        upgrades = data.get("upgrades")
        entries = upgrades if isinstance(upgrades, list) else [data]
    elif isinstance(data, list):
        entries = data
    else:
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _current_value_from_data(
    data: object | None, *, dep: str, new_version: str, package_file: str
) -> str | None:
    for entry in _iter_upgrade_entries(data):
        if entry.get("depName") != dep:
            continue
        entry_package = entry.get("packageFile")
        if entry_package and entry_package != package_file:
            continue
        entry_new_version = entry.get("newVersion") or entry.get("newValue")
        if entry_new_version and entry_new_version != new_version:
            continue
        current_value = entry.get("currentValue")
        if isinstance(current_value, str) and current_value:
            return current_value
    return None


def _parse_numeric_segments(version: str, *, context: str) -> list[int]:
    if not _STRICT_VERSION_RE.match(version):
        msg = (
            f"Unsupported {context} {version!r}; expected digits and dots "
            "only."
        )
        raise ValueError(msg)
    return [int(part) for part in version.split(".")]


def _upper_bound_precision(op: str, upper: str) -> tuple[int, int]:
    segments = _parse_numeric_segments(upper, context="upper bound")
    if op != "<":
        return len(segments), 0

    last_non_zero = None
    for index in range(len(segments) - 1, -1, -1):
        if segments[index] != 0:
            last_non_zero = index
            break

    precision = 1 if last_non_zero is None else last_non_zero + 1

    trailing_zeros = len(segments) - precision
    return precision, trailing_zeros


def _replace_last_upper_bound(constraint: str, *, new_version: str) -> str:
    matches = list(_UPPER_BOUND_RE.finditer(constraint))
    if not matches:
        return constraint

    for match in matches:
        _parse_numeric_segments(match.group(2), context="upper bound")

    last = matches[-1]
    op = last.group(1)
    old_upper = last.group(2)
    precision, trailing_zeros = _upper_bound_precision(op, old_upper)

    segments = _parse_numeric_segments(new_version, context="new version")
    new_len = len(segments)
    extra_segments = segments[precision:] if new_len > precision else []
    if new_len < precision:
        segments.extend([0] * (precision - len(segments)))
    else:
        segments = segments[:precision]

    if op == "<":
        segments[-1] += 1
        if trailing_zeros:
            segments.extend([0] * trailing_zeros)
    elif op == "<=" and new_len > precision and any(extra_segments):
        # Round up to the next representable value at this precision.
        segments[-1] += 1

    new_upper = ".".join(str(value) for value in segments)
    if new_upper == old_upper:
        return constraint

    start, end = last.span(2)
    return f"{constraint[:start]}{new_upper}{constraint[end:]}"


def _is_block_line(line: str) -> bool:
    if line.strip() == "":
        return True
    if line.lstrip().startswith("#"):
        return True
    return _LIST_ITEM_RE.match(line.rstrip("\n")) is not None


def _find_block(
    lines: list[str], *, key_re: re.Pattern[str]
) -> tuple[int, int] | None:
    for index, line in enumerate(lines):
        if not key_re.match(line):
            continue
        end = index + 1
        while end < len(lines) and _is_block_line(lines[end]):
            end += 1
        return index, end
    return None


def _updated_block_lines(
    block_lines: list[str],
    *,
    new_version: str,
    marker_current_value: str | None = None,
    data_current_value: str | None = None,
) -> tuple[list[str], bool]:
    updated_lines: list[str] = []
    changed = False
    marker_used = False
    apply_marker_to_next = False
    for line in block_lines:
        marker_match = _MARKER_RE.match(line)
        if marker_match:
            changed = True
            if marker_current_value is not None:
                apply_marker_to_next = True
            continue
        match = _LIST_ITEM_RE.match(line.rstrip("\n"))
        if match is None:
            updated_lines.append(line)
            continue

        indent = match.group("indent")
        quote = match.group("quote")
        value = match.group("value")
        if (
            marker_current_value is not None
            and apply_marker_to_next
            and not marker_used
        ):
            value = marker_current_value
            marker_used = True
            apply_marker_to_next = False
        elif (
            marker_current_value is None
            and data_current_value is not None
            and not marker_used
        ):
            value = data_current_value
            marker_used = True
        suffix = match.group("suffix")
        updated_value = _replace_last_upper_bound(
            value, new_version=new_version
        )
        if updated_value != value:
            changed = True
        updated_lines.append(
            f"{indent}- {quote}{updated_value}{quote}{suffix}\n"
        )
    return updated_lines, changed


def _update_file(
    path: Path, *, dep: str, new_version: str, root: Path, data: object | None
) -> bool:
    current_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    key_re = re.compile(_KEY_RE_TEMPLATE.format(dep=re.escape(dep)))

    current_block = _find_block(current_lines, key_re=key_re)
    if current_block is None:
        return False
    start, end = current_block
    current_block_lines = current_lines[start:end]
    marker_current_value = None
    for line in current_block_lines:
        marker_match = _MARKER_RE.match(line)
        if marker_match and marker_match.group("dep") == dep:
            marker_current_value = marker_match.group("current_value").strip()
            _debug(f"Marker currentValue for {dep}: {marker_current_value}")
            break

    rel_path = path.resolve().relative_to(root).as_posix()
    data_current_value = _current_value_from_data(
        data, dep=dep, new_version=new_version, package_file=rel_path
    )
    if data_current_value:
        _debug(
            "Data-file currentValue for "
            f"{dep} in {rel_path}: {data_current_value}"
        )
    elif data is not None and marker_current_value is None:
        msg = (
            "Post-upgrade data file missing currentValue for "
            f"{dep} in {rel_path} (newVersion={new_version})."
        )
        raise ValueError(msg)

    updated_block, updated = _updated_block_lines(
        current_block_lines,
        new_version=new_version,
        marker_current_value=marker_current_value,
        data_current_value=data_current_value,
    )

    changed = updated or current_lines[start:end] != updated_block
    if not changed:
        return False

    current_lines[start:end] = updated_block
    path.write_text("".join(current_lines), encoding="utf-8")
    return True


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Update conda_build_config.yaml upper bounds (Renovate helper)."
        )
    )
    parser.add_argument(
        "--dep", required=True, help="Dependency name (e.g. numpy)"
    )
    parser.add_argument(
        "--new-version",
        required=True,
        help="New upstream version (e.g. 2.5.1)",
    )
    args = parser.parse_args()

    try:
        _parse_numeric_segments(args.new_version, context="new version")
        root = _repo_root()
        data = _load_post_upgrade_data()
        # Keep upper bounds consistent across all conda subpackages.
        for path in sorted(root.glob("conda/**/conda_build_config.yaml")):
            _update_file(
                path,
                dep=args.dep,
                new_version=args.new_version,
                root=root,
                data=data,
            )
    except ValueError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
