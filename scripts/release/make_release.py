#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from argparse import ArgumentParser

from util.bump_docs_version import rotate_switcher
from util.bump_versions import (
    bump_cross_repo_dependencies,
    bump_pyproject_dependency,
    bump_version_file,
)
from util.context import Context


def parse_args() -> Context:
    parser = ArgumentParser()
    parser.add_argument(
        "--version-after-this",
        required=True,
        help=(
            "The next version after this release. If we are about to release "
            "25.01, this should be e.g. 25.03."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices={"post-cut"},
        help="Release process mode",
    )
    parser.add_argument(
        "--legate-git-tag",
        help=(
            "Tag or SHA for the Legate dependency in cmake/versions.json. "
            "If omitted, the existing value is left unchanged."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="Dry-run")
    args = parser.parse_args()
    return Context(args)


def post_cut(ctx: Context) -> None:
    rotate_switcher(ctx)
    bump_version_file(ctx)
    bump_cross_repo_dependencies(ctx)
    bump_pyproject_dependency(ctx)


def main() -> None:
    ctx = parse_args()

    match ctx.mode:
        case "post-cut":
            post_cut(ctx)
        case _:
            # TODO: port the remaining release flow (e.g. cut-branch) from legate.internal.
            raise ValueError(ctx.mode)


if __name__ == "__main__":
    main()
