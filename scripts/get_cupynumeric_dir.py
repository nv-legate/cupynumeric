#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path


def get_cupynumeric_dir() -> str:
    """Return the absolute path to the cuPyNumeric repository root."""

    return str(Path(__file__).resolve().parents[1])


def main() -> None:
    print(get_cupynumeric_dir(), end="", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
