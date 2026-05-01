#!/usr/bin/env python3

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

import argparse
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ARRAY_API_TESTS_REPO = "https://github.com/data-apis/array-api-tests.git"
# The upstream suite release and the Array API standard version are separate
# pins. Keep the suite tag SHA-pinned so local baseline runs are repeatable.
ARRAY_API_TESTS_TAG = "2026.02.26"
ARRAY_API_TESTS_SHA = "41379d15d26d67a1e66c840e775d41a8a7fb1516"
ARRAY_API_VERSION = "2025.12"
DEFAULT_DISABLED_EXTENSIONS = ("linalg", "fft")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKOUT = ROOT / ".tmp" / "array-api-tests" / ARRAY_API_TESTS_TAG
ARRAY_API_TEST_DIR = ROOT / "tests" / "array_api"
DEFAULT_CONSTRAINTS = (
    ARRAY_API_TEST_DIR / f"constraints-{ARRAY_API_VERSION}.txt"
)
DEFAULT_CORE_SKIPS = ARRAY_API_TEST_DIR / f"skips-{ARRAY_API_VERSION}-core.txt"
DEFAULT_XFAILS = ARRAY_API_TEST_DIR / f"xfails-{ARRAY_API_VERSION}.txt"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def output(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def has_pinned_commit(path: Path) -> bool:
    commit = f"{ARRAY_API_TESTS_SHA}^{{commit}}"
    return (
        subprocess.run(
            ["git", "cat-file", "-e", commit],
            cwd=path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def has_ref(path: Path, ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", ref],
            cwd=path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def is_git_checkout(path: Path) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def check_clean_git_tree(path: Path, label: str) -> None:
    status = output(["git", "status", "--short"], cwd=path)
    if status:
        msg = (
            f"{label} has local modifications:\n{status}\n"
            "Clean the checkout or remove it so the runner can recreate the "
            "pinned upstream baseline."
        )
        raise RuntimeError(msg)


def ensure_checkout(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--branch",
                ARRAY_API_TESTS_TAG,
                ARRAY_API_TESTS_REPO,
                str(path),
            ]
        )
    elif not (path / ".git").exists():
        msg = f"{path} exists but is not a git checkout"
        raise RuntimeError(msg)

    check_clean_git_tree(path, "array-api-tests checkout")
    submodule = path / "array-api"
    if submodule.exists() and is_git_checkout(submodule):
        check_clean_git_tree(submodule, "array-api submodule")

    tag_ref = f"refs/tags/{ARRAY_API_TESTS_TAG}"
    if not has_pinned_commit(path) or not has_ref(path, tag_ref):
        run(["git", "fetch", "origin", "tag", ARRAY_API_TESTS_TAG], cwd=path)

    tag_sha = output(["git", "rev-parse", f"{tag_ref}^{{commit}}"], cwd=path)
    if tag_sha != ARRAY_API_TESTS_SHA:
        msg = (
            f"array-api-tests tag {ARRAY_API_TESTS_TAG} resolves to "
            f"{tag_sha}, expected {ARRAY_API_TESTS_SHA}"
        )
        raise RuntimeError(msg)

    run(["git", "checkout", "--detach", ARRAY_API_TESTS_SHA], cwd=path)
    actual_sha = output(["git", "rev-parse", "HEAD"], cwd=path)
    if actual_sha != ARRAY_API_TESTS_SHA:
        msg = (
            f"array-api-tests checkout is at {actual_sha}, "
            f"expected {ARRAY_API_TESTS_SHA}"
        )
        raise RuntimeError(msg)

    run(["git", "submodule", "update", "--init"], cwd=path)
    check_clean_git_tree(path, "array-api-tests checkout")
    if submodule.exists() and is_git_checkout(submodule):
        check_clean_git_tree(submodule, "array-api submodule")


def pinned_constraints(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, version = stripped.partition("==")
        if separator:
            pins[name] = version
    return pins


def check_test_dependencies(checkout: Path) -> None:
    mismatches: list[str] = []
    for name, expected in pinned_constraints(DEFAULT_CONSTRAINTS).items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{name} missing, expected {expected}")
            continue
        if actual != expected:
            mismatches.append(f"{name}=={actual}, expected {expected}")

    if mismatches:
        install = (
            f"{sys.executable} -m pip install -r "
            f"{checkout / 'requirements.txt'} -c {DEFAULT_CONSTRAINTS}"
        )
        msg = (
            "Array API test dependency versions do not match the local "
            "constraints:\n"
            + "\n".join(f"- {item}" for item in mismatches)
            + f"\nInstall the pinned local prerequisites with: {install}"
        )
        raise RuntimeError(msg)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned upstream array-api-tests baseline for cuPyNumeric."
        )
    )
    parser.add_argument(
        "--checkout",
        type=Path,
        default=DEFAULT_CHECKOUT,
        help=(
            "array-api-tests checkout path "
            f"(default: {DEFAULT_CHECKOUT.relative_to(ROOT)})"
        ),
    )
    parser.add_argument(
        "--xfails-file",
        type=Path,
        default=DEFAULT_XFAILS,
        help=(
            "expected-failures file "
            f"(default: {DEFAULT_XFAILS.relative_to(ROOT)})"
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="forwarded to upstream pytest as --max-examples",
    )
    parser.add_argument(
        "--include-extensions",
        action="store_true",
        help=(
            "remove the default linalg and fft extension skips. The xfails "
            "file still applies unless a different --xfails-file is passed."
        ),
    )
    parser.add_argument(
        "--run-expected-failures",
        action="store_true",
        help=(
            "run entries from the xfails file as pytest xfails. By default "
            "they are skipped so native aborts stay reproducible. XPASS "
            "results are strict in this mode."
        ),
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="prepare the pinned upstream checkout and exit before pytest",
    )
    parser.add_argument(
        "--pytest-target",
        action="append",
        help=(
            "pytest target to run from the upstream checkout. May be "
            "specified more than once. Defaults to array_api_tests/."
        ),
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help=(
            "extra pytest options and filters; prefix with -- to separate "
            "them. Use --pytest-target for positional test selectors."
        ),
    )
    return parser


def clean_pytest_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> int:
    args = parser().parse_args()
    checkout = args.checkout.resolve()
    xfails_file = args.xfails_file.resolve()

    ensure_checkout(checkout)
    if args.setup_only:
        print(
            "Install upstream test dependencies with: "
            f"{sys.executable} -m pip install -r "
            f"{checkout / 'requirements.txt'} -c {DEFAULT_CONSTRAINTS}",
            flush=True,
        )
        return 0

    if not xfails_file.is_file():
        msg = f"expected-failures file does not exist: {xfails_file}"
        raise RuntimeError(msg)
    if not args.include_extensions and not DEFAULT_CORE_SKIPS.is_file():
        msg = f"core skip file does not exist: {DEFAULT_CORE_SKIPS}"
        raise RuntimeError(msg)
    check_test_dependencies(checkout)

    pytest_targets = args.pytest_target or ["array_api_tests/"]
    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        *pytest_targets,
        "-c",
        "pytest.ini",
        "--rootdir",
        ".",
        "--xfails-file",
        str(xfails_file),
    ]
    if not args.include_extensions:
        pytest_args.extend(
            [
                "--skips-file",
                str(DEFAULT_CORE_SKIPS),
                "--disable-extension",
                *DEFAULT_DISABLED_EXTENSIONS,
            ]
        )
    if args.max_examples is not None:
        pytest_args.extend(["--max-examples", str(args.max_examples)])
    if args.run_expected_failures:
        pytest_args.extend(["-o", "xfail_strict=true"])
    pytest_args.extend(clean_pytest_args(args.pytest_args))

    env = os.environ.copy()
    env["ARRAY_API_TESTS_MODULE"] = "cupynumeric"
    env["ARRAY_API_TESTS_VERSION"] = ARRAY_API_VERSION
    env["ARRAY_API_TESTS_XFAIL_MARK"] = (
        "xfail" if args.run_expected_failures else "skip"
    )
    # Run pytest from the upstream checkout while importing this cuPyNumeric
    # source tree as the array module under test.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(ROOT)]
    )

    print(f"array-api-tests tag: {ARRAY_API_TESTS_TAG}", flush=True)
    print(f"array-api-tests sha: {ARRAY_API_TESTS_SHA}", flush=True)
    print(f"Array API version: {ARRAY_API_VERSION}", flush=True)
    if not args.include_extensions:
        print(f"core skips file: {DEFAULT_CORE_SKIPS}", flush=True)
    print(f"xfails file: {xfails_file}", flush=True)
    print("+", " ".join(pytest_args), flush=True)
    return subprocess.run(pytest_args, cwd=checkout, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
