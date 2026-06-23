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
"""Enforce that every pytest xfail/skip/skipif marker carries a classified
reason, and that severe failures reference a tracking issue.

reason grammar:
    [severe|not-severe|triage] <category>: <description> (#<issue>)

Rules:
    * bare @pytest.mark.xfail / .skip  -> violation (nowhere to put a reason)
    * .xfail(...)/.skip(...)/.skipif(...) must pass a non-empty reason=
      that is a string literal (so it can be statically classified)
    * the reason must start with a [severity] tag
    * [severe] reasons must reference an issue (e.g. #732)
    * imperative pytest.skip()/xfail() must carry a message (no format rule)

A baseline file grandfathers the existing backlog so this can be adopted
incrementally; it is a ratchet that may only shrink (see --update-baseline).

Enforcement has two layers:
    * pre-commit / normal mode blocks *new code* violations that aren't in
      the working-tree baseline.
    * --against <ref> (run in CI) blocks growth of the baseline *file* itself
      relative to the merge-base, so a PR can't both add a violation and pad
      the baseline to hide it. The baseline may only shrink.
"""

from __future__ import annotations

import ast
import re
import sys
import subprocess
import warnings
from collections import Counter
from pathlib import Path

# Parsing test files can surface unrelated SyntaxWarnings (e.g. invalid escape
# sequences in regex strings); keep the hook output focused on policy issues.
warnings.filterwarnings("ignore", category=SyntaxWarning)

REPO = Path(__file__).resolve().parents[2]
TEST_DIRS = (REPO / "tests" / "unit", REPO / "tests" / "integration")
BASELINE = Path(__file__).with_name("marker_reason_baseline.txt")

MARKERS = {"xfail", "skip", "skipif"}
IMPERATIVE = {"skip", "xfail"}
SEVERITIES = ("severe", "not-severe", "triage")
CATEGORIES = {
    "wrong-result",
    "hang",
    "crash",
    "unexpected-exception",
    "not-implemented",
    "exception-mismatch",
    "dtype-divergence",
    "gpu-only",
    "numpy-version",
    "other",
}
TAG_RE = re.compile(rf"^\[({'|'.join(SEVERITIES)})\]\s+(.+)$", re.S)
ISSUE_RE = re.compile(r"(?:[\w.-]+/[\w.-]+)?#\d+")


def is_mark(node: ast.AST) -> str | None:
    """Return X if node is `*.mark.<X>` and X needs a reason, else None."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "mark"
        and node.attr in MARKERS
    ):
        return node.attr
    return None


def const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def validate_reason(text: str) -> tuple[str, str] | None:
    """Return (code, message) if the reason is malformed, else None.

    The code is a stable violation identifier baked into the baseline key, so
    swapping one malformed reason for a structurally different one in the same
    scope is detected even when the total count per scope is unchanged.
    """
    m = TAG_RE.match(text.strip())
    if not m:
        return (
            "bad-tag",
            "reason must start with [severe]/[not-severe]/[triage]",
        )
    sev, rest = m.group(1), m.group(2)
    if sev == "severe" and not ISSUE_RE.search(rest):
        return (
            "severe-no-issue",
            "[severe] reason must reference an issue (e.g. #732)",
        )
    if ":" in rest:
        cat = rest.split(":", 1)[0].strip()
        if cat and " " not in cat and cat not in CATEGORIES:
            return ("unknown-category", f"unknown category '{cat}'")
    return None


class Visitor(ast.NodeVisitor):
    def __init__(self, relpath: str) -> None:
        self.rel = relpath
        self.scope: list[str] = []
        self.errs: list[tuple[int, str, str]] = []  # (lineno, key, msg)
        self._called: set[int] = set()

    def _key(self, kind: str, code: str) -> str:
        qual = ".".join(self.scope) if self.scope else "<module>"
        return f"{self.rel}::{qual}::{kind}::{code}"

    def _emit(self, node: ast.AST, kind: str, code: str, msg: str) -> None:
        self.errs.append((node.lineno, self._key(kind, code), msg))

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def _visit_func(self, node: ast.AST) -> None:
        self.scope.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _visit_func  # type: ignore[assignment]
    visit_AsyncFunctionDef = _visit_func  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        kind = is_mark(node.func)
        if kind is not None:
            self._called.add(id(node.func))
            reason_kw = next(
                (k for k in node.keywords if k.arg == "reason"), None
            )
            if reason_kw is None:
                self._emit(
                    node,
                    kind,
                    "missing-reason",
                    f"@pytest.mark.{kind}(...) missing reason=",
                )
            else:
                text = const_str(reason_kw.value)
                if text is None:
                    # A non-literal reason (variable, f-string, concat, call)
                    # can't be statically classified, so it would silently
                    # bypass the severity/issue checks. Require a literal.
                    self._emit(
                        node,
                        kind,
                        "non-literal",
                        f"@pytest.mark.{kind}(reason=...) must be a string "
                        "literal so the severity tag can be checked",
                    )
                elif (err := validate_reason(text)) is not None:
                    self._emit(node, kind, err[0], err[1])
        else:
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr in IMPERATIVE
                and isinstance(f.value, ast.Name)
                and f.value.id == "pytest"
            ):
                ok = bool(node.args) or any(
                    k.arg in ("reason", "msg") for k in node.keywords
                )
                if not ok:
                    self._emit(
                        node,
                        f"{f.attr}()",
                        "missing-message",
                        f"pytest.{f.attr}() missing message",
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        kind = is_mark(node)
        if kind is not None and id(node) not in self._called:
            self._emit(
                node,
                kind,
                "bare",
                f"@pytest.mark.{kind} is bare; needs reason=",
            )
        self.generic_visit(node)


def scan(path: Path) -> list[tuple[int, str, str]]:
    abspath = (path if path.is_absolute() else Path.cwd() / path).resolve()
    try:
        rel = abspath.relative_to(REPO).as_posix()
    except ValueError:
        rel = path.as_posix()  # outside the repo; key by the path as given
    tree = ast.parse(abspath.read_text(encoding="utf-8"), filename=rel)
    v = Visitor(rel)
    # visit_Call records id(node.func) before generic_visit descends into it,
    # so visit_Attribute sees called marks as already-handled without a pre-walk.
    v.visit(tree)
    return v.errs


def parse_baseline(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            n, key = line.split("\t", 1)
            counts[key] = int(n)
    return counts


def load_baseline() -> Counter[str]:
    if BASELINE.exists():
        return parse_baseline(BASELINE.read_text(encoding="utf-8"))
    return Counter()


def baseline_at_ref(ref: str) -> Counter[str] | None:
    """Parse the baseline as it existed at a git ref, or None if unavailable."""
    relpath = BASELINE.relative_to(REPO).as_posix()
    try:
        proc = subprocess.run(
            ["git", "show", f"{ref}:{relpath}"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:  # git not on PATH
        return None
    if proc.returncode != 0:  # ref or file absent at that ref
        return None
    return parse_baseline(proc.stdout)


def check_against(ref: str) -> int:
    """Refuse any growth of the committed baseline relative to `ref`.

    The normal hook trusts the working-tree baseline to decide what's
    grandfathered, so a PR could add a violation *and* pad the baseline to
    match and still pass. This ratchet closes that hole in CI by comparing
    the committed baseline against its merge-base version: the file may only
    shrink. Intentional growth (e.g. a key-format migration) is not possible
    through this gate by design and must be landed via a deliberate bypass.
    """
    old = baseline_at_ref(ref)
    if old is None:
        print(
            f"marker-reason ratchet: baseline not found at {ref!r}; skipping "
            "(expected off-CI or before the baseline existed at that ref)."
        )
        return 0
    cur = load_baseline()
    grew = {k: (old.get(k, 0), cur[k]) for k in cur if cur[k] > old.get(k, 0)}
    if grew:
        print(f"Baseline grew vs {ref} (the ratchet may only shrink):\n")
        for k, (o, n) in sorted(grew.items()):
            print(f"  {o} -> {n}  {k}")
        print(
            "\nThe grandfather baseline may not gain violations. Fix the new "
            "markers to comply instead of adding them to the baseline."
        )
        return 1
    print(
        f"OK: baseline did not grow vs {ref} "
        f"({sum(cur.values())} grandfathered)."
    )
    return 0


def write_baseline(counts: Counter[str]) -> None:
    lines = [
        "# Auto-generated marker-reason baseline (ratchet: only shrink me).",
        "# Regenerate after cleaning markers with --update-baseline.",
        "# Format: <count>\\t<relpath>::<qualname>::<marker>::<violation-code>",
    ]
    lines += [f"{counts[k]}\t{k}" for k in sorted(counts)]
    BASELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gather_paths(argv: list[str]) -> list[Path]:
    files = [Path(a) for a in argv if a.endswith(".py")]
    if files:
        return files
    return [p for d in TEST_DIRS for p in d.rglob("*.py")]


def main(argv: list[str]) -> int:
    update = "--update-baseline" in argv
    allow_grow = "--allow-grow" in argv

    against: str | None = None
    has_against = False
    for i, a in enumerate(argv):
        if a == "--against":
            has_against = True
            against = argv[i + 1] if i + 1 < len(argv) else None
        elif a.startswith("--against="):
            has_against = True
            against = a.split("=", 1)[1]
    if has_against:
        if not against:
            print("--against requires a git ref, e.g. --against origin/main")
            return 2
        return check_against(against)

    rest = [a for a in argv if not a.startswith("--")]

    errs = [e for p in gather_paths(rest) if p.is_file() for e in scan(p)]
    cur = Counter(key for _, key, _ in errs)

    if update:
        base = load_baseline()
        grew = {
            k: (base.get(k, 0), cur[k]) for k in cur if cur[k] > base.get(k, 0)
        }
        if grew and not allow_grow:
            print("Refusing to grow the baseline (ratchet may only shrink):\n")
            for k, (old, new) in sorted(grew.items()):
                print(f"  {old} -> {new}  {k}")
            print(
                "\nFix the new violations, or pass --allow-grow to defer "
                "them intentionally."
            )
            return 1
        write_baseline(cur)
        print(
            f"Baseline written: {sum(cur.values())} grandfathered violations "
            f"across {len(cur)} sites."
        )
        return 0

    base = load_baseline()
    bad = [
        (ln, key, msg) for ln, key, msg in errs if cur[key] > base.get(key, 0)
    ]
    if bad:
        print("Marker-reason policy violations (not grandfathered):\n")
        for ln, key, msg in sorted(bad):
            relpath = key.split("::", 1)[0]
            print(f"  {relpath}:{ln}: {msg}")
        print(
            "\nAdd a [severe]/[not-severe]/[triage] reason (severe must link an "
            "issue), or run --update-baseline to defer intentionally."
        )
        return 1
    print(
        f"OK: no new marker-reason violations "
        f"({sum(base.values())} still grandfathered)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
