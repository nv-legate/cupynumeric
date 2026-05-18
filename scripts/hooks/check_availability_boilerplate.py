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
"""Pre-commit hook: every function/method whose docstring contains an
"Availability" section heading must be decorated with @add_boilerplate.
"""

import ast
import re
import sys
from pathlib import Path

# Matches "Availability" as a NumPy-style section heading, i.e. the word
# "Availability" (possibly indented) followed on the next line by dashes.
AVAILABILITY_RE = re.compile(r"^\s*Availability\s*\n\s*-+", re.MULTILINE)

# Decorators that don't compose with @add_boilerplate. Wrapping a property
# (or other descriptor) would wrap the descriptor object rather than the
# underlying function, breaking attribute access.
DESCRIPTOR_DECORATORS = {
    "property",
    "staticmethod",
    "classmethod",
    "cached_property",
}


def _is_descriptor(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for dec in node.decorator_list:
        text = ast.unparse(dec)
        # bare name or attribute reference (e.g. functools.cached_property)
        leaf = text.rsplit(".", 1)[-1]
        if leaf in DESCRIPTOR_DECORATORS:
            return True
        # @foo.setter / @foo.getter / @foo.deleter on a property
        if text.endswith((".setter", ".getter", ".deleter")):
            return True
    return False


def _is_private(name: str) -> bool:
    # Private helpers: single leading underscore. Dunders (`__init__`,
    # `__getitem__`, …) are public-facing and don't count.
    return name.startswith("_") and not (
        name.startswith("__") and name.endswith("__")
    )


def check_file(path: Path) -> list[str]:
    try:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: could not parse: {exc}"]

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = ast.get_docstring(node)
        if not docstring or not AVAILABILITY_RE.search(docstring):
            continue
        if _is_descriptor(node) or _is_private(node.name):
            continue
        has_boilerplate = any(
            "add_boilerplate" in ast.unparse(dec)
            for dec in node.decorator_list
        )
        if not has_boilerplate:
            violations.append(
                f"{path}:{node.lineno}: '{node.name}' has an 'Availability' "
                f"docstring section but is not decorated with @add_boilerplate"
            )
    return violations


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:] if p.endswith(".py")]
    violations = []
    for path in paths:
        violations.extend(check_file(path))

    if violations:
        print(
            "ERROR: the following functions have an 'Availability' docstring "
            "section but are missing the @add_boilerplate decorator:\n",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
