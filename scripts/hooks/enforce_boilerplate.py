# Copyright 2024 NVIDIA Corporation
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
import sys
from pathlib import Path
from typing import NoReturn

BOILERPLATE = ["sys.exit(pytest.main(sys.argv))"]

TESTS_TOP = Path(__file__).parents[2] / "tests"

TEST_DIRS = (TESTS_TOP / "unit", TESTS_TOP / "integration")


def enforce_boilerplate() -> NoReturn:
    for dir in TEST_DIRS:
        for path in dir.rglob("*.py"):
            if not path.is_file() or not path.name.startswith("test_"):
                continue

            last_lines = open(path).readlines()[-len(BOILERPLATE) :]
            if any(a.strip() != b for a, b in zip(last_lines, BOILERPLATE)):
                print(f"Test file {path} missing required boilerplate")
                sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    enforce_boilerplate()
