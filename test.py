#!/usr/bin/env python

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
from __future__ import annotations

import sys

from legate.tester import CustomTest, FeatureType
from legate.tester.config import Config
from legate.tester.project import Project
from legate.tester.test_plan import TestPlan
from legate.tester.test_system import TestSystem
from legate.util.types import EnvDict


class CPNProject(Project):
    def custom_files(self) -> list[CustomTest]:
        return [
            CustomTest("examples/cholesky.py"),
            CustomTest("examples/quantiles.py"),
            CustomTest("examples/solve.py"),
            CustomTest("examples/sort.py"),
            CustomTest("tests/integration/test_argsort.py"),
            CustomTest("tests/integration/test_cholesky.py"),
            CustomTest("tests/integration/test_nanpercentiles.py"),
            CustomTest("tests/integration/test_nanquantiles.py"),
            CustomTest("tests/integration/test_partition.py"),
            CustomTest("tests/integration/test_percentiles.py"),
            CustomTest("tests/integration/test_quantiles.py"),
            CustomTest("tests/integration/test_solve.py"),
            CustomTest("tests/integration/test_sort_complex.py"),
            CustomTest("tests/integration/test_sort.py"),
            CustomTest("tests/integration/test_unique.py"),
        ]

    def stage_env(self, feature: FeatureType) -> EnvDict:
        match feature:
            case "eager":
                return {
                    "CUPYNUMERIC_FORCE_THUNK": "eager",
                    "CUPYNUMERIC_MAX_EAGER_VOLUME": "2000000000",
                }
            case _:
                return {}


if __name__ == "__main__":
    config = Config(sys.argv, project=CPNProject())

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    sys.exit(plan.execute())
