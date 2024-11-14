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

from subprocess import run

import pytest

# TODO: (bev) This probably only works in bash. Skipping the tests until
# something more reliable can be implemented.
# legate = os.environ["_"]
legate = ""


@pytest.mark.skip
def test_no_patch() -> None:
    cmd = "import sys; import cupynumeric; import numpy; sys.exit(numpy is cupynumeric)"  # noqa E501
    proc = run([legate, "-c", cmd])
    assert proc.returncode == 0, "numpy is unexpectedly patched"


@pytest.mark.skip
def test_patch() -> None:
    cmd = "import sys; import cupynumeric.patch; import numpy; sys.exit(numpy is cupynumeric)"  # noqa E501
    proc = run([legate, "-c", cmd])
    assert proc.returncode == 1, "numpy failed to patch"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
