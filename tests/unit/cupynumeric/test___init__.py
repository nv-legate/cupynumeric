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

import re
from importlib import reload
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

import cupynumeric  # noqa: [F401]


def test___version___override(monkeypatch: pytest.MonkeyPatch) -> None:
    global cupynumeric  # noqa: PLW0603
    monkeypatch.setenv("CUPYNUMERIC_USE_VERSION", "24.01.00")
    cupynumeric = reload(cupynumeric)
    assert cupynumeric.__version__ == "24.01.00"


def test___version___format() -> None:
    global cupynumeric  # noqa: PLW0603
    cupynumeric = reload(cupynumeric)

    # just being cautious, if the test are functioning properly, the
    # actual non-overriden version should never equal the bogus version
    # from test___version___override above
    assert cupynumeric.__version__ != "24.01.00"

    assert re.match(r"^\d{2}\.\d{2}\.\d{2}$", cupynumeric.__version__[:8])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
