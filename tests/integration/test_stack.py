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

import inspect

import pytest

import cupynumeric._utils.stack as m  # module under test


def test_find_last_user_stacklevel() -> None:
    n = m.find_last_user_stacklevel()
    assert isinstance(n, int)
    assert n == 1


def test_get_line_number_from_frame() -> None:
    frame = inspect.currentframe()
    result = m.get_line_number_from_frame(frame)
    assert isinstance(result, str)
    filename, lineno = result.split(":")

    # NOTE: this will break if this test filename is changed
    assert filename.endswith("test_stack.py")

    # it would be too fragile to compare more specific than this
    assert int(lineno) > 0


class Test_find_last_user_line_numbers:
    def test_default_top_only(self) -> None:
        result = m.find_last_user_line_numbers(top_only=True)
        assert isinstance(result, str)
        assert "|" not in result
        assert "\n" not in result
        assert len(result.split(":")) == 2

    def test_top_only_True(self) -> None:
        result = m.find_last_user_line_numbers(top_only=True)
        assert isinstance(result, str)
        assert "|" not in result
        assert "\n" not in result
        assert len(result.split(":")) == 2

    def test_top_only_False(self) -> None:
        result = m.find_last_user_line_numbers(top_only=False)
        assert isinstance(result, str)
        assert "|" in result

        # it would be too fragile to compare more specific than this
        assert len(result.split("|")) > 1
        assert all(len(x.split(":")) == 2 for x in result.split("|"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
