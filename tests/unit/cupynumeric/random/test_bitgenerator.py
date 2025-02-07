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

import pytest
from mock import patch

import cupynumeric.random._bitgenerator as m  # module under test
from cupynumeric.config import BitGeneratorType


class TestXORWOW:
    @patch("time.perf_counter_ns")
    def test_seed_None(self, mock_perf_counter_ns) -> None:
        mock_perf_counter_ns.return_value = 5
        bg = m.XORWOW()
        mock_perf_counter_ns.assert_called_once()
        assert bg.generatorType == BitGeneratorType.XORWOW
        assert bg.flags == 0
        assert bg.seed == 5
        assert bg.handle is not None

    @patch("time.perf_counter_ns")
    def test_seed(self, mock_perf_counter_ns) -> None:
        bg = m.XORWOW(seed=10)
        mock_perf_counter_ns.assert_not_called()
        assert bg.generatorType == BitGeneratorType.XORWOW
        assert bg.flags == 0
        assert bg.seed == 10
        assert bg.handle is not None


class TestMRG32k3a:
    @patch("time.perf_counter_ns")
    def test_seed_None(self, mock_perf_counter_ns) -> None:
        mock_perf_counter_ns.return_value = 5
        bg = m.MRG32k3a()
        mock_perf_counter_ns.assert_called_once()
        assert bg.generatorType == BitGeneratorType.MRG32K3A
        assert bg.flags == 0
        assert bg.seed == 5
        assert bg.handle is not None

    @patch("time.perf_counter_ns")
    def test_seed(self, mock_perf_counter_ns) -> None:
        bg = m.MRG32k3a(seed=10)
        mock_perf_counter_ns.assert_not_called()
        assert bg.generatorType == BitGeneratorType.MRG32K3A
        assert bg.flags == 0
        assert bg.seed == 10
        assert bg.handle is not None


class TestPHILOX4_32_10:
    @patch("time.perf_counter_ns")
    def test_seed_None(self, mock_perf_counter_ns) -> None:
        mock_perf_counter_ns.return_value = 5
        bg = m.PHILOX4_32_10()
        mock_perf_counter_ns.assert_called_once()
        assert bg.generatorType == BitGeneratorType.PHILOX4_32_10
        assert bg.flags == 0
        assert bg.seed == 5
        assert bg.handle is not None

    @patch("time.perf_counter_ns")
    def test_seed(self, mock_perf_counter_ns) -> None:
        bg = m.PHILOX4_32_10(seed=10)
        mock_perf_counter_ns.assert_not_called()
        assert bg.generatorType == BitGeneratorType.PHILOX4_32_10
        assert bg.flags == 0
        assert bg.seed == 10
        assert bg.handle is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
