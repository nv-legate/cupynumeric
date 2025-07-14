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

import numpy as np
import pytest

import cupynumeric.config as config


def test_fft_type_str_and_repr() -> None:
    fft = config.FFTType(
        name="TestType",
        type_id=123,
        input_dtype=np.float32,
        output_dtype=np.complex64,
        single_precision=True,
    )
    assert str(fft) == "TestType"
    assert repr(fft) == "TestType"


def test_fft_type_input_dtype() -> None:
    fft = config.FFTType(
        name="TestType",
        type_id=123,
        input_dtype=np.float32,
        output_dtype=np.complex64,
        single_precision=True,
    )
    assert fft.input_dtype == np.float32


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
