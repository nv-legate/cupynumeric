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

import cupynumeric as num

M = 32
alpha = 4.0
beta = -10.0
rate = 0.01


def test():
    x = num.linspace(-4.0, 4.0, M)
    dz = 1.0 + 1j * rate * (12 * x**2 + 2 * alpha)

    xn = np.linspace(-4.0, 4.0, M)
    dzn = 1.0 + 1j * rate * (12 * xn**2 + 2 * alpha)

    assert num.all(num.abs(dz - dzn) < 1e-6)


if __name__ == "__main__":
    test()
