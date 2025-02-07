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


def test():
    anp = np.array(
        [[1 + 2j, 3 + 4j, 5 + 6j], [7 + 8j, 9 + 10j, 11 + 12j]], np.complex
    )
    a = num.array(anp)

    r = a.sum(0)
    assert num.all(num.abs(r - anp.sum(0)) < 1e-5)

    r = a.sum(1)
    assert num.all(num.abs(r - anp.sum(1)) < 1e-5)

    assert num.all(num.abs(a.mean(0) - anp.mean(0)) < 1e-5)
    assert num.all(num.abs(a.mean(1) - anp.mean(1)) < 1e-5)
    assert num.all(num.abs(a.mean() - anp.mean()) < 1e-5)

    return


if __name__ == "__main__":
    test()
