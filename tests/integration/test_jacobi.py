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
from utils.comparisons import allclose

import cupynumeric as num


def test_basic():
    height = 10
    width = 10
    grid = num.zeros((height + 2, width + 2), np.float32)
    grid[:, 0] = -273.15
    grid[:, -1] = -273.15
    grid[-1, :] = -273.15
    grid[0, :] = 40.0
    center = grid[1:-1, 1:-1]
    north = grid[0:-2, 1:-1]
    east = grid[1:-1, 2:]
    west = grid[1:-1, 0:-2]
    south = grid[2:, 1:-1]
    for i in range(2):
        average = center + north + east + west + south
        work = 0.2 * average
        delta = num.sum(num.absolute(work - center))
        center[:] = work
    npGrid = np.zeros((height + 2, width + 2), np.float32)
    npGrid[:, 0] = -273.15
    npGrid[:, -1] = -273.15
    npGrid[-1, :] = -273.15
    npGrid[0, :] = 40.0
    npcenter = npGrid[1:-1, 1:-1]
    npnorth = npGrid[0:-2, 1:-1]
    npeast = npGrid[1:-1, 2:]
    npwest = npGrid[1:-1, 0:-2]
    npsouth = npGrid[2:, 1:-1]
    for i in range(2):
        npaverage = npcenter + npnorth + npeast + npwest + npsouth
        npwork = 0.2 * npaverage
        nptemp = np.absolute(npwork - npcenter)
        npdelta = np.sum(nptemp)
        npcenter[:] = npwork
    assert allclose(delta, npdelta)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
