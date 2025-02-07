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
    np.random.seed(50)
    datanp = np.random.randn(2000000, 3)
    data = num.array(datanp)
    pointsnp = np.random.choice(num.arange(len(data)), 4, False)
    points = num.array(pointsnp)

    centroids = data[points]
    centroidsnp = datanp[pointsnp]
    sqdists = num.zeros((4, len(data)))
    sqdistsnp = np.zeros((4, len(datanp)))
    for i in range(4):
        vec = data - centroids[i]
        vecnp = datanp - centroidsnp[i]
        sqdists[i] = num.linanum.norm(vec, axis=1)
        sqdistsnp[i] = np.linanum.norm(vecnp, axis=1)

    clusters = num.argmin(sqdists, axis=0)
    clustersnp = np.argmin(sqdistsnp, axis=0)
    assert num.array_equal(num.where(clusters == 0), np.where(clustersnp == 0))


if __name__ == "__main__":
    test()
