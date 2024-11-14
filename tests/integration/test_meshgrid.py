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
from utils.generators import mk_0to1_array, mk_seq_array

import cupynumeric as num


@pytest.mark.parametrize(
    "indexing", ("xy", "ij"), ids=lambda indexing: f"{indexing=}"
)
@pytest.mark.parametrize(
    "sparse", (True, False), ids=lambda sparse: f"{sparse=}"
)
@pytest.mark.parametrize("ndim", range(4), ids=lambda ndim: f"{ndim=}")
def test_meshgrid_ints(ndim, sparse, indexing):
    xi = tuple(mk_seq_array(np, (10,)) for i in range(ndim))
    xin = tuple(mk_seq_array(num, (10,)) for i in range(ndim))

    out = np.meshgrid(*xi, sparse=sparse, indexing=indexing)
    outn = num.meshgrid(*xin, sparse=sparse, indexing=indexing)

    assert len(out) == len(outn) == ndim
    assert all(isinstance(xvn, num.ndarray) for xvn in outn)
    for xv, xvn in zip(out, outn):
        assert np.array_equal(xv, xvn)


@pytest.mark.parametrize(
    "indexing", ("xy", "ij"), ids=lambda indexing: f"{indexing=}"
)
@pytest.mark.parametrize(
    "sparse", (True, False), ids=lambda sparse: f"{sparse=}"
)
@pytest.mark.parametrize("ndim", range(4), ids=lambda ndim: f"{ndim=}")
def test_meshgrid_floats(ndim, sparse, indexing):
    xi = tuple(mk_0to1_array(np, (10,)) for i in range(ndim))
    xin = tuple(mk_0to1_array(num, (10,)) for i in range(ndim))

    out = np.meshgrid(*xi, sparse=sparse, indexing=indexing)
    outn = num.meshgrid(*xin, sparse=sparse, indexing=indexing)

    assert len(out) == len(outn) == ndim
    assert all(isinstance(xvn, num.ndarray) for xvn in outn)
    for xv, xvn in zip(out, outn):
        assert np.array_equal(xv, xvn)


def test_bad_indexing():
    with pytest.raises(ValueError):
        num.meshgrid(num.array([1, 2, 3]), indexing="abc")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
