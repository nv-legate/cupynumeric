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


def assert_partition(a_num, kth, axis):
    # compute volume
    shape = a_num.shape
    volume = np.prod(shape)

    # move axis to end and flatten other
    sort_dim = shape[axis]
    flattened = (int)(volume / sort_dim)
    a_mod = a_num.swapaxes(axis, a_num.ndim - 1)
    a_mod = a_mod.reshape((flattened, sort_dim))

    for segment in range(flattened):
        if kth > 0:
            left_part = a_mod[segment, 0:kth]
            # numpy supports lexicografic complex comparrison
            max_left = left_part.__array__().max()
            if max_left > a_mod[segment, kth]:
                print(a_num)
                assert False
        if kth < sort_dim:
            right_part = a_mod[segment, kth:sort_dim]
            # numpy supports lexicografic complex comparrison
            min_right = right_part.__array__().min()
            if min_right < a_mod[segment, kth]:
                print(a_num)
                assert False


def assert_argpartition(a_num, a_org, kth, axis):
    assert a_num.dtype == np.int64
    assert a_num.ndim == a_org.ndim

    # compute volume
    shape = a_num.shape
    shape_org = a_org.shape
    for i in range(a_num.ndim):
        assert shape[i] == shape_org[i]

    # convert to numpy for 'take_along_axis'
    indexes_np = a_num.__array__()
    a_np = a_org.__array__()
    a_reindexed = np.take_along_axis(a_np, indexes_np, axis=axis)
    a_reindexed_num = num.array(a_reindexed)

    assert_partition(a_reindexed_num, kth, axis)


def check_api(a=None):
    if a is None:
        a = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    a_num = num.array(a)

    shape = a.shape
    volume = np.prod(shape)

    # partition axes
    for i in range(-a.ndim, a.ndim):
        kth = shape[i] // 2
        print(f"partition axis {i}")
        assert_partition(num.partition(a_num, kth=kth, axis=i), kth, i)

    # flatten
    print("partition flattened")
    kth = volume // 2
    assert_partition(num.partition(a_num, kth=kth, axis=None), kth, 0)

    # in-place partition
    kth = shape[a.ndim - 1] // 2
    copy_a_num = a_num.copy()
    copy_a_num.partition(kth)
    assert_partition(copy_a_num, kth, a.ndim - 1)

    # argpartition
    for i in range(-a.ndim, a.ndim):
        kth = shape[i] // 2
        print(f"argpartition axis {i}")
        assert_argpartition(
            num.argpartition(a_num, kth, axis=i), a_num, kth, i
        )

    # flatten
    print("argpartition flattened")
    kth = volume // 2
    assert_argpartition(
        num.argpartition(a_num, kth, axis=None), a_num.flatten(), kth, 0
    )

    # nd.argpartition -- no change to array
    kth = shape[a.ndim - 1] // 2
    copy_a_num = a_num.copy()
    assert_argpartition(copy_a_num.argpartition(kth), a_num, kth, a.ndim - 1)
    assert allclose(copy_a_num, copy_a_num)


def generate_random(shape, datatype):
    print(f"Generate random for {datatype}")
    volume = np.prod(shape)

    if np.issubdtype(datatype, np.integer):
        a_np = np.array(
            np.random.randint(
                np.iinfo(datatype).min, np.iinfo(datatype).max, size=volume
            ),
            dtype=datatype,
        )
    elif np.issubdtype(datatype, np.floating):
        a_np = np.array(np.random.random(size=volume), dtype=datatype)
    elif np.issubdtype(datatype, np.complexfloating):
        a_np = np.array(
            np.random.random(size=volume) + np.random.random(size=volume) * 1j,
            dtype=datatype,
        )
    else:
        print(f"UNKNOWN type {datatype}")
        assert False
    return a_np.reshape(shape)


CASES = [
    ((220,), np.uint32),
    ((2, 5, 7), np.int8),
    ((8, 5), np.float32),
    ((2, 5, 7), np.complex128),
]


@pytest.mark.parametrize("shape, dtype", CASES, ids=str)
def test_dtypes(shape, dtype):
    print("\n\n -----------  dtype test ------------\n")
    check_api(generate_random(shape, dtype))


class TestPartitionErrors:
    def setup_method(self):
        shape = (3, 4, 5)
        volume = np.prod(shape)
        self.a_np = np.array(np.random.random(size=volume)).reshape(shape)
        self.a_num = num.array(self.a_np)

    @pytest.mark.parametrize("axis", (-4, 3))
    def test_axis_out_of_bound(self, axis):
        expected_exc = ValueError
        kth = 1
        with pytest.raises(expected_exc):
            np.partition(self.a_np, kth=kth, axis=axis)
        with pytest.raises(expected_exc):
            num.partition(self.a_num, kth=kth, axis=axis)

    @pytest.mark.xfail
    @pytest.mark.parametrize("kth", (-4, 3, (-4, 0), (0, 3), (3, 3)))
    def test_kth_out_of_bound(self, kth):
        # For all cases,
        # In numpy, it raises ValueError
        # In cupynumeric, it pass
        expected_exc = ValueError
        axis = 0
        with pytest.raises(expected_exc):
            np.partition(self.a_np, kth=kth, axis=axis)
        with pytest.raises(expected_exc):
            num.partition(self.a_num, kth=kth, axis=axis)


class TestArgPartitionErrors:
    def setup_method(self):
        shape = (3, 4, 5)
        volume = np.prod(shape)
        self.a_np = np.array(np.random.random(size=volume)).reshape(shape)
        self.a_num = num.array(self.a_np)

    @pytest.mark.parametrize("axis", (-4, 3))
    def test_axis_out_of_bound(self, axis):
        expected_exc = ValueError
        kth = 1
        with pytest.raises(expected_exc):
            np.argpartition(self.a_np, kth=kth, axis=axis)
        with pytest.raises(expected_exc):
            num.argpartition(self.a_num, kth=kth, axis=axis)

    @pytest.mark.xfail
    @pytest.mark.parametrize("kth", (-4, 3, (-4, 0), (0, 3), (3, 3)))
    def test_kth_out_of_bound(self, kth):
        # For all cases,
        # In numpy, it raises ValueError
        # In cupynumeric, it pass
        expected_exc = ValueError
        axis = 0
        with pytest.raises(expected_exc):
            np.argpartition(self.a_np, kth=kth, axis=axis)
        with pytest.raises(expected_exc):
            num.argpartition(self.a_num, kth=kth, axis=axis)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
