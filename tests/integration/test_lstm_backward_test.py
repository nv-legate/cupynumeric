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
    word_size = 10
    hidden_size = 10
    sentence_length = 5
    batch_size = 3

    WLSTM_np = np.random.randn(
        word_size + hidden_size, 4 * hidden_size
    ) / np.sqrt(word_size + hidden_size)

    xphpb = WLSTM_np.shape[0]
    d = hidden_size
    n = sentence_length
    b = batch_size

    WLSTM_num = num.array(WLSTM_np)

    dHout_np = np.random.randn(n, b, d)
    IFOGf_np = np.random.randn(n, b, d * 4)
    C_np = np.random.randn(n, b, d)
    Ct_np = np.random.randn(n, b, d)
    Hin_np = np.random.randn(n, b, xphpb)

    dIFOG_np = np.zeros((n, b, d * 4))
    dIFOGf_np = np.zeros(IFOGf_np.shape)
    dHin_np = np.zeros(Hin_np.shape)
    dC_np = np.zeros(C_np.shape)
    dh0_np = np.zeros((1, d))

    dHout_num = num.array(dHout_np)
    IFOGf_num = num.array(IFOGf_np)
    C_num = num.array(C_np)
    Ct_num = num.array(Ct_np)
    Hin_num = num.array(Hin_np)

    dIFOG_num = num.zeros((n, b, d * 4))
    dIFOGf_num = num.zeros(IFOGf_num.shape)
    dHin_num = num.zeros(Hin_num.shape)
    dC_num = num.zeros(C_num.shape)
    dh0_num = num.zeros((1, d))

    for t in reversed(range(n)):
        tanhCt_np = Ct_np[t]
        tanhCt_num = Ct_num[t]
        # assert allclose(tanhCt_np, tanhCt_num)

        dIFOGf_np[t, :, 2 * d : 3 * d] = tanhCt_np * dHout_np[t]
        dIFOGf_num[t, :, 2 * d : 3 * d] = tanhCt_num * dHout_num[t]
        # assert allclose(dIFOGf_np[t,:,2*d:3*d], dIFOGf_num[t,:,2*d:3*d])

        # backprop tanh non-linearity first then continue backprop
        dC_np[t] += (1 - tanhCt_np**2) * (
            IFOGf_np[t, :, 2 * d : 3 * d] * dHout_np[t]
        )
        dC_num[t] += (1 - tanhCt_num**2) * (
            IFOGf_num[t, :, 2 * d : 3 * d] * dHout_num[t]
        )
        # assert allclose(dC_np[t], dC_num[t])

        if t > 0:
            dIFOGf_np[t, :, d : 2 * d] = C_np[t - 1] * dC_np[t]
            dIFOGf_num[t, :, d : 2 * d] = C_num[t - 1] * dC_num[t]
            # assert allclose(dIFOGf_np[t,:,d:2*d], dIFOGf_num[t,:,d:2*d])

            dC_np[t - 1] += IFOGf_np[t, :, d : 2 * d] * dC_np[t]
            dC_num[t - 1] += IFOGf_num[t, :, d : 2 * d] * dC_num[t]
            # assert allclose(dC_np[t-1], dC_num[t-1])

        dIFOGf_np[t, :, :d] = IFOGf_np[t, :, 3 * d :] * dC_np[t]
        dIFOGf_num[t, :, :d] = IFOGf_num[t, :, 3 * d :] * dC_num[t]
        # assert allclose(dIFOGf_np[t,:,:d], dIFOGf_num[t,:,:d])

        dIFOGf_np[t, :, 3 * d :] = IFOGf_np[t, :, :d] * dC_np[t]
        dIFOGf_num[t, :, 3 * d :] = IFOGf_num[t, :, :d] * dC_num[t]
        # assert allclose(dIFOGf_np, dIFOGf_num)

        # backprop activation functions
        dIFOG_np[t, :, 3 * d :] = (
            1 - IFOGf_np[t, :, 3 * d :] ** 2
        ) * dIFOGf_np[t, :, 3 * d :]
        dIFOG_num[t, :, 3 * d :] = (
            1 - IFOGf_num[t, :, 3 * d :] ** 2
        ) * dIFOGf_num[t, :, 3 * d :]
        # assert allclose(dIFOG_np[t,:,3*d:], dIFOG_num[t,:,3*d:])

        y_np = IFOGf_np[t, :, : 3 * d]
        y_num = IFOGf_num[t, :, : 3 * d]
        # assert allclose(y_np, y_num)

        dIFOG_np[t, :, : 3 * d] = (y_np * (1.0 - y_np)) * dIFOGf_np[
            t, :, : 3 * d
        ]
        dIFOG_num[t, :, : 3 * d] = (y_num * (1.0 - y_num)) * dIFOGf_num[
            t, :, : 3 * d
        ]
        # assert allclose(dIFOG_np[t,:,:3*d], dIFOG_num[t,:,:3*d])

        # backprop matrix multiply
        dHin_np[t] = dIFOG_np[t].dot(WLSTM_np.transpose())
        dHin_num[t] = dIFOG_num[t].dot(WLSTM_num.transpose())
        # assert allclose(dHin_np[t], dHin_num[t])

        # backprop the identity transforms into Hin
        if t > 0:
            dHout_np[t - 1, :] += dHin_np[t, :, word_size:]
            dHout_num[t - 1, :] += dHin_num[t, :, word_size:]
            # assert allclose(dHout_np[t-1,:], dHout_num[t-1,:])
        else:
            dh0_np[0] += np.sum(dHin_np[t, :, word_size:], 0)
            dh0_num[0] += num.sum(dHin_num[t, :, word_size:], 0)
            # Check this one at the end
    # print(dh0_np[0])
    # print(dh0_num[0])
    assert allclose(dh0_np[0], dh0_num[0])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
