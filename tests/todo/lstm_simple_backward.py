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


def testtion():
    word_size = 10
    hidden_size = 10
    sentence_length = 5
    batch_size = 3
    num.random.seed(42)

    WLSTM = num.random.randn(
        word_size + hidden_size, 4 * hidden_size
    ) / num.sqrt(word_size + hidden_size)

    xphpb = WLSTM.shape[0]
    d = hidden_size
    n = sentence_length
    b = batch_size

    dHout = num.random.randn(n, b, d)
    IFOGf = num.random.randn(n, b, d * 4)
    C = num.random.randn(n, b, d)
    Ct = num.random.randn(n, b, d)
    Hin = num.random.randn(n, b, xphpb)

    dIFOG = num.zeros((n, b, d * 4))
    dIFOGf = num.zeros(IFOGf.shape)
    dHin = num.zeros(Hin.shape)
    dC = num.zeros(C.shape)
    dh0 = num.zeros((1, d))

    for t in reversed(range(n)):
        tanhCt = Ct[t]
        dIFOGf[t, :, 2 * d : 3 * d] = tanhCt * dHout[t]
        # backprop tanh non-linearity first then continue backprop
        dC[t] += (1 - tanhCt**2) * (IFOGf[t, :, 2 * d : 3 * d] * dHout[t])

        if t > 0:
            dIFOGf[t, :, d : 2 * d] = C[t - 1] * dC[t]
            dC[t - 1] += IFOGf[t, :, d : 2 * d] * dC[t]

        dIFOGf[t, :, :d] = IFOGf[t, :, 3 * d :] * dC[t]
        dIFOGf[t, :, 3 * d :] = IFOGf[t, :, :d] * dC[t]

        # backprop activation functions
        dIFOG[t, :, 3 * d :] = (1 - IFOGf[t, :, 3 * d :] ** 2) * dIFOGf[
            t, :, 3 * d :
        ]
        y = IFOGf[t, :, : 3 * d]
        dIFOG[t, :, : 3 * d] = (y * (1.0 - y)) * dIFOGf[t, :, : 3 * d]

        # backprop matrix multiply
        dHin[t] = dIFOG[t].dot(WLSTM.transpose())

        # backprop the identity transforms into Hin
        if t > 0:
            dHout[t - 1, :] += dHin[t, :, word_size:]
        else:
            dh0[0] += num.sum(dHin[t, :, word_size:], 0)

    np.random.seed(42)

    WLSTM = np.random.randn(
        word_size + hidden_size, 4 * hidden_size
    ) / np.sqrt(word_size + hidden_size)

    xphpb = WLSTM.shape[0]
    d = hidden_size
    n = sentence_length
    b = batch_size

    dHout = np.random.randn(n, b, d)
    IFOGf = np.random.randn(n, b, d * 4)
    C = np.random.randn(n, b, d)
    Ct = np.random.randn(n, b, d)
    Hin = np.random.randn(n, b, xphpb)

    dIFOG = np.zeros((n, b, d * 4))
    dIFOGf = np.zeros(IFOGf.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dhnp0 = np.zeros((1, d))

    for t in reversed(range(n)):
        tanhCt = Ct[t]
        dIFOGf[t, :, 2 * d : 3 * d] = tanhCt * dHout[t]
        # backprop tanh non-linearity first then continue backprop
        dC[t] += (1 - tanhCt**2) * (IFOGf[t, :, 2 * d : 3 * d] * dHout[t])

        if t > 0:
            dIFOGf[t, :, d : 2 * d] = C[t - 1] * dC[t]
            dC[t - 1] += IFOGf[t, :, d : 2 * d] * dC[t]

        dIFOGf[t, :, :d] = IFOGf[t, :, 3 * d :] * dC[t]
        dIFOGf[t, :, 3 * d :] = IFOGf[t, :, :d] * dC[t]

        # backprop activation functions
        dIFOG[t, :, 3 * d :] = (1 - IFOGf[t, :, 3 * d :] ** 2) * dIFOGf[
            t, :, 3 * d :
        ]
        y = IFOGf[t, :, : 3 * d]
        dIFOG[t, :, : 3 * d] = (y * (1.0 - y)) * dIFOGf[t, :, : 3 * d]

        # backprop matrix multiply
        dHin[t] = dIFOG[t].dot(WLSTM.transpose())

        # backprop the identity transforms into Hin
        if t > 0:
            dHout[t - 1, :] += dHin[t, :, word_size:]
        else:
            dhnp0[0] += np.sum(dHin[t, :, word_size:], 0)

    assert np.allclose(dh0[0], dhnp0[0])
    # print(dh0[0])
    # print(dhnp0[0])


if __name__ == "__main__":
    testtion()
