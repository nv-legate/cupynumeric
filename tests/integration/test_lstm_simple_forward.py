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

import cupynumeric as num


def test_basic():
    word_size = 10
    hidden_size = 10
    sentence_length = 2
    batch_size = 3
    X = num.random.randn(sentence_length, batch_size, hidden_size)
    h0 = num.random.randn(1, hidden_size)
    WLSTM = num.random.randn(
        word_size + hidden_size, 4 * hidden_size
    ) / num.sqrt(word_size + hidden_size)

    xphpb = WLSTM.shape[0]
    d = hidden_size
    n = sentence_length
    b = batch_size

    Hin = num.zeros((n, b, xphpb))
    Hout = num.zeros((n, b, d))
    IFOG = num.zeros((n, b, d * 4))
    IFOGf = num.zeros((n, b, d * 4))
    C = num.zeros((n, b, d))
    Ct = num.zeros((n, b, d))

    for t in range(0, n):
        if t == 0:
            prev = num.tile(h0, (b, 1))
        else:
            prev = Hout[t - 1]

        Hin[t, :, :word_size] = X[t]
        Hin[t, :, word_size:] = prev
        # compute all gate activations. dots:
        IFOG[t] = Hin[t].dot(WLSTM)
        # non-linearities
        IFOGf[t, :, : 3 * d] = 1.0 / (
            1.0 + num.exp(-IFOG[t, :, : 3 * d])
        )  # sigmoids these are the gates
        IFOGf[t, :, 3 * d :] = num.tanh(IFOG[t, :, 3 * d :])  # tanh
        # compute the cell activation
        C[t] = IFOGf[t, :, :d] * IFOGf[t, :, 3 * d :]
        if t > 0:
            C[t] += IFOGf[t, :, d : 2 * d] * C[t - 1]
        Ct[t] = num.tanh(C[t])
        Hout[t] = IFOGf[t, :, 2 * d : 3 * d] * Ct[t]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
