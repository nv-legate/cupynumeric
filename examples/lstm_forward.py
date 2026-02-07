#!/usr/bin/env python

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

import argparse

from benchmark import parse_args, run_benchmark


def run_lstm(
    batch_size, hidden_size, sentence_length, word_size, *, print_timing=False
):
    timer.start()

    X = np.random.randn(sentence_length, batch_size, hidden_size)
    h0 = np.random.randn(1, hidden_size)
    WLSTM = np.random.randn(
        word_size + hidden_size, 4 * hidden_size
    ) / np.sqrt(word_size + hidden_size)

    xphpb = WLSTM.shape[0]
    d = hidden_size
    n = sentence_length
    b = batch_size

    Hin = np.zeros((n, b, xphpb))
    Hout = np.zeros((n, b, d))
    IFOG = np.zeros((n, b, d * 4))
    IFOGf = np.zeros((n, b, d * 4))
    C = np.zeros((n, b, d))
    Ct = np.zeros((n, b, d))

    for t in range(0, n):
        if t == 0:
            prev = np.tile(h0, (b, 1))
        else:
            prev = Hout[t - 1]

        Hin[t, :, :word_size] = X[t]
        Hin[t, :, word_size:] = prev
        # compute all gate activations. dots:
        IFOG[t] = Hin[t].dot(WLSTM)
        # non-linearities
        IFOGf[t, :, : 3 * d] = 1.0 / (
            1.0 + np.exp(-IFOG[t, :, : 3 * d])
        )  # sigmoids these are the gates
        IFOGf[t, :, 3 * d :] = np.tanh(IFOG[t, :, 3 * d :])  # tanh
        # compute the cell activation
        C[t] = IFOGf[t, :, :d] * IFOGf[t, :, 3 * d :]
        if t > 0:
            C[t] += IFOGf[t, :, d : 2 * d] * C[t - 1]
        Ct[t] = np.tanh(C[t])
        Hout[t] = IFOGf[t, :, 2 * d : 3 * d] * Ct[t]

    total = timer.stop()
    if print_timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-B",
        "--batch",
        type=int,
        nargs="+",
        default=[32],
        dest="batch",
        help="batch size",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[10],
        dest="hidden",
        help="hidden size",
    )
    parser.add_argument(
        "-s",
        "--sentence",
        type=int,
        nargs="+",
        default=[4],
        dest="sentence",
        help="sentence length",
    )
    parser.add_argument(
        "-w",
        "--word",
        type=int,
        nargs="+",
        default=[10],
        dest="word",
        help="word size",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )

    args, np, timer = parse_args(parser)

    run_benchmark(
        run_lstm,
        args.benchmark,
        "LSTM Forward",
        [
            ("batch size", args.batch),
            ("hidden size", args.hidden),
            ("sentence length", args.sentence),
            ("word size", args.word),
        ],
        ["time (milliseconds)"],
        print_timing=args.timing,
    )
