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

float_type = "float32"

# A simplified implementation of Richardson-Lucy deconvolution


def run_richardson_lucy(
    shape, filter_shape, num_iter, warmup, conv_method, *, print_timing=False
):
    image = np.random.rand(*shape).astype(float_type)
    psf = np.random.rand(*filter_shape).astype(float_type)
    im_deconv = np.full(image.shape, 0.5, dtype=float_type)
    psf_mirror = np.flip(psf)

    timer.start()

    for idx in range(num_iter + warmup):
        if idx == warmup:
            timer.start()
        conv = np.convolve(im_deconv, psf, mode="same", method=conv_method)
        relative_blur = image / conv
        im_deconv *= np.convolve(
            relative_blur, psf_mirror, mode="same", method=conv_method
        )

    total = timer.stop()
    if print_timing:
        print("Elapsed Time: " + str(total) + " ms")
    return total


if __name__ == "__main__":

    def tuple_of_ints(arg):
        return tuple(map(int, arg.split(",")))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=10,
        dest="I",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=1,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "--shape",
        type=tuple_of_ints,
        nargs="+",
        default=[(20, 20, 20)],
        dest="shape",
        help="number of elements in X,Y,Z dimensions (default '20,20,20')",
    )
    parser.add_argument(
        "--filter-shape",
        type=tuple_of_ints,
        nargs="+",
        default=[(4, 4, 4)],
        dest="filter_shape",
        help="number of filter weights in X,Y,Z dimendins (default '4,4,4')",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "--conv-method",
        dest="conv_method",
        type=str,
        default="auto",
        help="convolution method (auto by default)",
    )

    args, np, timer = parse_args(parser)

    run_benchmark(
        run_richardson_lucy,
        args.benchmark,
        "Richardson Lucy",
        [
            ("shape", args.shape),
            ("filter shape", args.filter_shape),
            ("iterations", args.I),
            ("warmup iterations", args.warmup),
            ("convolution method", args.conv_method),
        ],
        ["time (milliseconds)"],
        print_timing=args.timing,
    )
