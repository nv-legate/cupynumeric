<!--
Copyright 2024 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->

[![Build Nightly release package](https://github.com/nv-legate/cupynumeric.internal/actions/workflows/ci-gh-nightly-release.yml/badge.svg)](https://github.com/nv-legate/cupynumeric.internal/actions/workflows/ci-gh-nightly-release.yml)

# cuPyNumeric

cuPyNumeric is a library that aims to provide a distributed and accelerated
drop-in replacement for [NumPy](https://numpy.org/) built on top of the
[Legate](https://github.com/nv-legate/legate) framework.

With cuPyNumeric you can write code productively in Python, using the familiar
NumPy API, and have your program scale with no code changes from single-CPU
computers to multi-node-multi-GPU clusters.

For example, you can run
[the final example of the Python CFD course](https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb)
completely unmodified on 2048 A100 GPUs in a
[DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/)
and achieve good weak scaling.

<img src="docs/figures/cfd-demo.png" alt="drawing" width="500"/>

cuPyNumeric works best for programs that have very large arrays of data
that cannot fit in the memory of a single GPU or a single node and need
to span multiple nodes and GPUs. While our implementation of the current
NumPy API is still incomplete, programs that use unimplemented features
will still work (assuming enough memory) by falling back to the
canonical NumPy implementation.

## Installation

Pre-built cuPyNumeric packages are available from
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) on the
[legate channel](https://anaconda.org/legate/cupynumeric) and from
[PyPI](https://pypi.org/project/nvidia-cupynumeric/). See
https://docs.nvidia.com/cupynumeric/latest/installation.html for details about
different install configurations, or building cuPyNumeric from source.

📌 **Note**

Packages are only offered for Linux (x86_64 and aarch64) supporting Python
versions 3.11 to 3.13.

## Documentation

The cuPyNumeric documentation can be found
[here](https://docs.nvidia.com/cupynumeric).

## Contributing

See the discussion on contributing in [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

For technical questions about cuPyNumeric and Legate-based tools, please visit
the [community discussion forum](https://github.com/nv-legate/discussion).

If you have other questions, please contact us at legate(at)nvidia.com.

## Note
*This project, i.e., cuPyNumeric, is separate and independent of the CuPy project. CuPy is a registered trademark of Preferred Networks.*
