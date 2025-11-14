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

cuPyNumeric is a high-performance array computing library that implements the
NumPy API on top of the Legate framework. It enables you to run existing NumPy
workflows on GPUs and distributed systems with little to no code changes.

Whether your work involves large-scale data analysis, complex simulations, or
machine learning, cuPyNumeric allows you to seamlessly scale from a single CPU,
to a single GPU, and up to thousands of GPUs across multiple nodes.

## Installation

Pre-built cuPyNumeric packages are available from
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) on the
[legate channel](https://anaconda.org/legate/cupynumeric) and from
[PyPI](https://pypi.org/project/nvidia-cupynumeric/). See
https://docs.nvidia.com/cupynumeric/latest/installation.html for details about
different install configurations, or building cuPyNumeric from source.

📌 **Note**

Packages are offered for Linux (x86_64 and aarch64) and macOS (aarch64, pip
wheels only), supporting Python versions 3.11 to 3.13. Windows is only supported
through WSL.

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

The cuPyNumeric project is independent of the CuPy project. CuPy is a trademark
of Preferred Networks, Inc, and the name 'cuPyNumeric' is used with their
permission.
