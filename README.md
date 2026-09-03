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

# cuPyNumeric

> [!IMPORTANT]
> cuPyNumeric has reached end of life and is no longer maintained or
> supported. The final release is `v26.06.01`. No further releases,
> fixes, or support are planned. Existing packages and documentation remain
> available for historical reference.

cuPyNumeric is a high-performance array computing library that implements the
NumPy API on top of the Legate framework. It enables you to run existing NumPy
workflows on GPUs and distributed systems with little to no code changes.

Whether your work involves large-scale data analysis, complex simulations, or
machine learning, cuPyNumeric allows you to seamlessly scale from a single CPU,
to a single GPU, and up to thousands of GPUs across multiple nodes.

## Installation

Existing pre-built cuPyNumeric packages remain available from
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) on the
[legate channel](https://anaconda.org/legate/cupynumeric) and from
[PyPI](https://pypi.org/project/nvidia-cupynumeric/). See
https://docs.nvidia.com/cupynumeric/26.06/installation.html for details about
different install configurations, or building cuPyNumeric from source.

📌 **Note**

Linux packages support Python versions 3.11 to 3.14. Windows is only supported
through WSL.

## Documentation

The cuPyNumeric documentation can be found
[here](https://docs.nvidia.com/cupynumeric/26.06/).

## Note

The cuPyNumeric project is independent of the CuPy project. CuPy is a trademark
of Preferred Networks, Inc, and the name 'cuPyNumeric' is used with their
permission.
