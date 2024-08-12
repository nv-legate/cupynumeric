<!--
Copyright 2021-2022 NVIDIA Corporation

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

# cuNumeric

cuNumeric is a [Legate](https://github.com/nv-legate/legate.core) library
that aims to provide a distributed and accelerated drop-in replacement for the
[NumPy API](https://numpy.org/doc/stable/reference/) on top of the
[Legion](https://legion.stanford.edu) runtime. Using cuNumeric you do things like run
[the final example of the Python CFD course](https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb)
completely unmodified on 2048 A100 GPUs in a [DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/) and achieve good weak scaling.

<img src="docs/figures/cfd-demo.png" alt="drawing" width="500"/>

cuNumeric works best for programs that have very large arrays of data
that cannot fit in the memory of a single GPU or a single node and need
to span multiple nodes and GPUs. While our implementation of the current
NumPy API is still incomplete, programs that use unimplemented features
will still work (assuming enough memory) by falling back to the
canonical NumPy implementation.

If you have questions, please contact us at legate(at)nvidia.com.

## Installation

cuNumeric is available from [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
on the [legate channel](https://anaconda.org/legate/cunumeric).
Please make sure you have at least conda version 24.1 installed, then create
a new environment containing cuNumeric:

```
conda create -n myenv -c conda-forge -c legate cunumeric
```

or install it into an existing environment:

```
conda install -c conda-forge -c legate cunumeric
```

In an environment without GPUs available, `conda install` will by default choose a CPU-only package.
To install a version with GPU support in such an environment, use environment variable `CONDA_OVERRIDE_CUDA`.

```shell
CONDA_OVERRIDE_CUDA="12.2" \
  conda install -c conda-forge -c legate legate-core
```

Once installed, you can verify the installation by running one of the examples
from the cuNumeric repository, for instance:

```
$ legate examples/black_scholes.py
Running black scholes on 10K options...
Elapsed Time: 129.017 ms
```

Only linux-64 packages are available at the moment.

The default package contains GPU support, and is compatible with CUDA >= 11.8
(driver >= 520), and Volta or later GPU architectures. There are also CPU-only
packages available, which will be automatically selected when installing on a
machine without GPUs available. See https://nv-legate.github.io/cunumeric for
details about manually forcing different install configurations, or building 
cuNumeric from source.

## Usage and Execution

Using cuNumeric as a replacement for NumPy is easy. Users only need
to replace:

```
import numpy as np
```

with:

```
import cunumeric as np
```

These programs can then be run by the Legate driver script described in the
[Legate Core](https://github.com/nv-legate/legate.core) documentation.

```
legate cunumeric_program.py
```

For execution with multiple nodes (assuming Legate Core is installed with networking support)
users can supply the `--nodes` option. For execution with GPUs, users can use the
`--gpus` flags to specify the number of GPUs to use per node. We encourage all users
to familiarize themselves with these resource flags as described in the Legate Core
documentation or simply by passing `--help` to the `legate` driver script.

You can use `test.py` to run the test suite. Invoke the script directly or through
standard `python`; the script will invoke the `legate` driver script internally.
Check out `test.py --help` for further options.

## Supported and Planned Features

cuNumeric is currently a work in progress and we are gradually adding support for
additional NumPy operators. Unsupported NumPy operations will provide a
warning that we are falling back to canonical NumPy. Please report unimplemented
features that are necessary for attaining good performance so that we can triage
them and prioritize implementation appropriately. The more users that report an
unimplemented feature, the more we will prioritize it. Please include a pointer
to your code if possible too so we can see how you are using the feature in context.

## Supported Types and Dimensions

cuNumeric currently supports the following NumPy types: `float16`, `float32`,
`float64`, `int16`, `int32`, `int64`, `uint16`, `uint32`, `uint64`, `bool`,
`complex64`, and `complex128`.

cuNumeric supports up to 4D arrays by default, you can adjust this setting by
installing legate.core with a larger `--max-dim`.

## Documentation

The cuNumeric documentation can be found
[here](https://nv-legate.github.io/cunumeric).

## Future Directions

There are three primary directions that we plan to investigate
with cuNumeric going forward:

* More features: we plan to identify a few key lighthouse applications
  and use the demands of these applications to drive the addition of
  new features to cuNumeric.
* We plan to add support for sharded file I/O for loading and
  storing large data sets that could never be loaded on a single node.
  Initially this will begin with native support for hdf5 and zarr,
  but will grow to accommodate other formats needed by our lighthouse
  applications.
* Strong scaling: while cuNumeric is currently implemented in a way that
  enables weak scaling of codes on larger data sets, we would also like
  to make it possible to strong-scale Legate applications for a single
  problem size. This will require leveraging some of the more advanced
  features of Legion from inside the Python interpreter.

We are open to comments, suggestions, and ideas.

## Contributing

See the discussion of contributing in [CONTRIBUTING.md](CONTRIBUTING.md).

## Known Issues

 * When using certain operations with high scratch space requirements (e.g.
   `einsum` or `convolve`) you might run into the following error:
   ```
   LEGION ERROR: Failed to allocate DeferredBuffer/Value/Reduction in task [some task] because [some memory] is full. This is an eager allocation ...
   ```
   Currently, Legion splits its memory reservations between two pools: the
   "deferred" pool, used for allocating cuNumeric `ndarray`s, and the "eager"
   pool, used for allocating scratch memory for operations. The above error
   message signifies that not enough memory was available for an operation's
   scratch space requirements. You can work around this by allocating more
   memory overall to cuNumeric (e.g. adjusting `--sysmem`, `--numamem` or
   `--fbmem`), and/or by adjusting the split between the two pools (e.g. by
   passing `-lg:eager_alloc_percentage 60` on the command line to allocate 60%
   of memory to the eager pool, up from the default of 50%).
 * cuNumeric can exercise a bug in OpenBLAS when it is run with
   [multiple OpenMP processors](https://github.com/xianyi/OpenBLAS/issues/2146)
 * On Mac OSX, cuNumeric can trigger a bug in Apple's implementation of libc++.
   The [bug](https://bugs.llvm.org/show_bug.cgi?id=43764) has since been fixed but
   likely will not show up on most Apple machines for quite some time. You may have
   to manually patch your implementation of libc++. If you have trouble doing this
   please contact us and we will be able to help you.
