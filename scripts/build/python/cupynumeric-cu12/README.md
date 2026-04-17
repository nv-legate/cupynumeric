# Python PyPi Binary Wheels

This directory is the root of the Python PyPi binary pip wheels for the
project. The wheels are built on Rocky Linux containers using a recent
version of the CUDA Toolkit. The simplest way to build the wheels locally
if desired is to replicate that environment using Docker.

You would want to clone the source directory, change into that directory and
then run something like the following to mount the source within the container:
```
docker run --rm --runtime=nvidia --gpus all -it --mount type=bind,src=.,dst=/src rapidsai/ci-wheel:cuda12.8.0-rockylinux8-py3.12 bash
cd /src
export PATH=/src/continuous_integration/scripts/tools/:$PATH
dnf install -y gcc-toolset-11-libatomic-devel
```

It is necessary to provide a compatible binary wheel for `legate` to build against. To use
the build scripts place that wheel in the `wheel` subdirectory of the source tree:
```
mkdir wheel
cp /path/to/legate.whl wheel/
```

At this point you are ready to build the wheels and have all the necessary
extra packages installed to run the script used by CI.
```
./continuous_integration/scripts/build_wheel_linux.bash
```

If everything went as planned you will have a binary wheel in the `final-dist`
directory. This is the same as the wheel produced by CI, and can be installed
with:
```
pip install final-dist/*.whl
```

## Build Steps

You can go through the bash script and run individual pieces of the build,
there are a number of steps at the current time. At a high level:

 * Setting up `sccache` if running in CI
 * Installing extra Rocky Linux 8 packages
 * Installing the required pip packages for the build
 * Adding symlinks and CMake to enable finding libraries
 * Building `openblas` for bundling among other depedencies
 * Building the `nvidia-cupynumeric` binary wheel
 * Repairing the binary wheel to ensure manylinux compliance
