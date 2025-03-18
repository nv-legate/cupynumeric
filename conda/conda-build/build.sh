#!/bin/bash

echo -e "\n\n--------------------- CONDA/CONDA-BUILD/BUILD.SH -----------------------\n"

set -xeo pipefail;

# Rewrite conda's -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY to
#                 -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH
CMAKE_ARGS="$(echo "$CMAKE_ARGS" | sed -r "s@_INCLUDE=ONLY@_INCLUDE=BOTH@g")"

# Add our options to conda's CMAKE_ARGS
CMAKE_ARGS+="
--log-level=VERBOSE
-DBUILD_SHARED_LIBS=ON
-DBUILD_MARCH=${BUILD_MARCH}
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_VERBOSE_MAKEFILE=ON
-DCMAKE_BUILD_PARALLEL_LEVEL=${JOBS:-$(nproc --ignore=1)}"

# We rely on an environment variable to determine if we need to build cpu-only bits
if [ -z "$CPU_ONLY" ]; then
  # cutensor, relying on the conda cutensor package
  CMAKE_ARGS+="
-Dcutensor_DIR=$PREFIX
-DCMAKE_CUDA_ARCHITECTURES=all-major"
else
  # When we build without cuda, we need to provide the location of curand
  CMAKE_ARGS+="
-Dcupynumeric_cuRAND_INCLUDE_DIR=$PREFIX/targets/x86_64-linux/include"
fi

# We rely on an environment variable to determine if we need to build cpp tests
if [[ "$BUILD_TESTS" == "1" ]]; then
  CMAKE_ARGS+="
-Dcupynumeric_BUILD_TESTS=ON"
fi

export CMAKE_GENERATOR=Ninja
export CUDAHOSTCXX=${CXX}
export OPENSSL_DIR="$PREFIX"

echo "Environment"
env

echo "Build starting on $(date)"
CUDAFLAGS="-isystem ${PREFIX}/include -L${PREFIX}/lib"
export CUDAFLAGS

cmake -S . -B build ${CMAKE_ARGS} -DCMAKE_BUILD_PARALLEL_LEVEL=$CPU_COUNT
cmake --build build -j$CPU_COUNT --verbose
cmake --install build

CMAKE_ARGS="
-DFIND_CUPYNUMERIC_CPP=ON
-Dcupynumeric_ROOT=$PREFIX"

SKBUILD_BUILD_OPTIONS=-j$CPU_COUNT \
$PYTHON -m pip install             \
  --root /                         \
  --no-deps                        \
  --prefix "$PREFIX"               \
  --no-build-isolation             \
  --upgrade                        \
  --cache-dir "$PIP_CACHE_DIR"     \
  --disable-pip-version-check      \
  . -vv

echo "Build ending on $(date)"
