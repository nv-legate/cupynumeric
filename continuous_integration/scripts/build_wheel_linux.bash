#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -euo pipefail

echo "Building a wheel..."

pwd

ls -lah

ls -lh wheel

export PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc --all --ignore=2)}
export CMAKE_BUILD_PARALLEL_LEVEL=${PARALLEL_LEVEL}

if [[ "${CI:-false}" == "true" ]]; then
  echo "Installing extra system packages"
  dnf install -y gcc-toolset-11-libatomic-devel
    # Enable gcc-toolset-11 environment
  source /opt/rh/gcc-toolset-11/enable
  # Verify compiler version
  gcc --version
  g++ --version
fi

echo "PATH: ${PATH}"

if [[ "${CUPYNUMERIC_DIR:-}" == "" ]]; then
  # If we are running in an action then GITHUB_WORKSPACE is set.
  if [[ "${GITHUB_WORKSPACE:-}" == "" ]]; then
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    CUPYNUMERIC_DIR="$(realpath "${script_dir}"/../../)"
  else
    # Simple path witin GitHub actions workflows.
    CUPYNUMERIC_DIR="${GITHUB_WORKSPACE}"
  fi
  export CUPYNUMERIC_DIR
fi
package_name="cupynumeric"
package_dir="${CUPYNUMERIC_DIR}/scripts/build/python/cupynumeric"

# This is all very hackish and needs to be fixed up.
echo "Installing build requirements"
python -m pip install -v --prefer-binary -r continuous_integration/requirements-build.txt

# Install the legate wheel that was downloaded.
pip install wheel/*.whl

sitepkgs=$(python -c 'import site; print(site.getsitepackages()[0], end="")')
# Add in the symbolic links for cuTensor so that CMake can find it (hack)
ln -fs "${sitepkgs}"/cutensor/lib/libcutensor.so.2 "${sitepkgs}"/cutensor/lib/libcutensor.so
ln -fs "${sitepkgs}"/cutensor/lib/libcutensorMg.so.2 "${sitepkgs}"/cutensor/lib/libcutensorMg.so

# TODO(cryos): https://github.com/nv-legate/cupynumeric.internal/issues/666
# This is a very hackish way to generate the version for now.
scm_version=$(python -m setuptools_scm -c "${CUPYNUMERIC_DIR}"/scripts/build/python/cupynumeric/pyproject.toml)
export SETUPTOOLS_SCM_PRETEND_VERSION="${scm_version}"
echo "Building wheels with version '${scm_version}'"

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

# The cupynumeric build system defaults to -march=native, which is not going to work
# for packages we want to reuse! Set some reasonable defaults for the wheels.
ARCH=$(uname -m)
echo "Building on architecture: ${ARCH}"
if [[ "$ARCH" == "aarch64" ]]; then
    BUILD_MARCH=armv8-a
else
    BUILD_MARCH=haswell
fi

echo "Building ${package_name}"
# TODO(cryos): https://github.com/nv-legate/legate.internal/issues/1894
# Improve the use of CMAKE_PREFIX_PATH to find legate and cutensor once
# scikit-build supports it.
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${sitepkgs}/legate;${sitepkgs}/cutensor"
export CMAKE_ARGS
SKBUILD_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES:STRING=all-major;-DBUILD_SHARED_LIBS:BOOL=ON;-DBUILD_MARCH=${BUILD_MARCH}"
export SKBUILD_CMAKE_ARGS
echo "SKBUILD_CMAKE_ARGS='${SKBUILD_CMAKE_ARGS}'"

# TODO: Remove this hackish removal of scikit-build files needed as conda
# uses scikit-build and wheels are using scikit-build-core. Migrate conda to
# be consistent with legate and wheels. If not deleted we get inconsistent
# metadata failure during the pip wheel build.
mv "${CUPYNUMERIC_DIR}"/cupynumeric/_version.py "${CUPYNUMERIC_DIR}"/cupynumeric/_version.py.bak
echo "Removed scikit-build _version.py file"
ls -lah

echo "Building wheel..."
cd "${package_dir}"

python -m pip wheel \
  -w "${CUPYNUMERIC_DIR}"/dist \
  -v \
  --no-deps \
  --disable-pip-version-check \
  .

echo "Show dist contents"
pwd
ls -lh "${CUPYNUMERIC_DIR}"/dist

echo "Repairing the wheel"
mkdir -p "${CUPYNUMERIC_DIR}"/final-dist
python -m auditwheel repair \
  --exclude libnvJitLink.so* \
  --exclude libcuda.so* \
  --exclude liblegate.so* \
  --exclude libcublas.so* \
  --exclude libcublasLt.so* \
  --exclude libnccl.so* \
  --exclude libcusparse.so* \
  --exclude libcutensor.so* \
  --exclude libcufft.so* \
  --exclude libcusolver.so* \
  --exclude liblegion-legate.so* \
  --exclude librealm-legate.so* \
  -w "${CUPYNUMERIC_DIR}"/final-dist \
  "${CUPYNUMERIC_DIR}"/dist/*.whl

echo "Wheel has been repaired. Contents:"
ls -lh "${CUPYNUMERIC_DIR}"/final-dist

echo "Restoring scikit-build _verion.py file"
mv "${CUPYNUMERIC_DIR}"/cupynumeric/_version.py.bak "${CUPYNUMERIC_DIR}"/cupynumeric/_version.py
