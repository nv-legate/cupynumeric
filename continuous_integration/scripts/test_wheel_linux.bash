#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -euo pipefail

echo "Are my wheels there???"

ls -lh

ls -lh wheel
ls -lh final-dist

# Install legate first and then cupynumeric.
pip install wheel/*.whl final-dist/*.whl

echo "Let's explore the wheels and see if they are installed correctly."
sitepkgs=$(python -c 'import site; print(site.getsitepackages()[0], end="")')
echo "=== cupynumeric ==="
ls -lh "${sitepkgs}/cupynumeric"
echo "=== legate ==="
ls -lh "${sitepkgs}/legate"

echo "Lamest of proof of life tests for legate"
export LEGATE_SHOW_CONFIG=1
export LEGATE_CONFIG="--fbmem 1024"
export LEGION_DEFAULT_ARGS="-ll:show_rsrv"

# Attempt to run the tests...
mv cupynumeric cupynumeric-moved
pip install pytest pynvml psutil scipy

echo "Attempt to run an example"
legate examples/gemm.py

echo "Example done, attempt to import cupynumeric"
python -c 'import cupynumeric as np'
echo "Maybe that worked"

echo "Running the CPU tests"
python test.py
echo "Done"

echo "Running the GPU tests"
python test.py --use cuda
echo "Done"
