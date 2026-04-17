#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -euo pipefail

export CUDA_MAJOR_VER=${CUDA_MAJOR_VER:=13}

# Install legate first and then cupynumeric.
pip install wheel/*.whl final-dist/*.whl

echo "Configure Legate and run some tests"
export LEGATE_SHOW_CONFIG=1
export LEGATE_CONFIG="--fbmem 1024"
export LEGION_DEFAULT_ARGS="-ll:show_rsrv"

# Attempt to run the tests, we must move cupynumeric to avoid it being used.
mv cupynumeric cupynumeric-moved
pip install cupy-cuda${CUDA_MAJOR_VER}x pytest pynvml psutil scipy

echo "Attempt to import cupynumeric"
python -c 'import cupynumeric as np'
echo "Maybe that worked"

echo "Running the GPU tests"
python test.py --use cuda
echo "Done"
