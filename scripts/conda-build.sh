#! /usr/bin/env bash

# mamba create -n cupynumeric_build python=$PYTHON_VERSION boa git

cd $(dirname "$(realpath "$0")")/..

mkdir -p /tmp/conda-build/cupynumeric
rm -rf /tmp/conda-build/cupynumeric/*

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

CUDA="$(nvcc --version | head -n4 | tail -n1 | cut -d' ' -f5 | cut -d',' -f1).*" \
conda mambabuild \
    --numpy 1.22 \
    --override-channels \
    -c conda-forge -c https://github.com/nv-legate/ucx-package/raw/main \
    -c file:///tmp/conda-build/legate_core \
    --croot /tmp/conda-build/cupynumeric \
    --no-test \
    --no-verify \
    --no-build-id \
    --build-id-pat='' \
    --merge-build-host \
    --no-include-recipe \
    --no-anaconda-upload \
    --variants "{gpu_enabled: 'true', python: $PYTHON_VERSION}" \
    ./conda/conda-build
