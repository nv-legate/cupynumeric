#! /usr/bin/env bash

cd $(dirname "$(realpath "$0")")/..

mkdir -p /tmp/conda-build/cupynumeric
rm -rf /tmp/conda-build/cupynumeric/*

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

conda build \
    --override-channels \
    -c conda-forge \
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
