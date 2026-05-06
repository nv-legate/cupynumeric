#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --output=legate_job_%j.out
#SBATCH --error=legate_job_%j.err
#SBATCH --job-name=gemm_2n_4gpu
#SBATCH --mem=32G

exec 2>&1
set -eou pipefail

# shellcheck disable=SC2154
echo "Starting job ${SLURM_JOB_ID} on the ${SLURM_NODELIST} node[s]"

if [[ $# -ge 2 ]]; then
    # source entrypoint
    # shellcheck disable=SC1090
    source "$1"
    shift

    # source test
    # shellcheck disable=SC1090
    source "$1"
    shift

    # run only prepare part
    test_cupynumeric sbatch "$@"
else
    echo "HOME: ${HOME}"
fi

# shellcheck disable=SC2154
echo "Conda environment activated: ${CONDA_PREFIX}"

# === TESTS ===

# Set unlimited locked memory for UCX/GPU operations
ulimit -l unlimited

(ldconfig -p | grep libcuda) || true
LEGATE_SHOW_CONFIG=1 legate-issue

# Resource configuration will be passed directly on the command line
# With 2 ranks per node, each rank gets: 1 GPU, 4 CPUs, 1 OMP thread, 2 utility threads (8 cores total)

# On Linux, Open MPI is built with CUDA awareness but it is disabled by default.
# To enable it, please set the environment variable
export OMPI_MCA_opal_cuda_support=true

# Use UCX for communication (modern high-performance transport)
export OMPI_MCA_pml="ucx"
export OMPI_MCA_osc="ucx"

# Disable older OpenFabrics/openib BTL to avoid warnings (we're using UCX instead)
export OMPI_MCA_btl="^openib"

# UCX configuration for optimal performance
export UCX_MEMTYPE_CACHE=n  # Required for CUDA awareness
# export UCX_TLS=rc,sm,cuda_copy,cuda_ipc  # InfiniBand (rc), shared memory (sm), CUDA
export UCX_TLS=tcp,sm,cuda_copy,cuda_ipc  # Ethernet (tcp), shared memory (sm), CUDA

# Fix UCX IB_SEG_SIZE warning - set to valid value or unset since we're using TCP
echo "UCX_IB_SEG_SIZE: ${UCX_IB_SEG_SIZE:-Not set}"
unset UCX_IB_SEG_SIZE  # Remove invalid IB segment size configuration
export UCX_WARN_UNUSED_ENV_VARS=n  # Suppress warnings about unused UCX environment variables
# before launching your MPI processes.
# Equivalently, you can set the MCA parameter in the command line:
# mpiexec --mca opal_cuda_support 1 ...
# Note that you might also need to set UCX_MEMTYPE_CACHE=n for CUDA awareness via
# UCX. Please consult UCX documentation for further details.

# Show environment variables
echo "Environment variables:"
# shellcheck disable=SC2312
env | grep -E '(CUDA|SLURM|UCX|MPI)' | sort
echo ""

# ============== cuPyNumeric GEMM Example ==============
echo ""
echo "=========================================="
echo "Running cuPyNumeric GEMM Example"
echo "=========================================="

GEMM_SCRIPT=examples/gemm.py

# Check SLURM GPU allocation
cat <<EOF
0. SLURM GPU Allocation:
# FIXME: compare those vars with another branch!!!
SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-Not set}
SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-Not set}
SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-Not set}
SLURM_GPUS_PER_TASK: ${SLURM_GPUS_PER_TASK:-Not set}
SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-Not set}
SLURM_NTASKS_PER_NODE: ${SLURM_NTASKS_PER_NODE:-Not set}
EOF

GEMM_SIZE=7000
GEMM_ITERATIONS=3

# shellcheck disable=SC2154
OPTS+=(
    --launcher mpirun
    --nodes "${SLURM_JOB_NUM_NODES}"
    --ranks-per-node "${SLURM_NTASKS_PER_NODE}"
    --cpus 4
    --gpus "${SLURM_GPUS_ON_NODE}"
    --omps 1
    --utility 2
    --sysmem 32000
    --fbmem 16000
    # --verbose
)

cat << EOF
LEGATE_CONFIG: ${OPTS[*]:-auto}

Running cuPyNumeric GEMM with:
  ${SLURM_JOB_NUM_NODES} node[s]
x ${SLURM_NTASKS_PER_NODE} rank/node
x ${SLURM_GPUS_ON_NODE} GPU/rank
 = $((SLURM_JOB_NUM_NODES*SLURM_NTASKS_PER_NODE*SLURM_GPUS_ON_NODE)) total GPUs

EOF

LOG_FILE="legate_gemm_${SLURM_JOB_ID}.log"
EXIT_CODE=0

(legate "${OPTS[@]}" \
    python3 "${GEMM_SCRIPT}" \
    -n $((GEMM_SIZE * SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE * SLURM_GPUS_ON_NODE)) \
    -i "${GEMM_ITERATIONS}" 2>&1 | \
    tee "${LOG_FILE}") || EXIT_CODE=$?

echo "EXIT_CODE: ${EXIT_CODE}"

if grep -q "RuntimeError" "${LOG_FILE}"; then
    echo "RuntimeError found in logs. Please check the logs for details."
    EXIT_CODE=1
elif grep -q "ModuleNotFoundError" "${LOG_FILE}"; then
    echo "ModuleNotFoundError found in logs. Please check the logs for details."
    EXIT_CODE=1
elif grep -q "Segmentation fault" "${LOG_FILE}"; then
    echo "Segmentation fault found in logs. Please check the logs for details."
    EXIT_CODE=1
elif grep -q "undefined symbol" "${LOG_FILE}"; then
    echo "undefined symbol found in logs. Please check the logs for details."
    EXIT_CODE=1
elif grep -q "ImportError" "${LOG_FILE}"; then
    echo "ImportError found in logs. Please check the logs for details."
    EXIT_CODE=1
fi

echo "Done running all GEMM tests"
echo "Logs saved to: ${LOG_FILE}"

exit "${EXIT_CODE}"
