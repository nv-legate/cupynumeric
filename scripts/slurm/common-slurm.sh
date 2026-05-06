#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Global variable used by the importing scripts
SLURM_JOB_ID=

function wait_for_job_completion {
    local slurm_job_id="$1"
    local log_files
    local tail_pid
    local job_exit_code

    # Get the stdout and stderr log files for the job
    mapfile -t log_files < <(scontrol show job "${slurm_job_id}" | awk -F= '/Std(Err|Out)/ { print $2 }')
    echo "LOG_FILES: ${log_files[*]}"
    tail -F "${log_files[@]}" &
    tail_pid=$!

    # Wait for the SLURM job to finish (avoid spamming trace from sleep/squeue under bash -x)
    local restore_xtrace=0
    if [[ "${-}" == *x* ]]; then
        set +x
        restore_xtrace=1
    fi
    while [[ -n "$(squeue -j "${slurm_job_id}" -h 2>/dev/null)" ]]; do
        sleep 5
    done
    if [[ "${restore_xtrace}" -eq 1 ]]; then
        set -x
    fi

    # Stop tailing the output
    sleep 3
    kill "${tail_pid}" 2>/dev/null || true
    wait "${tail_pid}" 2>/dev/null || true

    # Get the job exit code from scontrol (format is "ExitCode=0:0", we want the first number)
    # Job information is available for ~5min (MinJobAge = 300 sec) after the job is completed
    job_exit_code=$(scontrol show job "${slurm_job_id}" | grep -oP 'ExitCode=\K[0-9]+' | head -n1)
    if [[ -z "${job_exit_code}" ]]; then
        echo "Failed to get job exit code for job ${slurm_job_id}" >&2
        return 1
    fi
    echo "Job ${slurm_job_id} exited with code: ${job_exit_code}"
    return "${job_exit_code}"
}

function submit_job {
    local exit_code=0
    local slurm_job_id

    # TODO: scancel job when script is terminated
    echo "Submitting job: sbatch --parsable $*"
    slurm_job_id=$(sbatch --parsable "$@") || exit_code=$?
    echo "Submitted job ${slurm_job_id}"

    # shellcheck disable=SC2034
    SLURM_JOB_ID="${slurm_job_id}"

    if [[ ${exit_code} -ne 0 ]]; then
        echo "Failed to submit job" >&2
        return "${exit_code}"
    fi

    wait_for_job_completion "${slurm_job_id}" || exit_code=$?
    if [[ "${exit_code}" -ne 0 ]] ; then
        echo "Job ${slurm_job_id} failed: exit_code=${exit_code}" >&2
    else
        echo "Job ${slurm_job_id} completed with exit code: ${exit_code}"
    fi

    return "${exit_code}"
}
