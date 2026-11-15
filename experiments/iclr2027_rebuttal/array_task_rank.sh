#!/bin/bash
# Prints "<rank> <count>" of this SLURM array task within its own array job, so a manifest split as
# lines rank, rank+count, ... is covered by any contiguous --array=MIN-MAX (incl. resubmitted --array=2).
set -euo pipefail

min="${SLURM_ARRAY_TASK_MIN:-0}"
max="${SLURM_ARRAY_TASK_MAX:-0}"
count="${SLURM_ARRAY_TASK_COUNT:-1}"
id="${SLURM_ARRAY_TASK_ID:-0}"
if [ $((max - min + 1)) -ne "$count" ]; then
    echo "array_task_rank: --array must be one contiguous range without step (got min=$min max=$max count=$count)" >&2
    exit 1
fi
echo "$((id - min)) $count"
