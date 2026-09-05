#!/usr/bin/env bash
#SBATCH --job-name=efb-solve
#SBATCH --nodes=1
#SBATCH --ntasks=1
# time / cpus / mem / partition / array / output are passed by scripts/generate.py submit
# One shard per array task. No network is needed: inputs were materialised by `generate.py prepare`
# on the login node and the Julia depot is precompiled on scratch.
set -euo pipefail
source "${EFB_SCRATCH:-/storage/ice1/1/8/yxiao413/EcoFlowBench}/scripts/env.sh"
export JULIA_NUM_THREADS=1                       # one solve at a time; CHOLMOD/BLAS use the allocated cores
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JULIA_DEPOT_PATH="$EFB_SCRATCH/julia_depot"
export TMPDIR="${TMPDIR:-/tmp}/efb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
echo "host=$(hostname) build=$EFB_BUILD shard=$SHARD cpus=${SLURM_CPUS_PER_TASK:-1} start=$(date -u +%FT%TZ)"
cd "$EFB_SCRATCH"
EXTRA=()
[ -n "${EFB_CONFIGS:-}" ] && EXTRA+=(--configs "$EFB_CONFIGS")
[ "${EFB_FORCE:-0}" = "1" ] && EXTRA+=(--force)
python scripts/generate.py solve --build "$EFB_BUILD" --shard "$SHARD" \
    --solver "${EFB_SOLVER:-cholmod}" --fallback "${EFB_FALLBACK:-cg+amg}" --omniscape-solver "${EFB_OSOLVER:-cholmod}" "${EXTRA[@]}"
echo "end=$(date -u +%FT%TZ)"
