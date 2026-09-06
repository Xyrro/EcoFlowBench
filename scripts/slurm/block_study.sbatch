#!/usr/bin/env bash
#SBATCH --job-name=ampscape-blockstudy
#SBATCH --nodes=1
#SBATCH --ntasks=1
# env: AMPSCAPE_INPUTS (h5), AMPSCAPE_BA, AMPSCAPE_BB, AMPSCAPE_OUT (json)
set -euo pipefail
source "${AMPSCAPE_SCRATCH:-/storage/ice1/1/8/yxiao413/EcoFlowBench}/scripts/env.sh"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "$AMPSCAPE_SCRATCH/julia/AmpScapeSolve.jl"
julia --project=. examples/omniscape_block_study.jl "$AMPSCAPE_INPUTS" "${AMPSCAPE_SID:-first}" "$AMPSCAPE_BA" "$AMPSCAPE_BB" "$AMPSCAPE_OUT"
