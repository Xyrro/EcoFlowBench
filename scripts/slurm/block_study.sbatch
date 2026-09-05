#!/usr/bin/env bash
#SBATCH --job-name=efb-blockstudy
#SBATCH --nodes=1
#SBATCH --ntasks=1
# env: EFB_INPUTS (h5), EFB_BA, EFB_BB, EFB_OUT (json)
set -euo pipefail
source "${EFB_SCRATCH:-/storage/ice1/1/8/yxiao413/EcoFlowBench}/scripts/env.sh"
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "$EFB_SCRATCH/julia/EcoFlowBenchSolve.jl"
julia --project=. examples/omniscape_block_study.jl "$EFB_INPUTS" first "$EFB_BA" "$EFB_BB" "$EFB_OUT"
