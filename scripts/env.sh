#!/usr/bin/env bash
# EcoFlowBench environment bootstrap for PACE-ICE.
#
# Usage:   source scripts/env.sh
#
# What it does:
#   * Deactivates the unrelated `kneeoa` conda env that ~/.bashrc auto-activates
#     and strips it (and the anaconda3 module) from PATH, so this project is
#     fully isolated from it.
#   * Points every cache / depot / environment at scratch (home quota is 30 GB).
#   * Loads the Julia module and exposes `uv` and the project virtualenv.
#
# Nothing here is heavy; safe to source on login nodes and inside sbatch jobs.

# ---------------------------------------------------------------------------
# Locations (all under scratch)
# ---------------------------------------------------------------------------
export EFB_SCRATCH="${EFB_SCRATCH:-/storage/ice1/1/8/yxiao413/EcoFlowBench}"
export EFB_ROOT="$EFB_SCRATCH"                 # git repo root == scratch dir
export EFB_DATA="$EFB_SCRATCH/data"            # generated shards, tiles, caches
export EFB_TOOLS="$EFB_SCRATCH/tools"          # uv binary, misc. tools
export EFB_JULIA_VERSION="${EFB_JULIA_VERSION:-1.11.3}"

# ---------------------------------------------------------------------------
# Isolate from the kneeoa conda env / anaconda3 module
# ---------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  # deactivate as many nested envs as exist
  while [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_SHLVL" != "0" ]; do
    conda deactivate 2>/dev/null || break
  done
fi
unset KNEEOA_RAW KNEEOA_WORK KNEEOA_SCRATCH CONDA_DEFAULT_ENV CONDA_PREFIX
if command -v module >/dev/null 2>&1; then
  module unload anaconda3 2>/dev/null || true
fi
# Strip leftover kneeoa / anaconda3 entries from PATH.
PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
      | grep -v -E '/envs/kneeoa/|/anaconda3/' | paste -sd ':' -)"
export PATH

# ---------------------------------------------------------------------------
# Python (uv) — venv and caches on scratch
# ---------------------------------------------------------------------------
export UV_CACHE_DIR="$EFB_SCRATCH/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$EFB_TOOLS/python"
export UV_PROJECT_ENVIRONMENT="$EFB_SCRATCH/.venv"
export PIP_CACHE_DIR="$EFB_SCRATCH/.pip_cache"
export PATH="$EFB_TOOLS/uv/bin:$PATH"
if [ -d "$EFB_SCRATCH/.venv/bin" ]; then
  export VIRTUAL_ENV="$EFB_SCRATCH/.venv"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
fi

# ---------------------------------------------------------------------------
# Julia — depot on scratch, module-provided binary
# ---------------------------------------------------------------------------
export JULIA_DEPOT_PATH="$EFB_SCRATCH/julia_depot"
export JULIA_PROJECT="$EFB_SCRATCH/julia/EcoFlowBenchSolve.jl"
export JULIA_PKG_USE_CLI_GIT=true
if command -v module >/dev/null 2>&1; then
  module load "julia/$EFB_JULIA_VERSION" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Hugging Face / misc caches on scratch
# ---------------------------------------------------------------------------
export HF_HOME="$EFB_SCRATCH/.hf_cache"
export XDG_CACHE_HOME="$EFB_SCRATCH/.cache"
export MPLCONFIGDIR="$EFB_SCRATCH/.cache/matplotlib"
export NUMBA_CACHE_DIR="$EFB_SCRATCH/.cache/numba"
export TMPDIR="${TMPDIR:-/tmp}"

# GDAL CLI conda env (optional; created by scripts/setup_gdal_env.sh)
if [ -d "$EFB_SCRATCH/envs/gdal/bin" ]; then
  export PATH="$PATH:$EFB_SCRATCH/envs/gdal/bin"
  export GDAL_DATA="$EFB_SCRATCH/envs/gdal/share/gdal"
  export PROJ_LIB="$EFB_SCRATCH/envs/gdal/share/proj"
fi

mkdir -p "$EFB_DATA" "$EFB_TOOLS" "$UV_CACHE_DIR" "$JULIA_DEPOT_PATH" "$HF_HOME" "$XDG_CACHE_HOME" 2>/dev/null
