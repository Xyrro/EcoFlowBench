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
# EFB_MODULES (colon-separated) overrides the default module list; set by the cluster profile via generate.py.
export EFB_MODULES="${EFB_MODULES-julia/$EFB_JULIA_VERSION}"

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
# Portable multi-target pkgimages: the login node (Xeon Gold 6538Y+), the standard CPU nodes (Gold 6226)
# and the large node (Xeon 6972P) otherwise each rebuild the shared compiled cache and race on its
# pidfiles (observed hang, 2026-09-05). Same string Julia's own binaries are built with.
export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)"
export JULIA_PKG_PRECOMPILE_AUTO=0   # jobs never precompile; scripts/generate.py submit does it on the login node
export JULIA_PROJECT="$EFB_SCRATCH/julia/EcoFlowBenchSolve.jl"
export JULIA_PKG_USE_CLI_GIT=true
if command -v module >/dev/null 2>&1 && [ -n "$EFB_MODULES" ]; then
  for m in ${EFB_MODULES//:/ }; do module load "$m" 2>/dev/null || true; done
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
  # Appended (not prepended) so the venv's rasterio/pyproj keep their own bundled GDAL/PROJ.
  # Do NOT export GDAL_DATA / PROJ_LIB here: they would shadow rasterio's PROJ database
  # (different DATABASE.LAYOUT version) and break CRS lookups in Python.
  export PATH="$PATH:$EFB_SCRATCH/envs/gdal/bin"
fi

mkdir -p "$EFB_DATA" "$EFB_TOOLS" "$UV_CACHE_DIR" "$JULIA_DEPOT_PATH" "$HF_HOME" "$XDG_CACHE_HOME" 2>/dev/null
