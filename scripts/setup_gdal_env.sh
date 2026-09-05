#!/usr/bin/env bash
# Create a minimal conda environment on scratch that provides the GDAL command-line
# tools (gdalinfo, gdal_translate, gdalwarp, ogr2ogr, ...). The Python stack does NOT
# live here; it is a uv venv (see scripts/env.sh). This env is only appended to PATH.
#
# Run on the login node (needs network). Idempotent.
set -euo pipefail
source "$(dirname "$0")/env.sh"
ENV_DIR="$EFB_SCRATCH/envs/gdal"
export CONDA_PKGS_DIRS="$EFB_SCRATCH/.conda_pkgs"
export CONDA_ENVS_PATH="$EFB_SCRATCH/envs"
mkdir -p "$CONDA_PKGS_DIRS"
module load mamba/1.4.9
if [ -x "$ENV_DIR/bin/gdalinfo" ]; then
  echo "GDAL env already present: $("$ENV_DIR/bin/gdalinfo" --version)"; exit 0
fi
mamba create -y -p "$ENV_DIR" -c conda-forge --override-channels "gdal=3.9.*" "proj" "geos" 2>&1 | tail -5
"$ENV_DIR/bin/gdalinfo" --version
