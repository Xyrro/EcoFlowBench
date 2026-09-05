#!/usr/bin/env bash
# Print exact versions of every pinned component (for docs/compute_env.md and sample metadata).
set -euo pipefail
source "$(dirname "$0")/env.sh"
echo "julia: $(julia --version)"
julia --project="$JULIA_PROJECT" -e 'using Pkg; for (_, d) in Pkg.dependencies(); d.name in ("Circuitscape","Omniscape","AlgebraicMultigrid","Krylov") && println("  ", d.name, " ", d.version); end'
echo "python: $(python --version)"
python -c "import importlib.metadata as m; [print('  ',p, m.version(p)) for p in ['numpy','scipy','rasterio','torch','neuraloperator','torch_geometric','nlmpy','h5py','zarr','datasets']]"
command -v gdalinfo >/dev/null && echo "gdal cli: $(gdalinfo --version)"
