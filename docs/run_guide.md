# Run guide: generating AmpScape shards on a new Slurm cluster

Audience: someone launching v1.0 generation on a cluster other than PACE-ICE (for example a
University of Minnesota system). Everything cluster-specific lives in one profile file; the
scripts themselves contain no paths or partition names.

## 1. What the cluster needs

| Requirement | Why | How we did it on ICE |
|---|---|---|
| **Julia 1.11.x** (1.11.3 pinned) | Circuitscape.jl 5.17.1 / Omniscape.jl 0.6.2 manifest was resolved on 1.11.3 | `module load julia/1.11.3` |
| **Julia depot on scratch** (`JULIA_DEPOT_PATH`) | home quotas; shared by login + compute nodes | `julia_depot/` inside the repo dir on scratch |
| **`JULIA_CPU_TARGET`** = `generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)` | one compiled cache that works on every CPU model; without it, mixed-CPU nodes rebuild the cache and race on lock files | set in `scripts/env.sh` |
| **Precompile once on the login node**, never in jobs | jobs must not need network or compile locks | `scripts/generate.py submit` runs `Pkg.precompile()` before `sbatch` |
| **Python 3.11** env from `uv.lock` (or `pyproject.toml` via conda) | driver, QC, finalize | `uv sync --extra dev` with `UV_*` caches on scratch |
| **GDAL CLI** (optional) and rasterio's bundled GDAL | tile extraction only (Phase 2, login node) | conda env `envs/gdal`; **not needed for solving** |
| **No network on compute nodes** is fine | inputs are materialised by `prepare` on the login node; jobs read HDF5 and write HDF5 | verified: jobs ran with no downloads |
| Node-local temp (`/tmp` or `$TMPDIR`) | Circuitscape 5.17.1 reads/writes ASCII grids; keep them off the parallel FS | `node_tmp` in the profile |
| Memory per job | peak RSS: S/M 1.4 GB, L 1.6 GB, XL 3.3 GB, XXL 9.5 GB (+ Julia baseline ≈ 1 GB) | profile `defaults` |
| Walltime | one shard per array task; S shards of 200 landscapes ≈ 25 min; XXL shards of 2 landscapes ≈ 3 h | profile `defaults`, ≤ 4 h |

## 2. One-time setup (login node, network)

```bash
git clone https://github.com/Xyrro/AmpScape.git /scratch/<you>/AmpScape && cd $_
cp configs/cluster/template.yaml configs/cluster/<name>.yaml     # fill in scratch_root, partitions, modules
export AMPSCAPE_SCRATCH=$PWD AMPSCAPE_CLUSTER_PROFILE=<name>
source scripts/env.sh                                            # loads modules, sets JULIA_* and UV_* on scratch
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=$AMPSCAPE_SCRATCH/tools/uv/bin UV_NO_MODIFY_PATH=1 sh
uv python install 3.11 && uv sync --extra dev                    # exact versions from uv.lock
julia --project=julia/AmpScapeSolve.jl -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'   # exact Manifest.toml
python -m pytest -q                                              # 98 tests, ~40 s
srun -p <cpu-partition> -c 4 --mem 8G -t 20:00 \
  julia --project=julia/AmpScapeSolve.jl julia/AmpScapeSolve.jl/examples/smoke_test.jl data/solver_smoke
```
The smoke test must print `SMOKE TEST OK` (reference outputs from the Circuitscape/Omniscape test suites).

Real tiles (Phase 2 covariate stacks) are produced on the login node with network access
(`scripts/download_sources.py`, `sample_tiles.py`, `extract_tiles.py`) or copied from an existing
`data/tiles/<set>/` directory; solving never touches them again.

## 3. Generating a build

```bash
python scripts/generate.py plan     --dataset v1.0-S --out data/builds/v1.0-S --tier S --n-synthetic 60000 --n-real 40000 --shard-size 200
python scripts/generate.py prepare  --build data/builds/v1.0-S            # login node; writes inputs/*.h5 (≈ 0.1 MB/landscape at S)
python scripts/generate.py submit   --build data/builds/v1.0-S --profile <name>   # precompiles, then sbatch --array
python scripts/generate.py status   --build data/builds/v1.0-S
python scripts/generate.py finalize --build data/builds/v1.0-S --quicklooks      # QC + final shards + Parquet index
```
`submit` reads partition, account/QoS, cpus, memory, walltime, shard concurrency and node temp dir
from the profile; every flag can still be overridden. Re-running `submit` only submits shards
without a final file; `solve_shard` skips completed samples, so a killed job is resumed by resubmitting.

## 4. Operational notes

- Keep the local working set bounded: finalize → validate → upload → verify → delete (the sync tool
  is a Phase 6 deliverable; until then, finalize in batches).
- If jobs hang at start with "Being precompiled by another process": a stale cache lock. Kill the
  holders, `find $JULIA_DEPOT_PATH/compiled -name '*.pidfile' -delete`, precompile on the login node.
- `JULIA_NUM_THREADS=1`; CHOLMOD/BLAS use the allocated cores. S/M solves gain nothing from more
  than one core, so 1 core per job is the efficient setting there.
- Determinism was verified bitwise for repeated single-threaded solves on one machine; treat
  outputs from a different cluster as reproducible to solver tolerance (1e-11 relative), not bitwise.
