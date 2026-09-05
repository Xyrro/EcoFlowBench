# Compute environment: Georgia Tech PACE-ICE

Inspected 2026-09-05 from login node `login-ice-gnr-1.pace.gatech.edu` (RHEL 9.6, Slurm).
Everything below was read directly from `sinfo`, `scontrol`, `sacctmgr`, `module avail`,
`pace-quota`, and short `srun` probes. Re-run `scripts/inspect_cluster.sh` to refresh.

## 1. Account, QoS, and what we can actually use

| Item | Value |
|---|---|
| User | `yxiao413` (UID 3357618) |
| Slurm account | `coc` (only account) |
| QoS available | `coc-ice` (default and only) |
| `coc-ice` limits | MaxSubmitPU = 500 jobs, MaxTRESPU = cpu 30720 / gpu 960, priority 7, no MaxWall override |

Partitions whose `AllowQos` includes `coc-ice` or `ALL` (i.e. **usable by us**):

| Partition | MaxTime | DefaultTime | Nodes | Notes |
|---|---|---|---|---|
| `ice-cpu` (default) | 18 h | 1 h | 52 | AllowQos=ALL. Same hardware pool as `coc-cpu` plus one 192-core node |
| `coc-cpu` | 18 h | 1 h | 51 | AllowQos includes coc-ice; group cap cpu=2536 |
| `ice-gpu` | 16 h | 1 h | 37 | AllowQos=ALL. Broadest GPU mix (see §3) |
| `coc-gpu` | 16 h | 1 h | 25 | AllowQos includes coc-ice; group cap gpu=56; no H100/H200 |
| `ice-bw-cpu` | 18 h | 1 h | 4 | AllowQos=ALL. Blackwell nodes used as CPU nodes; MaxCPUsPerSocket=16; MaxMemPerNode=512 GB |
| `ice-bw-gpu` | 18 h | 1 h | 4 | AllowQos=ALL. RTX PRO 6000 Blackwell ×16 per node |

Partitions we **cannot** use with `coc-ice` (listed for completeness): `pace-cpu`, `pace-gpu`
(need `pace-ice`), `coe-gpu` (needs `coe-ice`; that is where the bulk of H100/H200 live).

All partitions: `DefMemPerCPU = 4096 MB`, `OverSubscribe = NO`, `MaxNodes = UNLIMITED`.
Partition priority tier: `ice-*` = 2, `coc-*` = 10 (coc-* jobs are scheduled ahead of ice-* jobs
on the shared nodes, so prefer `coc-cpu` / `coc-gpu` when the hardware there suffices).

## 2. CPU nodes

Typical CPU node (`atl1-1-02-010-*`, `atl1-1-03-007-*`, ...):

| Item | Value |
|---|---|
| CPU | 2 × Intel Xeon Gold 6226 @ 2.70 GHz, 24 cores total, feature tag `core24`, `cpu-small` |
| RAM | 191 000 MB (`RealMemory`), ~183 GB free |
| Local disk | `localNVMe` feature; `/tmp` is a 3.5 TB local volume on the probed nodes |

One large node in `ice-cpu` only: `atl1-1-01-002-36-0`, 2 × Intel Xeon 6972P, 192 cores, 1.5 TB RAM, 6 NUMA domains.

Snapshot at inspection: 45–46 idle CPU nodes on `ice-cpu`/`coc-cpu`, i.e. ~1100 idle cores.

Login node: `login-ice-gnr-1`, 2 × Xeon Gold 6538Y+, 64 cores, 251 GB RAM, shared. Keep it to
downloads, edits, and sub-minute checks.

## 3. GPU nodes

| GPU (GRES name) | Count/node | Nodes (ice-gpu) | Node CPU/RAM | Also in `coc-gpu`? |
|---|---|---|---|---|
| `v100` | 2 | 11 | 24 c / 385 GB | yes |
| `rtx_6000` (Quadro RTX 6000, 24 GB) | 4 | 2 | 24 c / 385 GB | yes |
| `a40` | 2 | 2 | 64 c / 515 GB | yes |
| `a100` | 2 | 4 | 64 c / 515 GB | yes |
| `l40s` | 8 | 4 | 64 c / 773 GB | yes |
| `mi210` (AMD) | 2 | 2 | 64 c / 515 GB | yes |
| `h100` | 8 | 6 | 64 c / 2 063 GB | **no** |
| `h200` | 8 | 6 | 64 c / 2 063 GB | **no** |
| `rtx_pro_6000_blackwell` | 16 | 4 (`ice-bw-gpu`) | 96 c / 2 063 GB | n/a |

Probe on `atl1-1-01-005-5-0` (`--gres=gpu:1` on `ice-gpu` landed on an RTX 6000):
driver 575.57.08, CUDA 12.9 runtime reported by `nvidia-smi`.
Request a specific type with e.g. `--gres=gpu:a100:1` or `--gres=gpu:l40s:1`.

CUDA modules available: `cuda/11.6.0 … 13.0.1` (default 12.9.1); `cudnn/9.2.0.82-12-cuda`.
We do not need the modules: the PyPI `torch` wheel bundles its own CUDA runtime.

## 4. Storage

| Location | Quota | Used at inspection | Notes |
|---|---|---|---|
| `$HOME` = `/home/hice1/yxiao413` | 30 GB | 16.2 GB (54 %) | NFS. Do not put anything project-related here. |
| Scratch = `/storage/ice1/1/8/yxiao413` (`~/scratch`) | 300 GB, 1 M files | 13.1 GB, 67 k files | Lustre, 4.3 PB filesystem. All project data, envs, depots live here. |
| Node-local `/tmp` | 3.5 TB (probed nodes) | — | Wiped per job; use for solver temp files and HDF5 staging. |

**Scratch purge policy:** no purge notice is printed at login and none is exposed by
`pace-quota`; I could not verify a purge schedule from the cluster itself. Treat scratch as
volatile anyway: `scripts/sync_offcluster.sh` (Phase 5) will rsync finished shards to a
destination the owner specifies (open question, see `docs/phase_01_report.md`).

**300 GB is the binding constraint** for the full-scale dataset (v1.0 targets in the brief are
multi-hundred GB). Options: request a quota increase, stream shards off-cluster as they finish,
or trim the ladder. Flagged for the owner.

Project layout on scratch (everything is inside the git repo dir, which *is* `EFB_SCRATCH`;
the heavy directories are git-ignored):

```
/storage/ice1/1/8/yxiao413/EcoFlowBench/     # git repo == $EFB_SCRATCH
├── .venv/           # uv-managed Python 3.11 venv
├── .uv_cache/       # uv download cache
├── julia_depot/     # JULIA_DEPOT_PATH (packages, artifacts, compiled cache)
├── tools/uv/        # uv binary;  tools/python/ = uv-managed CPython
├── .hf_cache/       # HF_HOME
├── .cache/          # XDG_CACHE_HOME (matplotlib, numba, ...)
├── data/            # tiles, shards, quicklooks (Phase 2+)
└── envs/gdal/       # conda env holding GDAL CLI tools only
```

## 5. Network access from compute nodes

Probed with `srun` on `ice-cpu` (`atl1-1-01-002-36-0`) and `ice-gpu` (`atl1-1-01-005-5-0`):
`curl -I` to github.com, pkg.julialang.org and huggingface.co all returned HTTP 200/301,
no proxy variables set. **Compute nodes do have outbound HTTPS.**

Project policy nevertheless stays as in `CLAUDE.md`: downloads, package installs and Hugging
Face pushes are done from the login node; `sbatch` jobs never depend on network. The only
exception used so far is Julia *precompilation* (CPU-heavy, no network needed) which runs via
`srun` to keep the login node light.

## 6. Software modules relevant to us

| Need | Module / source | Decision |
|---|---|---|
| Julia | `julia/1.10.1`, `julia/1.11.3`, `julia/1.12.5` (default) | Use **`julia/1.11.3`** (Circuitscape/Omniscape compat verified below); depot on scratch |
| Python | `python/3.10.10`, `3.11.9`, `3.12.5`, `3.14.0`; `anaconda3/2023.03`; `miniforge/24.3.0-0`; `mamba/1.4.9` | Use **uv-managed CPython 3.11** on scratch (fully isolated from the auto-activated `kneeoa` conda env) |
| GDAL CLI | no module (`module spider gdal/proj/geos` → not found) | conda env `envs/gdal` created with `mamba/1.4.9` (see §7) |
| CUDA | `cuda/12.9.1` default | not needed (torch wheel) |
| git-lfs, rsync | `/usr/bin` | available on login node |
| GNU parallel | `parallel/20220522` | optional |

`module load pace/2024.03` is auto-loaded and pulls in `gcc/12.3.0` + `mvapich2/2.3.7-1`.

## 7. Environment isolation from `kneeoa`

`~/.bashrc` runs `module load anaconda3` and `conda activate ~/scratch/envs/kneeoa`, and
exports `KNEEOA_*` and `HF_HOME=~/scratch/hf_cache`. `scripts/env.sh` undoes all of that:
it `conda deactivate`s, unloads the anaconda3 module, strips `envs/kneeoa` and `anaconda3`
from `PATH`, and sets project-specific `HF_HOME`, `UV_*`, `JULIA_DEPOT_PATH`, `XDG_CACHE_HOME`.
Every job script must start with `source "$EFB_SCRATCH/scripts/env.sh"`.

Verification after sourcing: `CONDA_DEFAULT_ENV` empty, `which python` → `.venv/bin/python`,
no `kneeoa` entries in `PATH`.

## 8. Installed versions (exact)

Recorded 2026-09-05 after installation; re-verify with `scripts/print_versions.sh`.

| Component | Version | Where |
|---|---|---|
| OS | RHEL 9.6, kernel 5.14.0-570 | login + compute |
| Slurm | `/opt/slurm/current` | — |
| Julia | **1.11.3** (`module load julia/1.11.3`) | `/usr/local/pace-apps/manual/packages/julia/1.11.3` |
| Circuitscape.jl | **5.17.1** | `julia_depot/packages/Circuitscape/GEA8x` |
| Omniscape.jl | **0.6.2** | `julia_depot/packages/Omniscape/VSLPS` |
| AlgebraicMultigrid.jl | 1.2.0 | (Circuitscape dependency, AMG preconditioner) |
| Krylov.jl | 0.10.9 | (Circuitscape dependency, CG solver) |
| Julia GDAL (ArchGDAL/GDAL_jll) | see `julia/EcoFlowBenchSolve.jl/Manifest.toml` | pulled in by Circuitscape |
| uv | 0.12.10 | `tools/uv/bin/uv` |
| CPython (uv-managed) | **3.11.16** | `tools/python/cpython-3.11-linux-x86_64-gnu` |
| numpy / scipy | 2.4.6 / 1.17.1 | `.venv` |
| rasterio (bundled GDAL) | 1.4.4 (GDAL 3.10.3) | `.venv` |
| pyproj / shapely / geopandas | 3.7.2 / 2.1.2 / 1.1.4 | `.venv` |
| pandas / pyarrow | 3.0.5 / 25.0.1 | `.venv` |
| h5py / zarr / xarray | 3.16.0 / 3.1.6 / 2026.7.0 | `.venv` |
| pydantic / PyYAML | 2.13.5 / 6.0.3 | `.venv` |
| datasets / huggingface_hub | 5.0.1 / 1.30.0 | `.venv` |
| torch / torchvision | **2.14.0 (CUDA 13.0 wheel)** / 0.29.0 | `.venv` |
| lightning | 2.6.5 | `.venv` |
| neuraloperator | 2.0.0 | `.venv` |
| torch_geometric | 2.8.0.post1 | `.venv` |
| matplotlib / scikit-image | 3.11.1 / 0.26.0 | `.venv` |
| nlmpy | 1.2.0 | `.venv` |
| juliacall | 0.9.35 | `.venv` |
| tqdm / joblib | 4.70.0 / 1.6.0 | `.venv` |
| pytest / ruff / pre-commit | 9.1.1 / 0.16.6 / 4.6.2 | `.venv` (dev extra) |
| GDAL CLI (conda env) | **3.9.3** | `envs/gdal/bin` (mamba/1.4.9, conda-forge) |

Exact Python pins are in `pyproject.toml` and `uv.lock`; exact Julia pins in
`julia/EcoFlowBenchSolve.jl/Manifest.toml`. Both lockfiles are committed.

Sizes on scratch after install: `.venv` 6.4 GB, `julia_depot` ~1.5 GB, `envs/gdal` ~0.5 GB.

### Solver smoke test (2026-09-05, `ice-cpu` node, 4 threads)

`julia --project=julia/EcoFlowBenchSolve.jl julia/EcoFlowBenchSolve.jl/examples/smoke_test.jl data/solver_smoke`

| Check | Result |
|---|---|
| Circuitscape pairwise test case 1 (10×10, 8 focal nodes+polygons, CHOLMOD) | ran; `cum_curmap` max abs diff vs shipped reference 9.3e-7 |
| Omniscape built-in test (30×30, radius 5, block 3, CG+AMG) | ran; `cum_currmap` max abs diff vs shipped reference 2.0e-5 (CG rtol 1e-6) |
| Determinism | two consecutive Circuitscape runs bitwise identical |
| Outputs produced | `*_cum_curmap.asc`, per-pair `*_curmap_i_j.asc`, `*_voltmap_*.asc`, `*_resistances.out`; Omniscape `cum_currmap`, `flow_potential`, `normalized_cum_currmap` |

### Solver facts verified from the installed source (important for the task spec)

- `Circuitscape.compute(::Dict{String,String})` and `compute(::String)` (INI path) are the only
  public entry points; `parse_config` returns a `CSConfig` struct convertible with `Dict(cfg)`.
- CG+AMG path: `Krylov.cg(..., rtol=1e-6, itmax=100_000)` with a post-check that the relative
  residual is `< 1e-4`. **The tolerance is hard-coded, not an INI option.** The brief's
  "tolerance ≤ 1e-8" therefore cannot be met with CG+AMG; the CHOLMOD direct solver
  (`solver = cholmod`) is exact to round-off and is our reference (see `DECISIONS.md`).
- `connect_using_avg_resistances` defaults to **False**, i.e. Circuitscape's default combines
  neighbouring cells by averaging *conductances*.
- `connect_four_neighbors_only` defaults to False (8-neighbour).
- Omniscape: `run_omniscape(path)` and an in-memory method `run_omniscape(cfg::Dict, resistance, ...)`
  returning `cum_currmap[, flow_potential][, normalized_cum_currmap]` arrays; `block_size` must be odd
  (even values are bumped by +1 with a warning); defaults `precision=double`, `solver=cg+amg`,
  `source_threshold=0`, `correct_artifacts=true`, `mask_nodata=true`.


## 8.1 Julia compiled cache on a heterogeneous cluster (finding, 2026-09-05)

Julia 1.11 package images are compiled for the host CPU. With one shared depot on scratch, the login
node (Xeon Gold 6538Y+), the standard CPU nodes (Gold 6226) and the 192-core node (Xeon 6972P) each
found the cache stale and rebuilt it; concurrent array tasks then blocked on the depot's
`*.ji.pidfile` locks (a hang of > 20 min was observed). Fix, in `scripts/env.sh`:
`JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)"` (the multi-target
string Julia's own binaries use) and `JULIA_PKG_PRECOMPILE_AUTO=0`; the cache is built **once on the
login node** (`scripts/generate.py submit` does this before every `sbatch`) and jobs only load it.
If a hang recurs: `find julia_depot/compiled -name '*.pidfile' -delete` after killing the holders.

## 9. Slurm conventions for this project

- Generation: job arrays on `coc-cpu` (fallback `ice-cpu`), `--time=04:00:00`, one shard per
  array task, `--cpus-per-task` = Julia thread count, `--mem` sized per tier; skip shards that
  already exist and validate.
- Baselines: `coc-gpu` / `ice-gpu`, `--gres=gpu:<type>:1`, ≤ 16 h.
- Never `srun`/`sbatch` from inside another job for installs; never rely on `$HOME` for caches.
- Job templates live in `scripts/slurm/`.

## 10. Storage plan: per-tier estimate, feasible v1.0 ladder, HF sync pipeline

### 10.1 Bytes per sample (raw, before compression)

Per pixel, a *fully solved* sample (T1+T2+T3+T4 on one landscape) stores:

| Group | Arrays (float32 unless noted) | bytes / pixel |
|---|---|---|
| inputs, synthetic | resistance, nodata (1 B), focal (int32), source_strength, ground (1 B) | 14 |
| inputs, real extra | covariates C = 8 channels | +32 |
| T1 outputs, K ≤ 4 | cum_current + P·(voltage + pairwise_current), P ≤ 6 | 4 + 48 = 52 |
| T1 outputs, K > 4 | cum_current | 4 |
| T3 outputs | current, voltage | 8 |
| T4 outputs | cum_current, flow_potential, normalized | 12 |
| **total, K ≤ 4** | | **86 (synthetic) / 118 (real)** |
| **total, K > 4** | | **38 (synthetic) / 70 (real)** |

Assuming half the T1 samples have K ≤ 4, the mean is ≈ 62 B/px synthetic, 94 B/px real. Solver
outputs and log-resistance compress poorly (smooth floats); gzip-4 on float32 maps typically
gives 0.6–0.8×. I use **0.7×** below.

| Tier | pixels | raw MB/sample (synthetic / real) | compressed MB/sample (synthetic / real) |
|---|---|---|---|
| S 128² | 16 384 | 1.0 / 1.5 | 0.7 / 1.1 |
| M 256² | 65 536 | 4.1 / 6.2 | 2.8 / 4.3 |
| L 512² | 262 144 | 16 / 25 | 11 / 17 |
| XL 1024² | 1 048 576 | 65 / 99 | 46 / 69 |
| XXL 2048² | 4 194 304 | 260 / 394 | 182 / 276 |

### 10.2 Brief's ladder vs a feasible ladder

| Tier | Brief target | Brief size (≈, 60 % synthetic) | **Proposed v1.0** | Proposed size |
|---|---|---|---|---|
| S | 100k | 85 GB | **60k** (40k synthetic + 20k real) | 50 GB |
| M | 50k | 170 GB | **20k** (12k + 8k) | 68 GB |
| L | 10k | 130 GB | **4k** (2.4k + 1.6k) | 53 GB |
| XL | 2k | 110 GB | **600** (360 + 240) | 33 GB |
| XXL | 200 | 44 GB | **100** (test only, cum_current + reff + T4 only) | 12 GB |
| **total** | | **≈ 540 GB** | | **≈ 215 GB** |

`ecoflowbench-mini` (≈ 250 samples at S, all tasks) ≈ 0.3 GB, well under the 500 MB target.

Sample counts are placeholders until the Phase 5 mini run gives measured solve times; the
*size* column is what the storage decision needs. Both ladders exceed the **100 GB free
private-storage quota on Hugging Face** (verified 2026-09-05,
https://huggingface.co/docs/hub/storage-limits): free accounts get 100 GB private, PRO gets
1 TB private + pay-as-you-go ($18/TB/month), public repos are "best-effort" (up to 10 TB on
PRO). **Decision needed from the owner** (see status report): (a) PRO account for `Xirro`,
(b) make the repo public earlier than planned, or (c) cap v1.0 at ≈ 90 GB while private
(roughly: S 30k, M 8k, L 1.5k, XL 250, XXL 50).

**Owner decision 2026-09-05:** deferred to the Phase 5 gate. Until then only `ecoflowbench-mini` and a small dev subset (< 100 GB total) are planned.

### 10.3 Shard sizing and HF layout

HF recommends < 100k files per repo, < 10k entries per folder, files well under 200 GB, and
Parquet/WebDataset-friendly layouts. EcoFlowBench keeps HDF5 shards but sizes them by bytes so
that every upload is resumable and no shard exceeds ~2 GB:

| Tier | samples / shard | shard size (compressed) |
|---|---|---|
| S | 1000 | ~1 GB |
| M | 400 | ~1.4 GB |
| L | 100 | ~1.4 GB |
| XL | 25 | ~1.4 GB |
| XXL | 4 | ~1 GB |

Repo layout: `data/{tier}/{task_group}/shard-{:05d}.h5`, `index/{tier}.parquet`,
`splits/*.parquet`, `stats/`, `croissant.json`, `README.md` (dataset card).

### 10.4 Sync pipeline (local working set < 150 GB)

`scripts/sync_shards.py` (Phase 5/6) runs on the login node on a cron-like loop or after each
job array:

1. find shards with `*.h5` + `*.ok` (written by the validator) and no `*.uploaded` marker;
2. upload with `huggingface_hub.HfApi.upload_file` (resumable, LFS/Xet);
3. verify: compare local sha256 against the LFS object's sha256 reported by
   `HfApi.get_paths_info(..., expand=True)`;
4. on match, write `*.uploaded` (containing the remote sha256 + commit id) and delete the local
   `.h5`; on mismatch, re-upload up to 3 times, then flag for the owner;
5. refuse to start new generation jobs while `du data/shards` > 120 GB (soft) and abort at
   150 GB (hard), so scratch never exceeds the owner's cap.

Only the Parquet index, split lists, quicklook PNGs and stats stay on scratch permanently
(< 5 GB). Re-download for evaluation uses `hf_hub_download` into a bounded LRU cache.
