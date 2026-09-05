# Phase 1 report — environment, scaffold, prior art, task specification

Date: 2026-09-05. Status: **complete, awaiting owner confirmation** before Phase 2.

## What was built

| Item | Location |
|---|---|
| Cluster inspection (partitions, QoS, GPUs, storage, network, modules) | `docs/compute_env.md` |
| Isolated toolchain on scratch: env bootstrap, uv venv, Julia depot, GDAL env | `scripts/env.sh`, `scripts/setup_gdal_env.sh`, `scripts/print_versions.sh`, `scripts/inspect_cluster.sh` |
| Repository scaffold (brief §2.2): 11 Python subpackages, config dirs, Julia package skeleton, tests | `ecoflowbench/`, `configs/`, `julia/EcoFlowBenchSolve.jl/`, `tests/` |
| Pinned dependencies | `pyproject.toml` + `uv.lock` (Python), `julia/EcoFlowBenchSolve.jl/{Project,Manifest}.toml` |
| Reference solver presets | `configs/solver/circuitscape_reference.yaml`, `configs/solver/omniscape_reference.yaml` |
| Solver smoke test | `julia/EcoFlowBenchSolve.jl/examples/smoke_test.jl` |
| Prior-art survey (42 verified references) | `docs/prior_art.md` |
| Task specification (T1, T1W, T2, T3, T4, T5-stretch; tensors, dtypes, units, conventions) | `docs/task_specification.md` |
| Decision log / changelog | `DECISIONS.md`, `CHANGELOG.md` |

## What was verified

- **Isolation:** after `source scripts/env.sh`, `CONDA_DEFAULT_ENV` is empty, `python` resolves to
  `.venv/bin/python` (CPython 3.11.16), no `kneeoa`/`anaconda3` entries remain in `PATH`.
- **Versions:** Julia 1.11.3, Circuitscape.jl 5.17.1, Omniscape.jl 0.6.2, torch 2.14.0 (CUDA 13.0
  wheel), neuraloperator 2.0.0, torch_geometric 2.8.0.post1, nlmpy 1.2.0, GDAL CLI 3.9.3
  (full table in `docs/compute_env.md` §8).
- **Solver works** (compute node, 4 threads): Circuitscape pairwise test case 1 reproduces the
  shipped reference `cum_curmap` to 9.3e-7; Omniscape built-in 30×30 test reproduces the
  reference to 2.0e-5 (CG tolerance); two consecutive Circuitscape runs are bitwise identical.
- **Tests:** `pytest` — 5 passed (imports, layout, pins, Julia manifest, solver preset);
  `ruff check` clean; Julia `Pkg.test()` for the skeleton passes.

## Measured numbers

| Quantity | Value |
|---|---|
| Idle CPU cores at inspection (`ice-cpu`/`coc-cpu`) | ~1100 (46 idle 24-core nodes) |
| GPU types reachable with our QoS | V100, RTX 6000, A40, A100, L40S, MI210, H100 (6 nodes), H200 (6 nodes), RTX PRO 6000 Blackwell |
| Max walltime | 18 h CPU partitions, 16 h GPU partitions |
| Scratch quota | 300 GB, 1 M files (13 GB used before this project; +~8.5 GB toolchain now) |
| Home quota | 30 GB (16.2 GB used, untouched by this project) |
| Julia depot download + precompile | 6 min (login, network) + 1.5 min (compute node, 8 threads) |
| Python env install | 1 m 40 s, 6.4 GB |

## Findings that deviate from the brief (logged in `DECISIONS.md`)

1. **CG tolerance is not configurable** in Circuitscape.jl 5.17.1 (hard-coded `rtol = 1e-6`);
   the reference solver is therefore **CHOLMOD** (direct, exact). CG+AMG timings will still be
   collected for the speed-up narrative.
2. **Circuitscape's default neighbour rule is average *conductance***, not average resistance
   as the brief's wording suggests. The spec follows the software default and documents both.
3. **Compute nodes have internet access.** Policy unchanged (installs on login node), noted in
   `docs/compute_env.md` §5.

## Risks and blockers (need owner input before or during Phase 2)

1. **Storage.** 300 GB scratch is far below the v1.0 sizes implied by the ladder
   (100k S + 50k M + 10k L + 2k XL + 200 XXL samples with maps ≈ several hundred GB even with
   gzip). Need either a quota increase, an off-cluster sync target (where?), or a smaller ladder.
2. **Compute partitions.** `coe-gpu` (bulk H100/H200) is not accessible with QoS `coc-ice`;
   12 H100/H200 nodes remain reachable via `ice-gpu`, plus L40S/A100. Enough for mini and
   probably for full baselines, but queue times are unknown.
3. **Scratch purge policy** could not be verified from the cluster; treat scratch as volatile.
4. **Venue timing.** NeurIPS 2026 Evaluations & Datasets deadline (May 6, 2026) has passed;
   the realistic target is the 2027 cycle (Croissant metadata is mandatory there).
5. **Open spec questions** (`docs/task_specification.md` §9): per-pair maps for K > 4, real-tile
   CRS convention, OSM/ODbL handling, Omniscape solver and r/b per tier, HF org name.

## Next step (Phase 2, on confirmation)

Synthetic landscape generators (`ecoflowbench/landscapes/synthetic.py`) with tests, then the
real-data source verification (URLs, licences, versions) and the ≥ 50-tile mini download
(< 5 GB, otherwise a gate). No data download will start before the owner answers the storage
question above.
