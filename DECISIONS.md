# Design decisions

Record every decision that refines or deviates from `docs/TASK_BRIEF.md`.

| Date | Decision | Rationale | Phase |
|------|----------|-----------|-------|
| 2026-09-05 | Project named EcoFlowBench | fixed by brief | 0 |
| 2026-09-05 | Julia **1.11.3** via `module load julia/1.11.3` (not the 1.12.5 default) | Circuitscape 5.17.1 / Omniscape 0.6.2 resolve and precompile cleanly on 1.11; 1.10 is LTS but older; 1.12 is newest and least tested with this dependency stack | 0 |
| 2026-09-05 | Python via **uv** (CPython 3.11.16 managed by uv on scratch) instead of conda | reproducible `uv.lock`; complete isolation from the `kneeoa` conda env auto-activated by `~/.bashrc`; `torch` wheels bundle CUDA so no cluster CUDA module is needed | 0 |
| 2026-09-05 | GDAL CLI tools from a separate conda-forge env (`envs/gdal`, GDAL 3.9.3) appended to `PATH`; Python uses rasterio's bundled GDAL 3.10.3 | no GDAL module on PACE-ICE; keeps the Python env pip-only | 0 |
| 2026-09-05 | All caches/depots/envs live inside `$EFB_SCRATCH` (= repo dir) and are git-ignored | 30 GB home quota; single scratch tree is easy to sync / purge-proof | 0 |
| 2026-09-05 | Keep the "no network in `sbatch` jobs" rule even though compute nodes were verified to have outbound HTTPS | robustness (policy could change; jobs must be replayable offline); exception: Julia precompile runs via `srun` because it needs no network but ~6 CPU-minutes | 0 |
| 2026-09-05 | **Reference solver = Circuitscape `solver = cholmod` (direct, double precision)**, not CG+AMG | the brief asks for relative residual ≤ 1e-8; in Circuitscape 5.17.1 the CG tolerance is hard-coded (`Krylov.cg rtol=1e-6`, accept if residual < 1e-4) and not exposed as an INI option; CHOLMOD is exact to round-off, deterministic, and fast for grids ≤ 1024². CG+AMG timings will still be recorded for the speed-up argument; XXL tier feasibility with CHOLMOD is measured in Phase 5 | 1 |
| 2026-09-05 | **Neighbour combination = Circuitscape default `connect_using_avg_resistances = False`** (average *conductance*: g_ij = (g_i+g_j)/2, diagonal /√2) | the brief says "Circuitscape default (average resistance)"; the actual default in the source is average conductance. We follow the software default because it is what practitioners get, and document it precisely in `docs/task_specification.md` §2 | 1 |
| 2026-09-05 | Circuitscape is driven through its public `compute(::Dict)` API with temporary ASCII/GeoTIFF files on node-local `/tmp`; Omniscape through the in-memory `run_omniscape(cfg, resistance, source)` method | Circuitscape 5.17.1 has no in-memory public API; writing to node-local disk is cheap and keeps us on the supported code path | 1 |
| 2026-09-05 | Omniscape `block_size` values in the tier table are odd (Omniscape bumps even values with a warning) | verified behaviour of Omniscape 0.6.2 | 1 |
| 2026-09-05 | Wall-to-wall T1 variant is stored as a separate task id `T1W` with 2 focal regions (N/S or E/W strips) | keeps T1 sample schema (K point nodes) uniform; strips are regions, not points | 1 |
