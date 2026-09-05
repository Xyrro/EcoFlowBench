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
| 2026-09-05 | Owner accepted CHOLMOD reference solver and average-conductance rule; brief §3.2 amended accordingly | owner decision | 1 |
| 2026-09-05 | Off-cluster sync = private HF dataset repo `Xirro/EcoFlowBench`; shards validated → uploaded → sha256-verified → deleted locally; local working set < 150 GB | 300 GB scratch quota; owner decision | 1 |
| 2026-09-05 | Per-pair current/voltage maps stored only for K ≤ 4 | storage; owner decision (as in brief) | 1 |
| 2026-09-05 | Real-tile CRS = WGS84/UTM zone of tile centre (EPSG:326xx/327xx), EPSG recorded per tile | owner decision; UTM is near-equal-area within a tile and universally supported | 1 |
| 2026-09-05 | Roads from GRIP4 (CC BY 4.0, to be verified), OSM not used | ODbL share-alike avoided; owner decision | 1 |
| 2026-09-05 | Target venue NeurIPS 2027 D&B; Croissant export added to Phase 7 | 2026 deadline passed; Croissant mandatory | 1 |
| 2026-09-05 | HF org `Xirro`; `HF_ORG=Xirro` in CLAUDE.md | owner decision | 1 |
| 2026-09-05 | Synthetic cost fields map to resistance by R = contrast^f (log-uniform), not linear | spreads values evenly in log-resistance, which is what matters for current flow; linear mapping kept as an option (`mapping: linear`) | 2 |
| 2026-09-05 | Synthetic NoData: valid pixels reduced to the largest 8-connected component | guarantees a connected resistance graph so no focal node can be isolated by construction | 2 |
| 2026-09-05 | NLMpy calls wrapped in a seeded-global-RNG context (`seeded_global_rng`) | NLMpy 1.2.0 draws from `np.random` global state; wrapping keeps landscapes reproducible from `seed` alone | 2 |
| 2026-09-05 | Real covariate stack = 8 channels: landcover, elevation, slope, road_distance, road_class, river_distance, river_order, ghm (GeoTIFF, deflate, band descriptions) | minimal set that supports the four resistance tables of Phase 3; class one-hot is derived at load time | 2 |
| 2026-09-05 | Raster sources read as windowed `/vsicurl/` COG requests (WorldCover, Copernicus DEM) instead of downloading 3°/1° tiles | pilot moves ~50 MB instead of ~4 GB; keeps every extraction under the 5 GB gate | 2 |
| 2026-09-05 | Vector sources (GRIP4, HydroRIVERS) downloaded per region as shapefiles with `.qix` spatial indexes; pilot restricted to GRIP4 regions 1,2,3,5,7 (N/S America, Africa, Middle East & Central Asia, Oceania) | keeps the pilot download at 2.76 GB (< 5 GB gate); Europe (1.2 GB) and S/E Asia (0.7 GB) are added at full scale after owner approval | 2 |
| 2026-09-05 | gHM v1 at 1 km used for both stratification and the `ghm` channel (bilinear to tile grid) | only version with a verified stable download + CC BY 4.0; the 300 m temporal version is a later upgrade | 2 |
| 2026-09-05 | Strata = RESOLVE biome × realm × gHM tercile; tercile edges computed on the attributed candidate sample and recorded in `tile_specs.json` (pilot: 0.0329, 0.1821) | the brief's "continent" is approximated by RESOLVE realm; edges are stored so later samples use the same cut points | 2 |
| 2026-09-05 | Tile usability QC: reject if WorldCover NoData + water + snow/ice > 90 % or DEM NaN > 50 %; rejected tiles are replaced from a same-stratum reserve list | brief §4.2 rule plus a DEM-coverage guard | 2 |
| 2026-09-05 | Tile grid origin snapped to a whole pixel in UTM so `(lat, lon, size, pixel_m)` reproduces the grid exactly | reproducibility of tiles from the manifest alone | 2 |
