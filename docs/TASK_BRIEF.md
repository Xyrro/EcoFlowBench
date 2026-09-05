# TASK BRIEF: Building a Benchmark Dataset for Learned Surrogates of Circuit-Theoretic Landscape Connectivity

**Project name:** `EcoFlowBench` (fixed; use this name consistently across repo, packages, HF dataset, and paper)
**Audience of this document:** an autonomous coding agent (Claude Code) executing the project end-to-end
**Target outcomes:** (1) a public dataset on Hugging Face, (2) an open-source generation + evaluation codebase, (3) baseline results and figures for a Datasets & Benchmarks paper

---

## 0. Executive summary

We want the "ImageNet / PGLearn" of landscape connectivity. Circuitscape (and its omnidirectional extension Omniscape) solve large sparse linear systems derived from circuit theory to map ecological flow across a resistance raster. The solver is exact but slow at scale. The goal is a standardized, large, diverse, well-documented dataset of (input resistance landscape, focal-node/source configuration) → (solver outputs) pairs, with official splits, out-of-distribution test sets, evaluation metrics, baselines, and a fully reproducible generation pipeline, so that the ML community can train and fairly compare surrogate models.

Model this after **PGLearn** (power-grid OPF dataset): fixed reference solver, standardized instance families, multiple problem formulations, complete solution data, generation code released as a package, official splits, baseline table, dataset card, and a companion paper.

---

## 1. Working principles for the agent

1. **Start small, then scale.** Every stage must first work end-to-end on a "mini" configuration (~200 samples, 128×128) before scaling up. Never launch a large generation run or large download without first validating the mini run.
2. **Ask before expensive actions.** Before downloading >5 GB of source rasters, launching >1000 CPU-hours of solves, or pushing to Hugging Face, stop and report the plan + estimated cost, and wait for confirmation.
3. **Reproducibility is a hard requirement.** Every artifact must be regenerable from: pinned software versions + config files + random seeds. Record all three in the metadata of every sample.
4. **The solver is the ground truth.** Use tight solver tolerances. Never post-process, normalize, or clip solver outputs in the stored dataset. Store raw float32 (float64 for effective resistance).
5. **Document as you go.** Every module has a docstring, every config has comments, every phase ends with a short `docs/phase_XX_report.md` summarizing what was built, what was verified, and open issues.
6. **Test as you go.** Unit tests for generation code, schema validation tests for data files, smoke tests for baselines. CI must be green before moving to the next phase.
7. **Keep a `CHANGELOG.md` and `DECISIONS.md`.** Log every design decision (with rationale) that deviates from or refines this brief.
8. **Do not fabricate.** If a data source, license term, or literature reference cannot be verified, say so explicitly rather than guessing.

---

## 2. Environment and tooling

### 2.1 Required software
- **Julia ≥ 1.10** with `Circuitscape.jl` and `Omniscape.jl` (pin exact versions in `Project.toml` / `Manifest.toml`; record versions in dataset metadata).
- **Python ≥ 3.10** with: `numpy`, `scipy`, `rasterio`, `pyproj`, `shapely`, `geopandas`, `h5py`, `zarr`, `xarray`, `pyyaml`, `pydantic`, `datasets` (Hugging Face), `huggingface_hub`, `torch`, `torchvision`, `lightning` (or plain torch), `neuraloperator` (for FNO) or a self-implemented FNO, `torch_geometric` (for GNN baseline), `matplotlib`, `pytest`, `ruff`, `pre-commit`, `tqdm`, `joblib`, `nlmpy` (neutral landscape models).
- **GDAL** command-line tools.
- Use `uv` or `conda` for Python; commit lockfiles.

### 2.2 Repository layout (create this)
```
ecoflowbench/
├── README.md
├── LICENSE                      # code: MIT or Apache-2.0; data: CC BY 4.0 (see §10)
├── CHANGELOG.md
├── DECISIONS.md
├── pyproject.toml
├── configs/
│   ├── tasks/                   # task definitions (T1–T4)
│   ├── landscapes/              # synthetic + real landscape sampling configs
│   ├── resistance_tables/       # YAML resistance tables (§5)
│   ├── solver/                  # Circuitscape/Omniscape parameter presets
│   └── datasets/                # full dataset build configs (mini, v1.0, ...)
├── julia/EcoFlowBenchSolve.jl/  # Julia package wrapping Circuitscape/Omniscape
├── ecoflowbench/                # Python package
│   ├── landscapes/              # synthetic generators, real-data tiling
│   ├── resistance/              # covariates → resistance surfaces
│   ├── sources/                 # focal node / source-strength generators
│   ├── solve/                   # Python driver that calls Julia, QC
│   ├── io/                      # HDF5/Zarr schema, readers, writers, validators
│   ├── splits/                  # split logic incl. OOD
│   ├── data/                    # torch Dataset/DataLoader, HF integration
│   ├── models/                  # baselines
│   ├── metrics/                 # evaluation metrics
│   ├── eval/                    # evaluation harness, leaderboard export
│   └── viz/                     # figure generation
├── scripts/                     # CLI entry points for each phase
├── tests/
├── docs/                        # phase reports, datasheet, tutorials
├── notebooks/                   # tutorials (quickstart, train U-Net, evaluate)
└── paper/                       # figures, tables, stats for the manuscript
```

---

## 3. Phase 1 — Prior-art survey and task definition

**Goal:** confirm novelty and lock down the task specification before generating anything.

1. Search and summarize existing work on: learned surrogates for Circuitscape/circuit-theory connectivity, CNN/GNN prediction of current density or effective resistance, connectivity benchmarks, related PDE-surrogate benchmarks (PDEBench, PDEArena), and PGLearn's design. Write `docs/prior_art.md` with citations (only include references you can verify).
2. Write `docs/task_specification.md` that formally defines the tasks below with input/output tensors, shapes, dtypes, and units.

### 3.1 Tasks

| ID | Name | Input | Output | Solver mode |
|----|------|-------|--------|-------------|
| T1 | Pairwise current mapping | resistance raster `R` (H×W), focal-node mask `F` (H×W, int labels) | cumulative current density map `C` (H×W); per-pair current maps (optional, stored for ≤4 nodes); voltage maps | Circuitscape `pairwise` |
| T2 | Effective resistance prediction | `R`, `F` with K focal nodes | K×K effective resistance matrix `Reff` | Circuitscape `pairwise` (from `resistances` output) |
| T3 | Advanced-mode flow | `R`, source-strength raster `S`, ground raster `G` | current map `C`, voltage map `V` | Circuitscape `advanced` |
| T4 | Omnidirectional connectivity | `R`, source-strength raster `S`, window radius `r`, block size `b` | Omniscape `cum_currmap`, `flow_potential`, `normalized_cum_currmap` | Omniscape |
| T5 (optional, stretch) | Inverse / intervention | `R`, `F`, `C_target` or a restoration budget | which pixels to modify | derived from T1 |

Additionally define a **wall-to-wall** variant of T1 (strip sources on opposite edges, both N–S and E–W) because it is common in practice.

### 3.2 Fixed solver conventions (must be identical across the dataset)
- Connectivity: **8-neighbour** (also generate a smaller 4-neighbour subset for ablations, flagged in metadata).
- Resistance combination between neighbours: **Circuitscape's default, average conductance** (`connect_using_avg_resistances = False`: g_ij = (g_i+g_j)/2, diagonal /√2); document explicitly. *(Amended 2026-09-05 after verifying the Circuitscape.jl 5.17.1 source; see DECISIONS.md.)*
- Inputs are resistances (not conductances); resistance ≥ 1; NoData handled as infinite resistance (stored as a separate mask channel).
- Solver: **reference = Circuitscape.jl `solver = cholmod` (direct Cholesky, exact to round-off, deterministic)**, `precision = double`. CG + AMG (`solver = cg+amg`, hard-coded `rtol = 1e-6` in Circuitscape.jl 5.17.1) is the documented fallback only for rasters whose Cholesky factor exceeds node memory; samples record which solver was used and the achieved relative residual. *(Amended 2026-09-05; the brief's original "CG+AMG with tolerance ≤ 1e-8" is not configurable in the installed version.)*
- Omniscape: fixed `block_size`, `radius` in pixels per resolution tier; `source_threshold` fixed; record all.
- All rasters stored north-up, row-major, consistent CRS per tile (EPSG:3857 or an equal-area projection per continent; record the CRS string).

---

## 4. Phase 2 — Landscape instance families

Produce two families plus a resolution/size ladder.

### 4.1 Synthetic landscapes (fully controllable)
Implement generators in `ecoflowbench/landscapes/synthetic.py`:
- Gaussian random fields (variable correlation length ℓ ∈ {2, 8, 32, 128} px, anisotropy).
- Midpoint-displacement fractals (roughness H ∈ {0.2, 0.5, 0.8}).
- NLMpy models: random cluster, planar gradient, edge gradient, distance gradient, mosaic.
- **Barrier overlays**: linear barriers (roads/rivers) with random density, width, orientation, and gap frequency; patch mosaics with categorical classes.
- **Contrast control**: resistance dynamic range spanning {10, 100, 1000, 10 000}.
- Each generator exposes all parameters; parameters are sampled from documented priors and stored in metadata.

### 4.2 Real landscapes (the main scientific value)
Implement `ecoflowbench/landscapes/real.py`:
- Source layers (verify current URLs/licenses before downloading; record version/year):
  - Land cover: ESA WorldCover (10 m) and/or Copernicus Global Land Cover (100 m).
  - Elevation: Copernicus DEM GLO-30 or GLO-90.
  - Roads: **GRIP4** (Global Roads Inventory Project, Meijer et al. 2018; CC BY 4.0 — verify). OSM is *not* used (ODbL share-alike; decided 2026-09-05).
  - Water: HydroSHEDS / HydroRIVERS.
  - Human modification: Global Human Modification Index (gHM) or Human Footprint.
  - Protected areas: WDPA (for realistic focal nodes).
  - Biomes/ecoregions: RESOLVE Ecoregions 2017 (for stratified sampling).
- **Stratified global sampling**: sample tile centres stratified by biome × continent × human-modification tercile. Target a balanced design; log the number of tiles per stratum. Avoid tiles that are >90% ocean/ice/NoData.
- Tile extraction at each resolution tier; reproject to the **WGS84 / UTM zone of the tile centre (EPSG:326xx north / 327xx south)**, recording the EPSG code per tile (decided 2026-09-05); cache tiles on disk as GeoTIFF; store the raw covariate stack alongside resistance so users can train end-to-end models.
- Keep a manifest (`tiles.parquet`) with tile ID, centre lat/lon, CRS, stratum, source layer versions, and download checksums.

### 4.3 Resolution and size ladder
| Tier | Pixel size | Raster size | Approx. samples (v1.0 target) |
|------|-----------|-------------|------------------------------|
| S | 30–100 m | 128×128 | 100k+ (mostly synthetic + real) |
| M | 100 m | 256×256 | 50k |
| L | 100–300 m | 512×512 | 10k |
| XL | 300 m–1 km | 1024×1024 | 2k |
| XXL | 1 km | 2048×2048 | 200 (test only; expensive) |

Adjust numbers after measuring actual solve times in the mini run. Report the compute budget before scaling.

---

## 5. Phase 3 — Resistance surface construction

Implement `ecoflowbench/resistance/` with a YAML-driven mapping from covariates to resistance.

- Provide **at least four resistance tables** in `configs/resistance_tables/`, each sourced from literature (cite in the YAML header):
  1. `generic_hm`: resistance = 1 + a·gHM^b (continuous human modification transform).
  2. `large_mammal`: land-cover classes → resistance (forest low, cropland moderate, urban very high), roads additive penalty by class, water barrier, slope penalty.
  3. `amphibian`: wetlands/forest low, dry open land high, roads extreme, slope penalty stronger.
  4. `forest_bird`: forest low, open land moderate, urban high, roads mild, elevation-band effect.
  5. (optional) `random_table`: randomly perturbed class → resistance mapping to decorrelate the surrogate from any single expert table.
- Each real tile × each table → one resistance raster (multiple samples per tile; record `table_id`).
- Ensure resistance ∈ [1, R_max]; NoData → mask; write unit tests checking value ranges and mask consistency.
- Save both the resistance raster and the covariate stack (land cover one-hot or categorical, DEM, slope, road distance, water distance, gHM) in the sample.

---

## 6. Phase 4 — Focal nodes and source configurations

Implement `ecoflowbench/sources/`:
- **Point pairs** (T1/T2): 2–8 random focal nodes with minimum separation constraints; also protected-area polygons (WDPA) as focal regions on real tiles.
- **Wall-to-wall** (T1 variant): opposite edge strips, N–S and E–W.
- **Advanced mode** (T3): source-strength raster from habitat suitability proxy (e.g., inverse of resistance, thresholded, or a random smooth field); ground nodes at edges or at random patches.
- **Omniscape** (T4): source raster from the same suitability proxy; radius and block size per tier.
- Store focal nodes both as a raster mask (int32 labels) and as a table (id, row, col, or polygon WKT).
- Ensure every focal node is connected in the resistance graph (no isolated nodes); otherwise resample and log.

---

## 7. Phase 5 — Solver pipeline (Julia + Python driver)

### 7.1 Julia package `EcoFlowBenchSolve.jl`
- Functions: `solve_pairwise`, `solve_advanced`, `solve_omniscape`, each taking in-memory arrays + parameter struct, returning outputs + a `SolveStats` struct (wall time, iterations if available, peak memory, solver version, thread count, converged flag).
- Write temp INI/config files only if the API requires them; prefer the programmatic API.
- Multithreading via `Threads.@threads` over samples; a batch mode that reads a manifest, solves, and writes results to shard files.
- Deterministic: same input → bitwise-identical output on the same machine; record machine info.

### 7.2 Python driver `ecoflowbench/solve/`
- Orchestrates: read sample manifest → dispatch batches to Julia (via `juliacall` or subprocess) → collect outputs → run QC → write shards.
- Parallelism across nodes via `joblib` locally and a SLURM template in `scripts/slurm/`.
- **QC checks** (fail → flag and exclude, keep log): non-convergence, NaN/Inf in outputs, current conservation residual above threshold, isolated focal nodes, Omniscape edge artifacts, output all-zeros.
- Every sample records solve time; aggregate into `stats/solve_times.parquet` (needed for the speed-up argument in the paper).

### 7.3 Mini run (mandatory gate)
- 200 synthetic + 50 real samples at tier S, all tasks. Verify outputs visually (save PNG quicklooks), verify schema, measure timings, extrapolate the full budget, and write `docs/phase_05_report.md`. **Stop and report before scaling.**

---

## 8. Phase 6 — Data format, schema, and splits

### 8.1 Storage
- Primary format: **HDF5 shards** (1000 samples per shard, gzip level 4, chunked per sample) with an accompanying **Parquet index** (one row per sample with all scalar metadata). Provide an optional Zarr export.
- One sample group contains:
  - `inputs/resistance` (float32, H×W), `inputs/nodata_mask` (bool), `inputs/covariates` (float32, C×H×W, real tiles only, channel names in attrs), `inputs/focal_mask` (int32), `inputs/source_strength` (float32, T3/T4), `inputs/ground` (int32, T3)
  - `outputs/cum_current` (float32), `outputs/voltage` (float32, optional), `outputs/pairwise_current` (float32, P×H×W, only if K ≤ 4), `outputs/reff` (float64, K×K), `outputs/omniscape/{cum_current, flow_potential, normalized}` (float32)
  - `meta/` attrs: `sample_id` (UUID), `task_ids`, `family` (synthetic/real), `generator` + params (JSON), `tile_id`, `lat`, `lon`, `crs`, `pixel_size_m`, `tier`, `resistance_table_id`, `source_config`, `solver_name`, `solver_version`, `solver_params` (JSON), `solve_time_s`, `converged`, `qc_flags`, `seed`, `created_at`, `pipeline_git_sha`, `dataset_version`.
- Validate every shard against a JSON Schema / pydantic model in `ecoflowbench/io/schema.py`; ship the schema in `docs/schema.md`.

### 8.2 Splits
Provide official splits as Parquet lists of sample IDs:
- `train`, `val`, `test_id` (in-distribution, same generators/regions/tables).
- **OOD test sets** (essential for the paper):
  - `test_ood_region`: held-out biomes/continents never seen in training.
  - `test_ood_scale`: larger raster sizes than any training tier.
  - `test_ood_table`: a held-out resistance table.
  - `test_ood_synth2real`: train on synthetic only, test on real (defined via subset flags, not a separate copy).
  - `test_ood_contrast`: higher dynamic range than training.
- Splits are by **tile**, not by sample, to avoid leakage between resistance tables of the same tile.

### 8.3 Sizes
- Ship `ecoflowbench-mini` (~500 MB) for quick experiments and `ecoflowbench-v1.0` (full). Report the total size.

---

## 9. Phase 7 — Python loaders and Hugging Face integration

- `ecoflowbench.data.EcoFlowBenchDataset(task, split, tier, root)` → returns dict tensors; supports lazy shard loading, optional log-transform helper, normalization statistics computed on train only (`stats/norm_stats.json`).
- Hugging Face: build a `datasets` loading script or use the Parquet + shard file convention so `load_dataset("Xirro/EcoFlowBench", "T1_S")` works; implement `scripts/push_to_hub.py` with dry-run mode. **Do not push without confirmation.** The dataset repo is **private** during development (`Xirro/EcoFlowBench`) and doubles as the off-cluster sync destination: every finished shard is validated → uploaded → checksum-verified → deleted from scratch, keeping the local working set under 150 GB (see `docs/compute_env.md` §10).
- Generate **Croissant** metadata (`scripts/export_croissant.py`) and validate it; ship `croissant.json` in the HF repo.
- Dataset card (`README.md` on HF) following the Datasheets for Datasets template: motivation, composition, collection process, preprocessing, uses, distribution, maintenance, licenses of upstream sources, known limitations, ethical considerations (sensitivity of protected-area locations; do not include any non-public data).

---

## 10. Phase 8 — Licensing and provenance

- Verify the license of each upstream layer and whether derivative redistribution is permitted (ESA WorldCover: CC BY 4.0; Copernicus DEM: check terms; OSM: ODbL — note share-alike implications for derived rasters and decide whether to store OSM-derived rasters or only distances/densities; WDPA: check redistribution restrictions — likely store only derived focal masks, not the polygons; gHM: check). Write `docs/licenses.md`. If a source cannot be redistributed, either drop it or store only non-reversible derived products and document this.
- Code license: MIT or Apache-2.0. Data license: CC BY 4.0 (subject to upstream constraints).
- Add `CITATION.cff`.

---

## 11. Phase 9 — Metrics and evaluation harness

Implement `ecoflowbench/metrics/` and `ecoflowbench/eval/`:

**Pixel-level (maps):** MSE, MAE in log space (log1p), relative L2 error, SSIM, PSNR.

**Domain-level:**
- High-flow region IoU: IoU between top-q% pixels (q ∈ {1, 5, 10}) of prediction and ground truth.
- Pinch-point recall: detect local maxima / high-current narrow regions in ground truth; measure recall in prediction.
- Rank correlation (Spearman) of pixel currents.
- Corridor overlap: overlap of thresholded corridor masks with Dice coefficient.

**Effective resistance (T2):** relative error, log-space MAE, Spearman correlation across pairs, rank agreement of nearest-neighbour ordering.

**Physics consistency:** divergence / current-conservation residual computed from predicted current and known resistance (document the formula); Kirchhoff residual if voltage is predicted.

**Efficiency:** inference time per sample vs. recorded solver time → speed-up factor; memory.

**Scale extrapolation curve:** metric vs. raster size on `test_ood_scale`.

Provide `scripts/evaluate.py --predictions <dir> --split <name>` producing a JSON + Markdown table, and `scripts/export_leaderboard.py`.

---

## 12. Phase 10 — Baselines

Implement in `ecoflowbench/models/` with a shared training script (`scripts/train.py --model unet --task T1 --tier S`):
1. **Non-learned baseline**: coarsen resistance ×4 → solve → upsample (measures what a cheap approximation achieves).
2. **U-Net** (input channels: log-resistance, nodata mask, focal/source channels).
3. **FNO** (2D Fourier Neural Operator) — tests long-range/global behaviour and resolution transfer.
4. **Swin-UNet or ViT-based** encoder-decoder.
5. **GNN** (grid-as-graph with edge conductances; message passing; tests exact graph structure use).
6. (Optional) physics-informed variant with the conservation residual as an auxiliary loss.

Requirements: fixed seeds, three runs per configuration for mean ± std, identical data normalization, early stopping on `val`, report all metrics on `test_id` and every OOD split, log to CSV/W&B-compatible format. Save checkpoints and prediction dumps for the leaderboard.

Produce `paper/tables/baselines.md` and `paper/figures/` (qualitative prediction panels, error vs. size, speed-up plot, dataset statistics).

---

## 13. Phase 11 — Documentation, tutorials, tests, CI

- `README.md`: one-paragraph pitch, install, quickstart (load mini, train U-Net in 10 minutes, evaluate), citation, links.
- `docs/`: task spec, schema, datasheet, licenses, generation guide (how to rebuild from scratch), contribution guide, phase reports.
- `notebooks/`: 01_quickstart, 02_train_unet, 03_evaluate_and_compare, 04_visualize_samples, 05_rebuild_mini_dataset.
- Tests: generators (shapes, ranges, determinism), resistance mapping, source generation (connectivity), schema validation, metrics (against hand-computed cases), dataloader.
- CI (GitHub Actions): lint, Python tests, Julia tests, mini-pipeline smoke test on 5 samples.

---

## 14. Phase 12 — Paper support package

Produce in `paper/`:
- `dataset_statistics.md`: sample counts per task/tier/family/stratum, resistance distribution plots, solve-time distributions, map of tile centres, biome coverage chart.
- `baselines.md`: full results tables with mean ± std.
- `ood_analysis.md`: per-OOD-split degradation plots.
- `figures/`: publication-quality PDFs/PNGs generated by `ecoflowbench/viz/`.
- `outline.md`: a suggested manuscript outline (Introduction/motivation, Related work, Task definitions, Dataset construction, Statistics, Baselines, OOD results, Limitations & ethics, Maintenance plan).

Target venue: **NeurIPS 2027 Datasets & Benchmarks track** (primary; the 2026 deadline has passed); ecological venue as secondary. Follow the D&B checklist (hosting, license, maintenance, datasheet, DOI). **Croissant metadata (core + RAI fields) is mandatory for the track and is a Phase 7 deliverable (`scripts/export_croissant.py`, validated with the Hugging Face Croissant validator).** Obtain a DOI via Zenodo mirror in addition to Hugging Face.

---

## 15. Acceptance criteria (per phase)

| Phase | Done when |
|-------|-----------|
| 1 | `prior_art.md` and `task_specification.md` exist; task tensor specs reviewed |
| 2 | Synthetic generators pass tests; ≥50 real tiles downloaded and tiled with manifest and checksums |
| 3 | ≥4 resistance tables produce valid rasters on all mini tiles; tests pass |
| 4 | All source configurations generate connected focal nodes; tests pass |
| 5 | Mini run (≈250 samples) solved for all tasks; quicklooks verified; timings and full-budget estimate reported; **user confirmation obtained** |
| 6 | Schema + validator implemented; mini shards validate; splits produced by tile with all OOD sets |
| 7 | `EcoFlowBenchDataset` works; HF dry-run succeeds; dataset card drafted |
| 8 | `licenses.md` complete; any non-redistributable layers handled |
| 9 | All metrics unit-tested; `evaluate.py` runs on mini predictions |
| 10 | All baselines train on mini; full results on v1.0 after scaling (3 seeds) |
| 11 | README, docs, notebooks complete; CI green |
| 12 | Paper package complete; figures regenerate from a single script |

---

## 16. Reporting cadence

After each phase, write the phase report and post a concise summary: what was built, what was verified, measured numbers (sizes, timings), risks, and the next step. Use the "ask before expensive actions" rule at every scaling point.

---

## 17. Open questions to flag early (do not silently decide)

- Compute budget available (CPU-hours, node count, storage) — needed to fix v1.0 sample counts.
- ~~Hugging Face organization / repository name and whether the repo should be gated during review.~~ Decided 2026-09-05: `Xirro/EcoFlowBench`, private during development.
- ~~Which equal-area CRS convention to adopt for real tiles.~~ Decided 2026-09-05: WGS84/UTM zone of tile centre.
- ~~Whether to include OSM-derived rasters given ODbL share-alike.~~ Decided 2026-09-05: no OSM; roads from GRIP4.
- ~~Whether per-pair current maps should be stored for all samples (storage cost) or only for K ≤ 4.~~ Decided 2026-09-05: only for K ≤ 4; K > 4 stores cumulative map + Reff only.

Begin with Phase 1 and the repository scaffold. Proceed phase by phase, honouring the gates above.
