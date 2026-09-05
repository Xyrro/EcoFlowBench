# EcoFlowBench task specification (v0.1, Phase 1)

Status: **v0.2, owner-reviewed 2026-09-05**. Remaining ⚠ items are Phase 5 measurements (Omniscape
window sizes and solver), see §9.

This document is normative for Phases 2–12. It defines the mathematical model, the fixed solver
conventions, and the exact tensors (name, shape, dtype, units, semantics) stored for every task.
Solver behaviour statements were verified against the installed sources
(Circuitscape.jl 5.17.1, Omniscape.jl 0.6.2; see `docs/compute_env.md` §8).

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| H, W | raster height (rows) and width (columns), pixels; north-up, row 0 = northernmost row |
| Δ | pixel size in metres (per tier) |
| R ∈ [1, R_max]^{H×W} | resistance raster, dimensionless per-pixel resistance ("Ω per cell") |
| M ∈ {0,1}^{H×W} | NoData mask, 1 = pixel is NoData (no graph node) |
| g = 1/R | per-pixel conductance |
| V = {(r,c) : M_rc = 0} | graph nodes (valid pixels), n = |V| |
| E | edges between 8-neighbours that are both valid |
| G | n×n weighted adjacency (conductance) matrix, L = D − G the graph Laplacian |
| F ∈ ℤ^{H×W} | focal-node label raster: 0 = none, k ≥ 1 = focal node / region k |
| K | number of focal nodes/regions |
| S ∈ ℝ_{≥0}^{H×W} | source-strength raster (T3, T4) |
| Gnd ∈ ℤ^{H×W} | ground raster (T3): 0 = none, 1 = grounded pixel (direct ground, R→0) |

## 2. Fixed solver conventions (identical for every sample)

Verified from `Circuitscape.jl 5.17.1/src/raster/pairwise.jl::construct_graph` and `src/core.jl`:

1. **Graph.** Each valid pixel is one node. Edges join cardinal (N/S/E/W) and diagonal
   neighbours (8-neighbourhood; INI `connect_four_neighbors_only = False`). A 4-neighbour
   ablation subset is generated separately and flagged `graph_connectivity = 4`.
2. **Edge conductance** (INI `connect_using_avg_resistances = False`, Circuitscape's default;
   see `DECISIONS.md`):
   - cardinal: g_ij = (g_i + g_j) / 2
   - diagonal: g_ij = (g_i + g_j) / (2·√2)
   
   (For reference, the non-default `= True` option uses g_ij = 1 / mean(R_i, R_j) and
   1 / (√2 · mean(R_i, R_j)); it is *not* used.)
3. **Inputs are resistances** (`habitat_map_is_resistances = True`), R ≥ 1 everywhere valid.
   NoData pixels are removed from the graph (equivalent to infinite resistance) and stored as
   the separate mask channel M; the stored R raster contains 1.0 at NoData positions
   (never used, keeps float ranges clean).
4. **Linear solver:** `solver = cholmod` (SuiteSparse direct Cholesky), `precision = double`.
   Rationale: the CG+AMG path has a hard-coded `rtol = 1e-6` (Krylov.cg) and accepts any
   relative residual `< 1e-4`; neither is exposed as an option, so the brief's ≤ 1e-8 target
   is only reachable with the direct solver. CHOLMOD is exact to round-off and bitwise
   deterministic on one machine. Each sample records the achieved relative residual
   ‖L v − b‖₂ / ‖b‖₂ recomputed in float64 by our driver (QC threshold 1e-10).
5. **Physics.** For a current vector b (injections, Σb = 0 when no ground, or with grounded
   nodes removed), voltages solve L v = b. Branch current i_ij = g_ij (v_i − v_j). Node current
   density (Circuitscape's "current map", `out.jl::get_node_currents`) is the total current
   passing through node i:
   c_i = max( Σ_j max(i_ji, 0), Σ_j max(i_ij, 0) ), which for non-source nodes equals
   ½ Σ_j |i_ij| and at a unit source/ground equals 1.
6. **Disconnected components.** If a focal node lies in a component without any other focal
   node, Circuitscape reports Reff = −1. EcoFlowBench forbids this: source generators resample
   until all K focal nodes share one connected component (Phase 4); samples violating this fail QC.
7. **Raster geometry.** Rasters are stored north-up, row-major (C order), with CRS, affine
   transform and pixel size recorded in metadata. Synthetic samples carry a nominal CRS
   (EPSG:3857-like, no georeference meaning) and `pixel_size_m` per tier. Real tiles use the
   WGS84/UTM zone of the tile centre (EPSG:326xx/327xx), recorded per tile.
8. **Units.** R is dimensionless (per-cell resistance); with unit current injection, voltages
   are in "volts" = Ω·A with the same dimensionless Ω, effective resistance in Ω, currents in A.
   No normalisation, clipping or log transform is applied to stored outputs.
9. **Pixel size and resistance are decoupled.** Circuitscape does not scale conductance by
   pixel size; the same R raster at a different Δ gives identical outputs. Δ is therefore
   *metadata only*, relevant for interpreting real tiles and for the Omniscape radius.

## 3. Tasks

All tasks share the input block in §4.1. Output tensors are stored raw (float32 maps,
float64 Reff) exactly as produced by the solver.

### 3.1 T1 — Pairwise current mapping

- **Solver mode:** Circuitscape `scenario = pairwise`, point focal nodes given as an integer
  raster (`point_file`), all K(K−1)/2 pairs.
- **Input:** R (H×W), M (H×W), F (H×W) with K ∈ [2, 8] point focal nodes (one pixel each).
- **Output:**

| Tensor | Shape | dtype | Semantics |
|---|---|---|---|
| `outputs/cum_current` | H×W | float32 | Σ over all pairs of node current c (Circuitscape `*_cum_curmap`) |
| `outputs/voltage` | P×H×W (P = K(K−1)/2) | float32 | per-pair voltage map, v_target = 0 (Circuitscape `*_voltmap_i_j`); stored only when K ≤ 4 (P ≤ 6) |
| `outputs/pairwise_current` | P×H×W | float32 | per-pair node-current maps (`*_curmap_i_j`); stored only when K ≤ 4 |
| `outputs/pair_index` | P×2 | int16 | (i, j) focal labels for each of the P slices, i < j |
| `outputs/reff` | K×K | float64 | effective resistance matrix (T2 target; always stored) |

  Note: Circuitscape's cumulative map includes the focal pixels themselves (each carries
  current 1 per pair it participates in); `set_focal_node_currents_to_zero = False`.
  For K > 4 only `cum_current` and `reff` are stored (owner decision, §9).

### 3.2 T1W — Wall-to-wall current mapping (T1 variant)

- **Solver mode:** Circuitscape `pairwise` with exactly K = 2 focal *regions*: two opposite
  edge strips of width w_strip = max(1, H/64) pixels. Two samples per landscape:
  `orientation = "NS"` (top strip label 1, bottom strip label 2) and `"EW"` (left 1, right 2).
  Regions are passed as the point raster with repeated labels, which Circuitscape treats as one
  short-circuited focal region (**verified 2026-09-05** with `examples/region_check.jl`: strip
  Reff 0.47 vs 1.04 for single pixels; every focal-region pixel carries the full injected
  current in the current map, hence strip pixels are excluded from metrics).
- **Input:** R, M, F (labels 1/2 on the strips, 0 elsewhere), scalar `orientation`.
- **Output:** `outputs/cum_current` (H×W float32; equals the single pair map), `outputs/voltage`
  (1×H×W float32), `outputs/reff` (2×2 float64, the wall-to-wall effective resistance = a
  scalar landscape "conductance" metric). Pixels inside the strips are excluded from
  metric computation (a mask is derivable from F).

### 3.3 T2 — Effective resistance prediction

- **Solver mode:** same run as T1; T2 is a different *target* on the same samples.
- **Input:** R, M, F (K point focal nodes) plus the focal-node table (id, row, col).
- **Output:** `outputs/reff` K×K float64, symmetric, zero diagonal, Reff_ij > 0.
  Also derived at load time (not stored): the vector of the K(K−1)/2 upper-triangular entries.
- Baselines predict either the full matrix (padded to 8×8 with a validity mask) or per-pair
  scalars conditioned on two focal positions; the loader exposes both views.

### 3.4 T3 — Advanced-mode flow (sources and grounds)

- **Solver mode:** Circuitscape `scenario = advanced`; the source raster values are the
  injected currents (`use_unit_currents = False`), `ground_file_is_resistances = False`,
  `use_direct_grounds = True` (grounds are ideal, resistance 0), `remove_src_or_gnd = keepall`.
- **Input:** R, M, S (H×W float32, ≥ 0, current injected per pixel; total ΣS = 1 by
  construction so outputs are comparable across samples), Gnd (H×W int8, 1 = grounded pixel).
  Source generators (Phase 4): thresholded habitat-suitability proxy from R, or a random smooth
  field; grounds: one full edge, all four edges, or 1–3 random patches. Every source pixel must
  be connected to at least one ground pixel.
- **Output:**

| Tensor | Shape | dtype | Semantics |
|---|---|---|---|
| `outputs/current` | H×W | float32 | node current density c (Circuitscape `*_curmap`) |
| `outputs/voltage` | H×W | float32 | node voltage v, grounds at 0 |

### 3.5 T4 — Omnidirectional connectivity (Omniscape)

- **Solver mode:** Omniscape `run_omniscape(cfg, resistance, source)` in-memory method with
  `configs/solver/omniscape_reference.yaml`; `precision = double`, `solver = cg+amg` (Omniscape
  window solves are small; CHOLMOD is also available and will be timed in Phase 5 ⚠).
- **Algorithm recap (Omniscape 0.6.2):** every pixel with S > `source_threshold` (= 0), taken
  on a grid with stride `block_size` (block-centred targets, odd block size), becomes a ground
  in turn; all source pixels within `radius` pixels inject current proportional to S; the
  resulting current maps are summed. Flow potential is the same computation on a uniform
  resistance raster; normalized current = cumulative / flow potential (0 where undefined).
  `correct_artifacts = true` (Omniscape's block-artefact correction) is applied and recorded.
- **Input:** R, M, S (H×W float32 ≥ 0), scalars `radius` r and `block_size` b (pixels) fixed per tier:

| Tier | H=W | r (px) | b (px) | r·Δ at nominal Δ |
|---|---|---|---|---|
| S | 128 | 16 | 3 | 1.6 km @100 m |
| M | 256 | 32 | 5 | 3.2 km @100 m |
| L | 512 | 64 | 9 | 6.4–19 km @100–300 m |
| XL | 1024 | 128 | 17 | 38–128 km @300 m–1 km |
| XXL | 2048 | 256 | 33 | 256 km @1 km |

  ⚠ r/b values are proposals to be confirmed against Phase 5 timings (Omniscape cost ∝
  (H·W/b²)·solve(r)).
- **Output:**

| Tensor | Shape | dtype | Semantics |
|---|---|---|---|
| `outputs/omniscape/cum_current` | H×W | float32 | Omniscape `cum_currmap` |
| `outputs/omniscape/flow_potential` | H×W | float32 | Omniscape `flow_potential` |
| `outputs/omniscape/normalized` | H×W | float32 | Omniscape `normalized_cum_currmap` |

  NoData pixels hold −9999 in Omniscape's rasters; we store 0 and rely on M (documented;
  the raw −9999 convention is not "solver output", it is file formatting).

### 3.6 T5 (stretch, optional) — Inverse / intervention

Not generated in v1.0. Defined for completeness: given R, F and a budget B (number of pixels),
choose a modification set Ω ⊂ V with |Ω| ≤ B that sets R_Ω ← 1 to maximise Σ Reff decrease
(or to best match a target C_target). Ground truth would come from greedy solver-in-the-loop
search on tier S only. Schema fields reserved: `inputs/target_current`, `inputs/budget`,
`outputs/intervention_mask`.

## 4. Sample schema (HDF5 group per sample; full schema in Phase 6 `docs/schema.md`)

### 4.1 Inputs (all tasks)

| Dataset | Shape | dtype | Notes |
|---|---|---|---|
| `inputs/resistance` | H×W | float32 | R ∈ [1, R_max]; 1.0 at NoData |
| `inputs/nodata_mask` | H×W | bool | M |
| `inputs/covariates` | C×H×W | float32 | real tiles only; channel names in attrs (`landcover_class` (categorical, stored as float code), `elevation_m`, `slope_deg`, `road_distance_m`, `water_distance_m`, `ghm`, …) |
| `inputs/focal_mask` | H×W | int32 | F (T1/T1W/T2) |
| `inputs/focal_table` | K×3 | int32 | (label, row, col) for point nodes; region samples store centroid |
| `inputs/source_strength` | H×W | float32 | S (T3/T4) |
| `inputs/ground` | H×W | int8 | Gnd (T3) |

### 4.2 Outputs
As in §3 per task. A sample may carry several task outputs (e.g. one landscape solved for
T1, T3 and T4 with `task_ids = ["T1","T2","T3","T4"]`); the Parquet index has one boolean
column per task.

### 4.3 Metadata attributes (`meta/`)

`sample_id` (UUID4), `task_ids`, `family` ∈ {synthetic, real}, `generator` (name) and
`generator_params` (JSON), `tile_id`, `lat`, `lon`, `crs` (WKT or EPSG), `transform` (6 floats),
`pixel_size_m`, `tier` ∈ {S,M,L,XL,XXL}, `H`, `W`, `resistance_table_id`, `r_max`,
`contrast` (= max/min of valid R), `source_config` (JSON), `graph_connectivity` ∈ {8, 4},
`solver_name`, `solver_version`, `julia_version`, `solver_params` (JSON of the preset),
`solve_time_s` (wall time of the solver call), `solver_threads`, `machine` (hostname, CPU model),
`residual_rel` (max over systems), `converged` (bool), `qc_flags` (list of strings, empty = clean),
`seed` (uint64), `created_at` (ISO-8601 UTC), `pipeline_git_sha`, `dataset_version`.

## 5. Resolution/size ladder (from brief §4.3, unchanged pending Phase 5 timings)

| Tier | Δ | H=W | v1.0 target |
|---|---|---|---|
| S | 30–100 m | 128 | 100k+ |
| M | 100 m | 256 | 50k |
| L | 100–300 m | 512 | 10k |
| XL | 300 m–1 km | 1024 | 2k |
| XXL | 1 km | 2048 | 200 (test only) |

Sizes will be re-budgeted after the mandatory mini run (brief §7.3) and the compute/storage
constraints in `docs/compute_env.md` (300 GB scratch quota).

## 6. Input-space design axes (what generalisation is tested over)

| Axis | Values / range | Split that stresses it |
|---|---|---|
| Landscape family | synthetic {GRF, fractal, NLMpy cluster/gradient/mosaic, barriers}, real (biome × continent × gHM tercile strata) | `test_ood_synth2real`, `test_ood_region` |
| Contrast R_max/R_min | {10, 100, 1000, 10 000} | `test_ood_contrast` (train ≤ 1000, test 10 000) |
| Resistance table (real) | generic_hm, large_mammal, amphibian, forest_bird, random_table | `test_ood_table` |
| Raster size | tiers S–XXL | `test_ood_scale` (train ≤ L, test XL/XXL) |
| Source configuration | K ∈ [2,8] points, WDPA regions, edge strips, S/Gnd fields, Omniscape windows | in-distribution |
| Graph connectivity | 8 (main), 4 (ablation subset) | flagged, not a split |

## 7. Evaluation targets per task (metric definitions in Phase 9)

| Task | Primary metric | Secondary |
|---|---|---|
| T1 / T1W | log1p-MAE of cum_current, relative L2 | top-q% IoU, pinch-point recall, Spearman, SSIM/PSNR, conservation residual |
| T2 | relative error of Reff (pairs), Spearman over pairs | log-MAE, nearest-neighbour rank agreement |
| T3 | log1p-MAE of current; MAE of voltage | conservation + Kirchhoff residuals |
| T4 | log1p-MAE of cum_current and normalized | top-q% IoU, corridor Dice |
| all | inference time vs `solve_time_s` → speed-up; metric vs size on `test_ood_scale` | |

Model inputs are standardised by the loader (log R, M, focal/source channels); targets stay raw.

## 8. Reproducibility contract

Every sample is a pure function of (pinned software versions, `configs/**` used, `seed`).
Regeneration: `scripts/generate.py --config configs/datasets/<name>.yaml --shard <i>` must
reproduce the shard bitwise on the same machine class (CHOLMOD is deterministic; Omniscape
CG+AMG is deterministic single-threaded, which is why `parallelize = false` inside a solve).

## 9. Open questions — resolved by the owner on 2026-09-05

1. **Storage/compute.** Off-cluster destination = private HF dataset repo `Xirro/EcoFlowBench`;
   shards are validated → uploaded → checksum-verified → deleted locally; local working set
   < 150 GB. Per-tier storage estimate and feasible v1.0 ladder: `docs/compute_env.md` §10.
2. **Per-pair maps:** stored only for K ≤ 4 (P ≤ 6). K > 4 samples store `cum_current` and
   `reff` only (as specified in §3.1).
3. **CRS for real tiles:** WGS84 / UTM zone of the tile centre (EPSG:326xx north, 327xx south),
   EPSG code recorded per tile in the manifest and in `meta/crs`.
4. **Roads:** GRIP4 (Global Roads Inventory Project), not OSM. URL and licence verified in
   `docs/licenses.md` before use; if unavailable, report back before choosing an alternative.
5. **Omniscape solver / r, b per tier:** proposals in `configs/solver/omniscape_reference.yaml`
   to be finalised from measured mini-run timings (each Omniscape solve < 10 min on one node
   for tiers S/M/L); justification goes to `DECISIONS.md`.
6. **Hugging Face org:** `Xirro`; repo private during development.
7. **Venue:** NeurIPS 2027 Datasets & Benchmarks; Croissant metadata is a Phase 7 deliverable.
