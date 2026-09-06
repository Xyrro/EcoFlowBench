# EcoFlowBench v1.0 dataset plan (compute-agnostic)

Date: 2026-09-05. Source of truth for counts: `configs/datasets/v1_0.yaml`; every table below is
produced by `scripts/dataset_plan_tables.py` from that file and from the Phase 5 measurements.
ICE hosts only the pipeline, the mini dataset and dev shards; where v1.0 runs is decided from §6.

## 1. Design principles

1. **Complete coverage.** Every cell of landscape family × resistance table × source configuration ×
   task × tier has a non-zero target; nothing is subset for cost. T4 is solved on every landscape.
2. **One reference solver.** Circuitscape.jl 5.17.1 / Omniscape.jl 0.6.2, CHOLMOD, double precision,
   8-neighbour, average conductance; CG+AMG only as an automatic memory fallback (flagged).
3. **Reproducible from (versions, configs, seeds).** Sample ids are UUID5 of (dataset, family, key);
   synthetic landscapes and source configurations regenerate from their seed; real tiles from the
   manifest (lat, lon, EPSG, size, pixel size).
4. **Splits by unit of leakage** (tile for real, seed family for synthetic), never by sample.

## 2. Coverage matrix

Two ladders: the brief's §4.3 ladder as baseline, and the recommended v1.0. Counts are
**landscapes** (one resistance raster); solves follow from the source configurations in §2.3.

### 2.1 Landscapes per tier

| tier | brief §4.3 | **recommended** | change | justification |
|---|---|---|---|---|
| S (128², 100 m) | 100 000 | **100 000** | — | as brief |
| M (256², 100 m) | 50 000 | **50 000** | — | as brief |
| L (512², 200 m) | 10 000 | **20 000** | +10 000 | L is the largest *training* tier for `test_ood_scale` (XL/XXL are test-only); 10k split 60/40 synthetic/real over 5 tables leaves 800 real tiles and a few hundred landscapes per generator, too thin for a 5-table × 7-generator matrix. |
| XL (1024², 500 m) | 2 000 | **4 000** | +2 000 | XL is pure OOD-scale test; 2k over 5 tables × 7 generators + hard cases gives < 30 per cell — metrics per cell would be noise. |
| XXL (2048², 1 km) | 200 | **400** | +200 | same argument; 400 gives 32 real tiles × 5 tables and ≈ 30 synthetic per generator. Cost is 589 CPU-h (§6), affordable. |
| **total** | 162 200 | **174 400** | +12 200 | |

Family share 60 % synthetic / 40 % real at every tier (brief: "mostly synthetic + real"). Real
landscapes = tiles × 5 tables (4 expert + 1 seeded random table per tile), so the tile counts are
S 8 000, M 4 000, L 1 600, XL 320, XXL 32 (**recommendation: more real tiles per biome than the
Phase 2 pilot** — the stratified sampler targets balanced biome × realm × gHM-tercile cells, with a
floor of 150 S-tier tiles per biome so that no biome has fewer than 750 real landscapes).

### 2.2 Recommended ladder: landscapes per tier × family × stratum / table

{land_rec}

The synthetic generator mix (24 % GRF, 16 % fractal, 12 % random cluster, 4 % planar, 4 % edge,
8 % distance gradient, 12 % mosaic, 20 % hard cases) is fixed across tiers so that every generator
appears at every tier; barrier and patch-mosaic overlays and NoData blobs are drawn from the
documented prior on top of every base generator (`DEFAULT_PRIOR`).

### 2.3 Solves per tier × family × source configuration (task instances)

{solves_rec}

Totals (recommended): **{tot_land_r:,} landscapes, {tot_solves_r:,} solves**; brief baseline: {tot_land_b:,} landscapes, {tot_solves_b:,} solves.
Every landscape gets `points` (T1 + T2, K ∈ [2, 8]), both wall-to-wall strips (T1W), `advanced` (T3)
and `omniscape` (T4); real tiles with ≥ 2 eligible habitat patches (≈ 40 %) also get `regions` (T1R).
A 5 % slice of tier-S synthetic landscapes is additionally solved with the 4-neighbour graph and
flagged `graph_connectivity = 4` (brief §3.2 ablation).

### 2.4 Brief baseline ladder (for comparison)

{land_brief}

## 3. Hard-case stratum (named `hard`, 20 % of synthetic landscapes at every tier)

| hard case | share of `hard` | definition | landscapes (recommended, all tiers) |
|---|---|---|---|
| `high_contrast_1e4` | 30 % | contrast 10⁴ from any base generator | {hc_high_contrast_1e4} |
| `rmax_saturated` | 15 % | > 50 % of valid pixels at r_max (near-degenerate); QC-flagged, train/val-excluded, kept in the index | {hc_rmax_saturated} |
| `narrow_corridor` | 30 % | 1–3 barrier walls spanning the raster with 1–3 gaps of 1–3 px each (all flow funnels through the gaps) | {hc_narrow_corridor} |
| `large_nodata` | 25 % | NoData fraction 0.25–0.45, single connected component | {hc_large_nodata} |

Real tiles contribute hard cases naturally (coastal tiles with large water bodies, high-Andes
`forest_bird` saturation); they are identified post hoc by the same criteria and get the same
`hard_case` flag in the index.

## 4. OOD test sets (brief §8.3)

| split | held-out unit | rule | expected size (recommended ladder) |
|---|---|---|---|
| `train` / `val` / `test_id` | tile (real), seed family (synthetic) | 80 / 10 / 10 of in-distribution units, stratified by tier × family × table / generator | ≈ 126k / 16k / 16k landscapes |
| `test_ood_region` | real tiles in biomes *Montane Grasslands & Shrublands* and *Mangroves*, and realm *Australasia* | every table of those tiles; never in train/val; 2 biomes + 1 realm ≈ 12 % of real tiles | ≈ 8 400 real landscapes |
| `test_ood_scale` | tiers XL and XXL | entire tiers are test-only; train/val use S–L | 4 400 landscapes |
| `test_ood_table` | resistance table `forest_bird` | all `forest_bird` landscapes (r_max 100, elevation bands: structurally different) are test-only; training sees generic_hm, large_mammal, amphibian, random | ≈ 14 000 real landscapes (S–L) |
| `test_ood_contrast` | synthetic contrast 10⁴ | all 10⁴-contrast landscapes are test-only; train/val ≤ 10³ | ≈ 6 300 synthetic landscapes (S–L) |
| `test_ood_synth2real` | flag only | evaluation subset = all real `test_id`; training restricted to `family = synthetic` by a loader flag | no extra samples |

A landscape can belong to several OOD sets (e.g. an XL `forest_bird` tile in Australasia); the index
carries one boolean column per set. Hold-outs are applied before the 80/10/10 draw so that
in-distribution splits contain no held-out biome, realm, table, tier or contrast.

## 5. Split design and leakage rules

- **Unit for real landscapes = tile.** All 5 tables and all source configurations of a tile share one
  split. Tiles of the same 3° WorldCover cell that overlap spatially are merged into one unit.
- **Unit for synthetic landscapes = seed family.** A base seed defines the landscape; its 4-neighbour
  ablation duplicate and any hard-case variant derived from the same base seed inherit the split.
- **No cross-tier leakage of real tiles:** a real tile centre used at tier S is not re-used at M/L
  (tiles are sampled independently per tier with a 50 km exclusion radius around every existing centre).
- **Synthetic seeds are tier-disjoint** (seed ranges partitioned per tier).
- Split assignment is a deterministic hash of the unit key with the dataset seed, so re-planning
  reproduces the same splits; the lists are shipped as Parquet (`splits/*.parquet`).

## 6. Cost table (measured on ICE, Phase 5)

Per-solve medians (CHOLMOD, single-threaded, warm JIT): points K = 4 0.17 / 0.87 / 4.1 / 13.3 / 70.9 s
(S/M/L/XL/XXL); T4 6.2 / 48 / 178 / 794 / 5 517 s (XXL = median of 4) at the Phase-5 block sizes (S b3 … XL b17, XXL b33)
and 52 / 134 / 577 / 1 896 / 9 613 s at the fidelity blocks (S b1 and M b1 measured, 630–1 548 s for
M b1 → b3 scaled; L/XL/XXL scaled by (b_old/b_new)², a law the XL 17→33 pair confirms: predicted
×0.265, measured ×0.28);
T1W 0.08 s, T3 0.07 s, T1R 0.21 s at S, scaling linearly in pixels (exponents 1.00–1.07).
Peak RSS (process high-water incl. ≈ 1 GB Julia baseline): 1.4 / 1.4 / 1.6 / 3.3 / 9.5 GB.
Compressed storage per landscape (all configs): synthetic 0.93 / 3.0 / 11 / 41 / 140 MB, real ≈ 1.5×.

### 6.1 Recommended ladder, recommended Omniscape blocks (fidelity rule b/r ≤ 0.10, §7)

{cost_rec}

{wall_rec}

### 6.2 Recommended ladder, Phase-5 Omniscape blocks (b/r 0.13–0.19) for comparison

{cost_rec_p5}

{wall_rec_p5}

### 6.3 Brief baseline ladder (fidelity blocks)

{cost_brief}

{wall_brief}

CPU-hours are *core-hours of single-threaded solving* (+15 % overhead). Because solves are
single-threaded, wall-clock at N concurrent cores = CPU-hours / N; memory (not cores) sets the
allocation for XL/XXL (9.5 GB per XXL job). T4 is ≈ 90 % of every total. Omniscape block sizes are
fixed on fidelity (§7): XL keeps block 17 (coarser blocks change the maps by > 2 %), so the XL/XXL costs
above are the ones to plan with.

## 7. Omniscape block size (fidelity study)

Omniscape's `block_size` is part of the method definition (targets are block centres; cost ∝
1/block²), so the benchmark must fix it explicitly and consistently. Two kinds of runs, 3 samples each,
same radius, CHOLMOD: **coarsening tests** (standard block vs the cheaper block proposed in the Phase 5
report: XL 17 vs 33, XXL 33 vs 65) and **anchors** against the exact block-1 Omniscape (S 3 vs 1,
M 5 vs 1). Metrics on valid pixels of `cum_current`: relative L2, max |Δ| / max, Pearson r.

<!-- BLOCK_STUDY -->

{block_study}

{block_verdict}

**Reading.** Error grows monotonically with the block/radius ratio: b/r 0.19 → 4.5 % from exact (S),
0.16 → 2.0 % (M), and doubling b/r from 0.13 to 0.26 changes `cum_current` by 2.5 % (XL) to 9.5 %
(XXL). Extrapolating the anchors, staying within ≈ 1 % of the exact block-1 map needs **b/r ≤ 0.10**.

**Recommendation (adopted in `configs/datasets/v1_0.yaml`, `omniscape_choice: fidelity`):**
`block = largest odd integer ≤ radius/10` at every tier → S 1 (exact), M 3, L 5, XL 11, XXL 25
(b/r 0.06–0.10). Costs per T4 solve: 52 / 134 / 577 / 1 896 / 9 613 s (S/M/L/XL/XXL), i.e. ≈ 3.3× the
Phase-5 blocks overall (§6.1 vs §6.2). The Phase-5 blocks (b/r 0.13–0.19, `phase5`) remain in the
config as the cheaper alternative if the run location cannot afford the difference; the coarser
XL 33 / XXL 65 blocks are rejected. Whatever option is run, `block_size` and `radius` are recorded
per sample and must be identical across all samples of a tier.

## 8. Portability

Everything cluster-specific lives in `configs/cluster/<profile>.yaml` (`ice.yaml` and
`template.yaml` provided): scratch root, node temp dir, modules, partitions, account/QoS, per-tier
cpus/memory/walltime/shard size, concurrency limits. `scripts/generate.py submit --profile <name>`
reads it, precompiles on the login node and exports the values to the array job; `scripts/env.sh`
sets `JULIA_DEPOT_PATH`, `JULIA_CPU_TARGET`, `JULIA_PKG_PRECOMPILE_AUTO=0` and the Python/uv caches
under `EFB_SCRATCH`. The step-by-step guide for a new Slurm system is `docs/run_guide.md`.

## 9. Gap analysis (what Phases 2–5 cannot yet deliver)

| Gap | Status | Plan |
|---|---|---|
| `narrow_corridor` hard-case generator | not implemented (barriers exist, but gap geometry is random, not guaranteed narrow) | Phase 6: barrier-wall generator with 1–3 gaps of 1–3 px + test that all flow crosses the gaps |
| `large_nodata` prior extension (0.25–0.45) | prior caps NoData at 0.25 | Phase 6: `DEFAULT_PRIOR["nodata"]["fraction"]` upper bound per stratum |
| Real tiles at M/L/XL/XXL | pilot extracted tier S only; extractor supports any size but larger tiles need more COG windows (XXL 2048 km tile spans many 3° WorldCover cells) | Phase 6: run extraction per tier on the login node in prefetch/reproject stages; 32 XXL tiles is feasible |
| Cross-tier tile exclusion and per-tier seed ranges | not in the planner | Phase 6: planner reads `v1_0.yaml` and enforces §5 |
| Habitat-patch regions on ≥ 40 % of real tiles | 38 % on the pilot | acceptable; WDPA-based regions only if UNEP-WCMC permission is obtained |
| Random table per tile (seeded) | one shared random table in the pilot | Phase 6: `perturb_table(seed = hash(tile))` per tile |
| HDF5 shard schema + validator, splits, HF/Croissant | Phase 6–7 | as planned |
| Shard sync (validate → upload → verify → delete) | design only | Phase 6, target decided with the run location |
| 4-neighbour ablation path | solver flag exists, planner does not emit the duplicates | Phase 6 |
| Cross-machine bitwise determinism | verified on one machine only | document as solver-tolerance reproducibility (1e-11) |
| GLO-30 withheld tiles | GLO-90 fallback coded, never triggered | keep |

## 10. Recommended deviations from the brief (summary)

1. L 10k → 20k, XL 2k → 4k, XXL 200 → 400 (scale-OOD statistics), +12 200 landscapes, +≈ 1 200 CPU-h.
2. Named `hard` stratum (20 % of synthetic) with four defined hard cases, instead of relying on the prior's tail.
3. One seeded random resistance table **per tile** (decorrelation) rather than one global random table.
4. Floor of 150 S-tier real tiles per biome.
5. Omniscape block size chosen on fidelity (§7), block/radius ratio fixed across tiers.
6. QC residual threshold 1e-6 (double-precision floor), CHOLMOD for T4.
