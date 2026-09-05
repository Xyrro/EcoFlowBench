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

### Landscapes per tier × family × stratum/table

|                                              |     S |    M |    L |   XL |   XXL |   total |
|:---------------------------------------------|------:|-----:|-----:|-----:|------:|--------:|
| ('real', 'tiles=1600', 'amphibian')          |     0 |    0 | 1600 |    0 |     0 |    1600 |
| ('real', 'tiles=1600', 'forest_bird')        |     0 |    0 | 1600 |    0 |     0 |    1600 |
| ('real', 'tiles=1600', 'generic_hm')         |     0 |    0 | 1600 |    0 |     0 |    1600 |
| ('real', 'tiles=1600', 'large_mammal')       |     0 |    0 | 1600 |    0 |     0 |    1600 |
| ('real', 'tiles=1600', 'random_lm')          |     0 |    0 | 1600 |    0 |     0 |    1600 |
| ('real', 'tiles=32', 'amphibian')            |     0 |    0 |    0 |    0 |    32 |      32 |
| ('real', 'tiles=32', 'forest_bird')          |     0 |    0 |    0 |    0 |    32 |      32 |
| ('real', 'tiles=32', 'generic_hm')           |     0 |    0 |    0 |    0 |    32 |      32 |
| ('real', 'tiles=32', 'large_mammal')         |     0 |    0 |    0 |    0 |    32 |      32 |
| ('real', 'tiles=32', 'random_lm')            |     0 |    0 |    0 |    0 |    32 |      32 |
| ('real', 'tiles=320', 'amphibian')           |     0 |    0 |    0 |  320 |     0 |     320 |
| ('real', 'tiles=320', 'forest_bird')         |     0 |    0 |    0 |  320 |     0 |     320 |
| ('real', 'tiles=320', 'generic_hm')          |     0 |    0 |    0 |  320 |     0 |     320 |
| ('real', 'tiles=320', 'large_mammal')        |     0 |    0 |    0 |  320 |     0 |     320 |
| ('real', 'tiles=320', 'random_lm')           |     0 |    0 |    0 |  320 |     0 |     320 |
| ('real', 'tiles=4000', 'amphibian')          |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=4000', 'forest_bird')        |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=4000', 'generic_hm')         |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=4000', 'large_mammal')       |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=4000', 'random_lm')          |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=8000', 'amphibian')          |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('real', 'tiles=8000', 'forest_bird')        |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('real', 'tiles=8000', 'generic_hm')         |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('real', 'tiles=8000', 'large_mammal')       |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('real', 'tiles=8000', 'random_lm')          |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('synthetic', 'distance_gradient', '-')      |  4800 | 2400 |  960 |  192 |    19 |    8371 |
| ('synthetic', 'edge_gradient', '-')          |  2400 | 1200 |  480 |   96 |    10 |    4186 |
| ('synthetic', 'fractal', '-')                |  9600 | 4800 | 1920 |  384 |    38 |   16742 |
| ('synthetic', 'grf', '-')                    | 14400 | 7200 | 2880 |  576 |    58 |   25114 |
| ('synthetic', 'hard:high_contrast_1e4', '-') |  3600 | 1800 |  720 |  144 |    14 |    6278 |
| ('synthetic', 'hard:large_nodata', '-')      |  3000 | 1500 |  600 |  120 |    12 |    5232 |
| ('synthetic', 'hard:narrow_corridor', '-')   |  3600 | 1800 |  720 |  144 |    14 |    6278 |
| ('synthetic', 'hard:rmax_saturated', '-')    |  1800 |  900 |  360 |   72 |     7 |    3139 |
| ('synthetic', 'mosaic', '-')                 |  7200 | 3600 | 1440 |  288 |    29 |   12557 |
| ('synthetic', 'planar_gradient', '-')        |  2400 | 1200 |  480 |   96 |    10 |    4186 |
| ('synthetic', 'random_cluster', '-')         |  7200 | 3600 | 1440 |  288 |    29 |   12557 |

The synthetic generator mix (24 % GRF, 16 % fractal, 12 % random cluster, 4 % planar, 4 % edge,
8 % distance gradient, 12 % mosaic, 20 % hard cases) is fixed across tiers so that every generator
appears at every tier; barrier and patch-mosaic overlays and NoData blobs are drawn from the
documented prior on top of every base generator (`DEFAULT_PRIOR`).

### 2.3 Solves per tier × family × source configuration (task instances)

### Solves per tier × family × source configuration (task instances)

|                                                         |     S |     M |     L |   XL |   XXL |   total |
|:--------------------------------------------------------|------:|------:|------:|-----:|------:|--------:|
| ('real', 'advanced', 'T3')                              | 40000 | 20000 |  8000 | 1600 |   160 |   69760 |
| ('real', 'omniscape', 'T4')                             | 40000 | 20000 |  8000 | 1600 |   160 |   69760 |
| ('real', 'points', 'T1,T2')                             | 40000 | 20000 |  8000 | 1600 |   160 |   69760 |
| ('real', 'regions', 'T1R')                              | 16000 |  8000 |  3200 |  640 |    64 |   27904 |
| ('real', 'wall_to_wall_EW', 'T1W')                      | 40000 | 20000 |  8000 | 1600 |   160 |   69760 |
| ('real', 'wall_to_wall_NS', 'T1W')                      | 40000 | 20000 |  8000 | 1600 |   160 |   69760 |
| ('synthetic', 'advanced', 'T3')                         | 60000 | 30000 | 12000 | 2400 |   240 |  104640 |
| ('synthetic', 'omniscape', 'T4')                        | 60000 | 30000 | 12000 | 2400 |   240 |  104640 |
| ('synthetic', 'points', 'T1,T2')                        | 60000 | 30000 | 12000 | 2400 |   240 |  104640 |
| ('synthetic', 'points (4-neighbour ablation)', 'T1,T2') |  3000 |     0 |     0 |    0 |     0 |    3000 |
| ('synthetic', 'wall_to_wall_EW', 'T1W')                 | 60000 | 30000 | 12000 | 2400 |   240 |  104640 |
| ('synthetic', 'wall_to_wall_NS', 'T1W')                 | 60000 | 30000 | 12000 | 2400 |   240 |  104640 |

Total landscapes: 174,400; total solves: 902,904; real tiles: {'L': 1600, 'M': 4000, 'S': 8000, 'XL': 320, 'XXL': 32}

Totals (recommended): **174,400 landscapes, 902,904 solves**; brief baseline: 162,200 landscapes, 839,952 solves.
Every landscape gets `points` (T1 + T2, K ∈ [2, 8]), both wall-to-wall strips (T1W), `advanced` (T3)
and `omniscape` (T4); real tiles with ≥ 2 eligible habitat patches (≈ 40 %) also get `regions` (T1R).
A 5 % slice of tier-S synthetic landscapes is additionally solved with the 4-neighbour graph and
flagged `graph_connectivity = 4` (brief §3.2 ablation).

### 2.4 Brief baseline ladder (for comparison)

### Landscapes per tier × family × stratum/table

|                                              |     S |    M |    L |   XL |   XXL |   total |
|:---------------------------------------------|------:|-----:|-----:|-----:|------:|--------:|
| ('real', 'tiles=16', 'amphibian')            |     0 |    0 |    0 |    0 |    16 |      16 |
| ('real', 'tiles=16', 'forest_bird')          |     0 |    0 |    0 |    0 |    16 |      16 |
| ('real', 'tiles=16', 'generic_hm')           |     0 |    0 |    0 |    0 |    16 |      16 |
| ('real', 'tiles=16', 'large_mammal')         |     0 |    0 |    0 |    0 |    16 |      16 |
| ('real', 'tiles=16', 'random_lm')            |     0 |    0 |    0 |    0 |    16 |      16 |
| ('real', 'tiles=160', 'amphibian')           |     0 |    0 |    0 |  160 |     0 |     160 |
| ('real', 'tiles=160', 'forest_bird')         |     0 |    0 |    0 |  160 |     0 |     160 |
| ('real', 'tiles=160', 'generic_hm')          |     0 |    0 |    0 |  160 |     0 |     160 |
| ('real', 'tiles=160', 'large_mammal')        |     0 |    0 |    0 |  160 |     0 |     160 |
| ('real', 'tiles=160', 'random_lm')           |     0 |    0 |    0 |  160 |     0 |     160 |
| ('real', 'tiles=4000', 'amphibian')          |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=4000', 'forest_bird')        |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=4000', 'generic_hm')         |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=4000', 'large_mammal')       |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=4000', 'random_lm')          |     0 | 4000 |    0 |    0 |     0 |    4000 |
| ('real', 'tiles=800', 'amphibian')           |     0 |    0 |  800 |    0 |     0 |     800 |
| ('real', 'tiles=800', 'forest_bird')         |     0 |    0 |  800 |    0 |     0 |     800 |
| ('real', 'tiles=800', 'generic_hm')          |     0 |    0 |  800 |    0 |     0 |     800 |
| ('real', 'tiles=800', 'large_mammal')        |     0 |    0 |  800 |    0 |     0 |     800 |
| ('real', 'tiles=800', 'random_lm')           |     0 |    0 |  800 |    0 |     0 |     800 |
| ('real', 'tiles=8000', 'amphibian')          |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('real', 'tiles=8000', 'forest_bird')        |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('real', 'tiles=8000', 'generic_hm')         |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('real', 'tiles=8000', 'large_mammal')       |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('real', 'tiles=8000', 'random_lm')          |  8000 |    0 |    0 |    0 |     0 |    8000 |
| ('synthetic', 'distance_gradient', '-')      |  4800 | 2400 |  480 |   96 |    10 |    7786 |
| ('synthetic', 'edge_gradient', '-')          |  2400 | 1200 |  240 |   48 |     5 |    3893 |
| ('synthetic', 'fractal', '-')                |  9600 | 4800 |  960 |  192 |    19 |   15571 |
| ('synthetic', 'grf', '-')                    | 14400 | 7200 | 1440 |  288 |    29 |   23357 |
| ('synthetic', 'hard:high_contrast_1e4', '-') |  3600 | 1800 |  360 |   72 |     7 |    5839 |
| ('synthetic', 'hard:large_nodata', '-')      |  3000 | 1500 |  300 |   60 |     6 |    4866 |
| ('synthetic', 'hard:narrow_corridor', '-')   |  3600 | 1800 |  360 |   72 |     7 |    5839 |
| ('synthetic', 'hard:rmax_saturated', '-')    |  1800 |  900 |  180 |   36 |     4 |    2920 |
| ('synthetic', 'mosaic', '-')                 |  7200 | 3600 |  720 |  144 |    14 |   11678 |
| ('synthetic', 'planar_gradient', '-')        |  2400 | 1200 |  240 |   48 |     5 |    3893 |
| ('synthetic', 'random_cluster', '-')         |  7200 | 3600 |  720 |  144 |    14 |   11678 |

## 3. Hard-case stratum (named `hard`, 20 % of synthetic landscapes at every tier)

| hard case | share of `hard` | definition | landscapes (recommended, all tiers) |
|---|---|---|---|
| `high_contrast_1e4` | 30 % | contrast 10⁴ from any base generator | 6,278 |
| `rmax_saturated` | 15 % | > 50 % of valid pixels at r_max (near-degenerate); QC-flagged, train/val-excluded, kept in the index | 3,139 |
| `narrow_corridor` | 30 % | 1–3 barrier walls spanning the raster with 1–3 gaps of 1–3 px each (all flow funnels through the gaps) | 6,278 |
| `large_nodata` | 25 % | NoData fraction 0.25–0.45, single connected component | 5,232 |

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
(S/M/L/XL/XXL); T4 6.2 / 48 / 178 / 794 / 4 468 s at the Phase-5 block sizes (S b3 … XL b17, XXL b33);
T1W 0.08 s, T3 0.07 s, T1R 0.21 s at S, scaling linearly in pixels (exponents 1.00–1.07).
Peak RSS (process high-water incl. ≈ 1 GB Julia baseline): 1.4 / 1.4 / 1.6 / 3.3 / 9.5 GB.
Compressed storage per landscape (all configs): synthetic 0.93 / 3.0 / 11 / 41 / 140 MB, real ≈ 1.5×.

### 6.1 Recommended ladder

### Cost (measured Phase 5, CHOLMOD, single-threaded solves, +15 % overhead)

| tier   |   landscapes |   cpu_s_per_landscape |   peak_rss_gb |   cpu_hours |   storage_gb |
|:-------|-------------:|----------------------:|--------------:|------------:|-------------:|
| S      |       100000 |                   7.6 |           1.4 |         212 |        111.8 |
| M      |        50000 |                  57.8 |           1.4 |         802 |        180   |
| L      |        20000 |                 214.3 |           1.6 |        1191 |        260   |
| XL     |         4000 |                 948   |           3.3 |        1053 |        194.4 |
| XXL    |          400 |                5298.1 |           9.5 |         589 |         65.6 |
| total  |       174400 |                       |               |        3847 |        811.8 |

### Wall-clock for the full ladder

|   concurrent_cores |   wall_days |
|-------------------:|------------:|
|                100 |         1.6 |
|                500 |         0.3 |
|               1000 |         0.2 |

### 6.2 Brief baseline ladder

### Cost (measured Phase 5, CHOLMOD, single-threaded solves, +15 % overhead)

| tier   |   landscapes |   cpu_s_per_landscape |   peak_rss_gb |   cpu_hours |   storage_gb |
|:-------|-------------:|----------------------:|--------------:|------------:|-------------:|
| S      |       100000 |                   7.6 |           1.4 |         212 |        111.8 |
| M      |        50000 |                  57.8 |           1.4 |         802 |        180   |
| L      |        10000 |                 214.3 |           1.6 |         595 |        130   |
| XL     |         2000 |                 948   |           3.3 |         527 |         97.2 |
| XXL    |          200 |                5298.1 |           9.5 |         294 |         32.8 |
| total  |       162200 |                       |               |        2430 |        551.8 |

### Wall-clock for the full ladder

|   concurrent_cores |   wall_days |
|-------------------:|------------:|
|                100 |         1   |
|                500 |         0.2 |
|               1000 |         0.1 |

CPU-hours are *core-hours of single-threaded solving* (+15 % overhead). Because solves are
single-threaded, wall-clock at N concurrent cores = CPU-hours / N; memory (not cores) sets the
allocation for XL/XXL (9.5 GB per XXL job). T4 is ≈ 90 % of every total. Omniscape block sizes are
fixed on fidelity (§7); if the study supports coarser blocks at XL/XXL the totals fall by ≈ 25 %.

## 7. Omniscape block size (fidelity study)

<!-- BLOCK_STUDY -->
*Running (3 XL samples b17 vs b33, 3 XXL samples b33 vs b65); results and the recommended
block/radius ratio are inserted here when the jobs finish.*

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
