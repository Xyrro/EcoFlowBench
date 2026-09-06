# AmpScape v1.0 dataset plan (compute-agnostic)

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

|                                              |     S |    M |    L |   XL |   XXL |   total |
|:---------------------------------------------|------:|-----:|-----:|-----:|------:|--------:|
| ('real', '(distinct tiles)', '-')            |  8000 | 4000 | 1600 |  320 |    32 |   13952 |
| ('real', 'tile × table', 'amphibian')        |  8000 | 4000 | 1600 |  320 |    32 |   13952 |
| ('real', 'tile × table', 'forest_bird')      |  8000 | 4000 | 1600 |  320 |    32 |   13952 |
| ('real', 'tile × table', 'generic_hm')       |  8000 | 4000 | 1600 |  320 |    32 |   13952 |
| ('real', 'tile × table', 'large_mammal')     |  8000 | 4000 | 1600 |  320 |    32 |   13952 |
| ('real', 'tile × table', 'random_lm')        |  8000 | 4000 | 1600 |  320 |    32 |   13952 |
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

Total landscapes: 174,400; total solves: 902,904; real tiles: {'S': 8000, 'M': 4000, 'L': 1600, 'XL': 320, 'XXL': 32}

Totals (recommended): **174,400 landscapes, 902,904 solves**; brief baseline: 162,200 landscapes, 839,952 solves.
Every landscape gets `points` (T1 + T2, K ∈ [2, 8]), both wall-to-wall strips (T1W), `advanced` (T3)
and `omniscape` (T4); real tiles with ≥ 2 eligible habitat patches (≈ 40 %) also get `regions` (T1R).
A 5 % slice of tier-S synthetic landscapes is additionally solved with the 4-neighbour graph and
flagged `graph_connectivity = 4` (brief §3.2 ablation).

### 2.4 Brief baseline ladder (for comparison)

|                                              |     S |    M |    L |   XL |   XXL |   total |
|:---------------------------------------------|------:|-----:|-----:|-----:|------:|--------:|
| ('real', '(distinct tiles)', '-')            |  8000 | 4000 |  800 |  160 |    16 |   12976 |
| ('real', 'tile × table', 'amphibian')        |  8000 | 4000 |  800 |  160 |    16 |   12976 |
| ('real', 'tile × table', 'forest_bird')      |  8000 | 4000 |  800 |  160 |    16 |   12976 |
| ('real', 'tile × table', 'generic_hm')       |  8000 | 4000 |  800 |  160 |    16 |   12976 |
| ('real', 'tile × table', 'large_mammal')     |  8000 | 4000 |  800 |  160 |    16 |   12976 |
| ('real', 'tile × table', 'random_lm')        |  8000 | 4000 |  800 |  160 |    16 |   12976 |
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

- **Hierarchical spatial regions shared across tiers (owner amendment C1, revised 2026-09-06).** Tiers
  are multi-resolution, so S/M tiles physically sit inside XL/XXL tiles, and a fixed lon/lat grid cannot
  hold an XXL tile outside the tropics (20° cells are 1 570 km wide at 45° latitude; measured: 0 % of
  random XXL centres fit, and equal-width 30° bands still confine XXL to three latitude strips). The
  adopted scheme: (i) an **equal-width grid** — 20° latitude bands with n ≈ 360·cos(lat)/20 longitude
  cells per band (104 cells, ≈ 2 200 km wide at every latitude) — receives one seeded assignment per
  cell (`train`/`val`/`test_id`/`ood_region`, stratified by realm); (ii) **XXL tiles are their own
  assignment regions** (overlapping XXL footprints merged, assigned unstratified from the same seed);
  (iii) every smaller tile inherits the assignment of the XXL footprint that fully contains it, or
  otherwise of the single grid cell that contains it; (iv) tiles straddling two regions are excluded
  and resampled by the sampler (`BlockGrid.interior_bounds` gives the sampler the admissible centre
  box). Leakage is checked **geometrically**: no train/val tile's bounding box intersects any test/OOD
  tile's box at any tier (`tests/test_splits.py`, 1 640 random tiles, 0 overlaps). Placeability is
  uniform in latitude: XXL 100 %, S–L ≥ 88 %, XL 65–86 % of random centres (before sampler placement).
  Held-out biomes / realms (`test_ood_region`) are whole cells; XXL parents centred in them are OOD.
- **Unit for synthetic landscapes = seed family.** A base seed defines the landscape; its 4-neighbour
  ablation duplicate and any hard-case variant derived from it inherit the split. Seed ranges are
  tier-disjoint.
- **XL (owner amendment C3):** 25 % of XL landscapes (by block assignment) are train/val so that models
  can be trained at XL; `test_ood_scale` is defined as the XL/XXL test landscapes evaluated by models
  trained on tiers ≤ L only, while the in-distribution XL test for XL-trained models is `test_id` at XL.
  XXL remains test-only.
- Split assignment is a deterministic function of (block id, dataset seed), so re-planning reproduces
  the same splits; the lists ship as Parquet (`splits/<subset>/<split>.parquet`), nested so that the
  mini splits are subsets of core, and core of full (owner amendment C4).

### 5.1 Resampling rules per source layer (owner amendment C2)

Recorded in every sample's provenance (`meta.resampling`) with the source resolution:

| channel | source | rule |
|---|---|---|
| landcover | WorldCover 10 m | **majority** class over the target pixel footprint |
| elevation | Copernicus DEM 30 m | **mean** (bilinear at 100 m, area mean at ≥ 200 m); slope derived on the target grid |
| road_distance / river_distance | GRIP4 / HydroRIVERS vectors | rasterise at the target grid (all touched) then Euclidean distance = **min** distance |
| road_class / river_order | vectors | attribute of the **nearest feature** |
| ghm | gHM 1 km | **mean** (bilinear at ≤ 1 km, area mean above) |

The Phase 2 extractor implements the S-tier (100 m) rules; majority/area-mean resampling for
M–XXL is a Phase 6 change to `landscapes/real.py` (GDAL mode/average).

### 5.2 Download subsets (owner amendment C4)

| subset | tiers | size | content |
|---|---|---|---|
| `mini` | S | ≈ 0.5 GB | ≈ 500 landscapes, all tasks and families, every split represented |
| `core` | S, M, L | ≈ 50 GB | stratified subsample, all tasks/families/tables/generators, all OOD sets |
| `full` | S–XXL | ≈ 810 GB | everything |

Splits are fixed and nested (mini ⊂ core ⊂ full). HF layout: `data/<tier>/<task_group>/shard-NNNNN.h5`,
`index/<tier>.parquet`, `splits/<subset>/<split>.parquet`, so any single tier or task group can be
downloaded alone (`load_dataset("Xirro/AmpScape", "T1_S")`).

### 5.3 T1R on synthetic landscapes (owner amendment C5)

Focal regions are also generated on synthetic **patch-mosaic** landscapes (the `mosaic` generator and
`patch_mosaic` overlays): patches of the lowest-cost class ≥ `min_patch_px` act as habitat patches,
using the same `sample_regions` code path as real tiles.

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

| tier   |   landscapes |   cpu_s_per_landscape |   peak_rss_gb |   cpu_hours |   storage_gb |
|:-------|-------------:|----------------------:|--------------:|------------:|-------------:|
| S      |       100000 |                  60.9 |           1.4 |        1691 |        111.8 |
| M      |        50000 |                 156.3 |           1.4 |        2171 |        180   |
| L      |        20000 |                 673.2 |           1.6 |        3740 |        260   |
| XL     |         4000 |                2215.3 |           3.3 |        2461 |        194.4 |
| XXL    |          400 |               11214.9 |           9.5 |        1246 |         65.6 |
| total  |       174400 |                       |               |       11309 |        811.8 |

|   concurrent_cores |   wall_days |
|-------------------:|------------:|
|                100 |         4.7 |
|                500 |         0.9 |
|               1000 |         0.5 |

### 6.2 Recommended ladder, Phase-5 Omniscape blocks (b/r 0.13–0.19) for comparison

| tier   |   landscapes |   cpu_s_per_landscape |   peak_rss_gb |   cpu_hours |   storage_gb |
|:-------|-------------:|----------------------:|--------------:|------------:|-------------:|
| S      |       100000 |                   7.6 |           1.4 |         212 |        111.8 |
| M      |        50000 |                  57.8 |           1.4 |         802 |        180   |
| L      |        20000 |                 214.3 |           1.6 |        1191 |        260   |
| XL     |         4000 |                 948   |           3.3 |        1053 |        194.4 |
| XXL    |          400 |                6504.5 |           9.5 |         723 |         65.6 |
| total  |       174400 |                       |               |        3981 |        811.8 |

|   concurrent_cores |   wall_days |
|-------------------:|------------:|
|                100 |         1.7 |
|                500 |         0.3 |
|               1000 |         0.2 |

### 6.3 Brief baseline ladder (fidelity blocks)

| tier   |   landscapes |   cpu_s_per_landscape |   peak_rss_gb |   cpu_hours |   storage_gb |
|:-------|-------------:|----------------------:|--------------:|------------:|-------------:|
| S      |       100000 |                  60.9 |           1.4 |        1691 |        111.8 |
| M      |        50000 |                 156.3 |           1.4 |        2171 |        180   |
| L      |        10000 |                 673.2 |           1.6 |        1870 |        130   |
| XL     |         2000 |                2215.3 |           3.3 |        1231 |         97.2 |
| XXL    |          200 |               11214.9 |           9.5 |         623 |         32.8 |
| total  |       162200 |                       |               |        7586 |        551.8 |

|   concurrent_cores |   wall_days |
|-------------------:|------------:|
|                100 |         3.2 |
|                500 |         0.6 |
|               1000 |         0.3 |

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

| tier   |   radius | blocks   | b/r            | sample   |   rel_L2_cum |   max_diff/max_cum |   pearson_cum |   rel_L2_normalized |   t_fine_s |   t_coarse_s |
|:-------|---------:|:---------|:---------------|:---------|-------------:|-------------------:|--------------:|--------------------:|-----------:|-------------:|
| S      |       16 | 3 vs 1   | 0.188 vs 0.062 | 0012f547 |       0.0292 |             0.0525 |        0.9995 |              0.0427 |          8 |           52 |
| S      |       16 | 3 vs 1   | 0.188 vs 0.062 | 04a5dd84 |       0.076  |             0.3004 |        0.9951 |              0.0318 |          9 |           55 |
| S      |       16 | 3 vs 1   | 0.188 vs 0.062 | 04e6bec1 |       0.0294 |             0.0721 |        0.9995 |              0.0361 |          8 |           52 |
| M      |       32 | 5 vs 1   | 0.156 vs 0.031 | 231ba336 |       0.0172 |             0.0923 |        0.9998 |              0.0275 |         37 |          810 |
| M      |       32 | 5 vs 1   | 0.156 vs 0.031 | 23f6aecc |       0.0245 |             0.1573 |        0.9874 |              0.0041 |         67 |         1548 |
| M      |       32 | 5 vs 1   | 0.156 vs 0.031 | 253f5565 |       0.0184 |             0.1257 |        0.9998 |              0.1373 |         28 |          630 |
| XL     |      128 | 17 vs 33 | 0.133 vs 0.258 | 21041fe6 |       0.0223 |             0.1404 |        0.9996 |              0.0646 |        917 |          257 |
| XL     |      128 | 17 vs 33 | 0.133 vs 0.258 | b26cfabc |       0.0293 |             0.1598 |        0.9992 |              0.0752 |        998 |          286 |
| XL     |      128 | 17 vs 33 | 0.133 vs 0.258 | efc8cecf |       0.0225 |             0.1657 |        0.9788 |              0.0063 |       1692 |          447 |
| XXL    |      256 | 33 vs 65 | 0.129 vs 0.254 | 3e481cda |       0.1061 |             0.1369 |        0.9921 |              0.0522 |       7488 |         1975 |
| XXL    |      256 | 33 vs 65 | 0.129 vs 0.254 | a2818d9e |       0.1185 |             0.2391 |        0.9559 |              0.0922 |       6565 |         1657 |
| XXL    |      256 | 33 vs 65 | 0.129 vs 0.254 | a64b2605 |       0.0591 |             0.2435 |        0.9972 |              0.065  |       4274 |         1143 |

- **S anchor, block 3 vs 1 (radius 16, b/r 0.19 vs 0.06)**: the standard block 3 deviates from block 1 by 4.49% mean / 7.60% max relative L2 (Pearson ≥ 0.9951); block 1 costs 6.4× more.
- **M anchor, block 5 vs 1 (radius 32, b/r 0.16 vs 0.03)**: the standard block 5 deviates from block 1 by 2.00% mean / 2.45% max relative L2 (Pearson ≥ 0.9874); block 1 costs 22.6× more.
- **XL coarsening, block 17 vs 33 (radius 128, b/r 0.13 vs 0.26)**: relative L2 2.47% mean / 2.93% max (Pearson ≥ 0.9788); block 33 is 3.6× cheaper → NOT negligible (≥ 1 %): rejected.
- **XXL coarsening, block 33 vs 65 (radius 256, b/r 0.13 vs 0.25)**: relative L2 9.46% mean / 11.85% max (Pearson ≥ 0.9559); block 65 is 3.8× cheaper → NOT negligible (≥ 1 %): rejected.

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
under `AMPSCAPE_SCRATCH`. The step-by-step guide for a new Slurm system is `docs/run_guide.md`.

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
