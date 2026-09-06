---
license: cc-by-4.0
pretty_name: AmpScape
task_categories: [image-to-image, other]
tags: [landscape-connectivity, circuitscape, omniscape, surrogate-modeling, neural-operators, scientific-ml, ecology]
size_categories: [n<1K]
configs: []   # filled by scripts/push_to_hub.py from the layout (one config per tier × task group)
---

# AmpScape (dataset card — draft, mini release)

AmpScape is a benchmark of **circuit-theoretic landscape connectivity** solved with the reference
solvers Circuitscape.jl 5.17.1 and Omniscape.jl 0.6.2, for training and fairly comparing learned
surrogates. Each sample pairs a resistance raster and a source configuration with the exact solver
outputs (current-density maps, voltage maps, effective resistances, omnidirectional connectivity).

**Honest framing.** (1) The *solver is the ground truth*: outputs are stored raw (float32 maps,
float64 effective resistances), never normalised, clipped or post-processed; every sample records
solver versions, parameters, timings and residuals. (2) *Real landscapes, synthesized resistance*:
real tiles are genuine covariate stacks (ESA WorldCover, Copernicus DEM, GRIP4 roads, HydroRIVERS,
gHM), but the resistance surfaces are produced by resistance tables whose class ordering and term
structure follow the literature (Zeller et al. 2012; Cushman et al. 2006; Koen et al. 2014; Spear et
al. 2010; Brennan et al. 2022) while the **numeric values are AmpScape's own** — they are not
species-calibrated and must not be read as ecological truth for any taxon. (3) Synthetic landscapes
come from neutral landscape models and random fields with documented priors. (4) Omniscape is run
with `block_size = largest odd ≤ radius/10`, a deliberate, documented approximation of the
per-pixel (block 1) Omniscape whose fidelity was measured (≈ 2–5 % relative L2 at coarser blocks).

## Dataset structure

- `data/<tier>/<task_group>/shard-NNNNN.h5` — HDF5 shards, one group per sample, inputs + one
  configuration each (`T1` = pairwise current/Reff, `T1W` = wall-to-wall, `T1R` = habitat-patch
  regions, `T3` = advanced source/ground, `T4` = Omniscape). Any single tier or task group can be
  downloaded alone. Schema: `docs/schema.md` in the code repository.
- `index/<tier>.parquet` — one row per (sample, configuration): identifiers, family, generator or
  resistance table, tile, K, placement, solver, timings, residuals, QC flags, split, OOD flags,
  subset membership (`subset_mini`, `subset_core`, `subset_full`).
- `splits/<subset>/<split>.parquet` — sample ids; subsets are nested (mini ⊂ core ⊂ full).
- `stats/norm_stats.json` — train-only normalisation statistics.
- `croissant.json` — Croissant 1.0 metadata (core + RAI fields).

Tiers: S 128² (100 m), M 256² (100 m), L 512² (200 m), XL 1024² (500 m), XXL 2048² (1 km).
This mini release: tier S only, 250 landscapes (200 synthetic, 50 real), 1 270 solved configurations.

## Splits and leakage

Real tiles are assigned by spatial regions shared across tiers (equal-width 20° cells plus XXL
footprints as their own regions), so no test region at any resolution overlaps a training region at
another; synthetic landscapes by seed family. OOD sets: held-out biomes/realms, held-out resistance
table (`forest_bird`), held-out contrast (10⁴), held-out scale (XL/XXL for models trained ≤ L), and a
synthetic→real flag. **Pilot caveat:** the mini's 50 real tiles over-represent the held-out regions
(20 of 50) because the Phase 2 pilot sampled those strata for coverage; this is not a v1.0 property.

## Collection process, preprocessing, uses, limitations, ethics, maintenance

To be completed for the full release from `docs/dataset_plan.md`, `docs/licenses.md` (upstream
attributions and the Copernicus WorldDEM-30 notices, which are reproduced verbatim in `LICENSE-DATA`)
and the generation statistics. Protected-area (WDPA) data are **not** used. Locations of real tiles
are public land-cover/terrain products at ≥ 100 m; no personal data.

## Citation

See `CITATION.cff`. Code: https://github.com/Xyrro/AmpScape (MIT). Data: CC BY 4.0 with upstream
attributions.
