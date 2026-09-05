# Phase 3 report — resistance surface construction

Date: 2026-09-05. Status: **complete, awaiting owner confirmation** before Phase 4 (focal nodes and
source configurations). Same summary in `docs/status/latest.md`.

## What was built

| Item | Location |
|---|---|
| YAML resistance-table schema (pydantic), covariates → resistance mapping with a fixed combination order, NoData handling | `ecoflowbench/resistance/tables.py` |
| Random perturbed tables (log-normal jitter, seeded, provenance recorded) | `ecoflowbench/resistance/random_table.py` |
| Five tables with literature citations in the headers | `configs/resistance_tables/{generic_hm,large_mammal,amphibian,forest_bird,random_lm_20260905}.yaml` + `README.md` |
| Build script: all tables × all accepted tiles → 2-band GeoTIFFs, `resistance.parquet` (stats + table sha256), summary JSON, gallery | `scripts/build_resistance.py` |
| Tests (11): schema validation and rejection, value ranges, mask consistency, determinism, each term's behaviour (roads, slope, water, elevation bands, gHM curve), random-table reproducibility, all tables on all pilot tiles | `tests/test_resistance.py` |

## What was verified

- Every table × every one of the 60 pilot tiles produces finite resistance in [1, r_max] with the
  NoData mask equal to the WorldCover NoData mask (no pilot tile has NoData pixels; masks are
  exercised by synthetic covariate tests including water-as-NoData).
- Term-level unit tests: highway pixels get exactly the class-1 penalty; slope multiplies by
  1 + 0.03·deg; water overrides to the barrier value; built-up = 500; NoData → 1.0 and masked;
  generic_hm reproduces 1 + 999·gHM² at 0/0.5/1 and treats NaN gHM as 0; forest_bird elevation
  bands give ×1/×2/×4.
- Random table: identical for the same seed, different for another seed, values within range,
  provenance (`base_table_id`, `seed`, `log_sd`, base sha256) stored in the YAML.
- Visual check of 4 tiles × 5 tables: `docs/figures/pilot_resistance_gallery.png` (roads, water
  barriers vs. permeable water for amphibians, forest = 1 for forest_bird all visible).

## Measured numbers (means over the 60 pilot tiles)

| table | r_min | r_max observed | median | frac at r_max | log10 contrast |
|---|---|---|---|---|---|
| amphibian | 6.26 | 765.97 | 38.88 | 0.000 | 2.45 |
| forest_bird | 3.88 | 56.28 | 17.47 | 0.050 | 1.5 |
| generic_hm | 34.38 | 678.86 | 72.23 | 0.020 | 1.54 |
| large_mammal | 14.97 | 664.68 | 28.2 | 0.000 | 1.77 |
| random_lm_20260905 | 15.91 | 704.33 | 31.15 | 0.000 | 1.75 |

- 300 rasters, 14 MB on scratch (deflate, 128² float32 × 2 bands).
- Source archives now 4.87 GB cumulative (18 files), 18 GB unzipped; all sha256 recorded.

## Observations

1. `forest_bird` saturates at r_max on up to 76 % of pixels for three Andean tiles above 3000 m
   (×4 elevation band on grass/bare). This is the intended "elevation-band effect" but it makes
   those tiles nearly uniform for this table; Phase 5 QC will flag samples with `frac_at_rmax > 0.5`
   so they can be excluded from training splits if they turn out degenerate.
2. `large_mammal` includes an additive 100·gHM term, so its floor is rarely 1 in modified
   landscapes (mean r_min ≈ 15). The dynamic range across tables (log10 contrast 1.5–2.5 on
   average, up to 3) already spans two of the four contrast levels; synthetic landscapes cover
   the rest.
3. All table values are EcoFlowBench's own choices with literature-based ordering (README in the
   table directory states this explicitly, as does `DECISIONS.md`), to avoid implying that
   any published species table was reproduced.

## Owner decisions applied this phase

- HF storage deferred to the Phase 5 gate (mini + dev subset < 100 GB until then).
- Download gate refined (single > 5 GB or cumulative > 20 GB); GRIP4 regions 4 and 6 and the
  remaining HydroRIVERS regions downloaded, unzipped and spatially indexed.
- Neighbour rule wording corrected in `DECISIONS.md` (average conductance *is* the Circuitscape
  default; explicit in presets and solver_params). GRIP4/OSM wording fixed in `docs/licenses.md`.
- WDPA: not downloaded; Phase 4 default is habitat-patch focal regions from land cover.

## Next step (Phase 4, on confirmation)

`ecoflowbench/sources/`: point pairs (K ∈ [2,8], min separation), wall-to-wall strips, habitat-
patch focal regions (WDPA-free default), advanced-mode source/ground rasters, Omniscape source
rasters, connectivity check against the resistance graph; tests on synthetic + pilot resistance.
