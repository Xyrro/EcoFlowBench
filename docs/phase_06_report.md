# Phase 6 report — data format, schema, validator, splits (on the regenerated mini)

Date: 2026-09-06. Status: **complete, awaiting owner confirmation** before Phase 7 (loaders + HF integration).
Same summary in `docs/status/latest.md`.

## What was built

| Item | Location |
|---|---|
| HDF5 shard schema v0.2: `MetaModel` (pydantic, all §4.3 metadata fields), per-kind input/output dtype & shape contracts, `validate_shard` (structure, dtypes, shapes, value ranges, finiteness, metadata, focal/Reff consistency), `docs/schema.md` rendered from the definitions | `ampscape/io/schema.py`, `docs/schema.md` |
| Optional Zarr export (same layout, attrs, per-array chunks) | `ampscape/io/zarr_export.py` |
| Streaming sync: validate → upload → sha256-verify against the Hub's LFS object → delete locally; `.ok` / `.uploaded` / `.invalid` markers; local-storage soft/hard limits; **dry-run by default, pushes owner-gated** (owner decision E) | `ampscape/io/sync.py`, `scripts/sync_shards.py` |
| Splits: 20° macro-cell assignment shared across tiers (one seeded assignment per cell, stratified by realm), straddling tiles excluded, footprint rule + fix-point propagation as safety net; synthetic splits by seed family; held-out table/contrast demotion; OOD flags; XL 25 % train/val, XXL test-only; `splits/<split>.parquet` + `README.json`; index columns `split`, `block_id`, `test_ood_*` | `ampscape/splits/spatial.py`, `ampscape/splits/assign.py`, `generate.py finalize` |
| Validator wired into `finalize` (writes `.ok` markers), index rows carry tile centre / pixel size | `scripts/generate.py`, `ampscape/solve/finalize.py` |
| Tests: 4 split tests (block ids/footprints incl. an XXL tile inside one cell, stratified determinism, **zero cross-tier overlap** on 1 640 random tiles over 5 tiers with kept proportions ≈ 78/8/10, seed-family + holdouts), 4 schema tests (valid shard passes; corrupted dtype / range / missing output / non-finite / metadata are caught; schema.md renders; Zarr round-trip) | `tests/test_splits.py`, `tests/test_schema.py` |
| Omniscape project-directory litter fixed (per-solve temp dir), per-job work dirs on node-local `$TMPDIR` or `$AMPSCAPE_SCRATCH/cache/<job>` | `julia/AmpScapeSolve.jl`, `scripts/generate.py` |

## Regenerated mini (owner decision D): radius/10 rule, S block 1, version `0.2.0-mini`

- 250 samples (200 synthetic, 50 real), 1 270 configurations, 5 shards, **232 MB**, all 5 shards
  **schema-valid**, `.ok` markers written (sync dry run lists them under `data/S/all/`).
- QC: **100 % pass**; one contrast-10⁴ landscape flagged `rmax_saturated` (kept, train/val-excluded).
- Solve times at S with block 1 (CHOLMOD): T4 median **53.7 s** (p90 75 s, max 105 s) vs 6.2 s with
  block 3; T1 0.19 s, T1W 0.09 s, T3 0.08 s, T1R 0.24 s. Per landscape 54 s median → 5 shards of 50 in
  48–53 min wall each (1 core, 8 GB). Peak RSS 2.3 GB (block 1 keeps more windows).
- The Phase-5 mini is kept as `data/builds/mini_phase5blocks` with `SUPERSEDED.txt`.
- Quicklooks: 250 PNGs; contact sheet `docs/figures/mini_contact_sheet.png`.

### Splits on the mini (seed 20260906, 20° cells)

| family | train | val | test_id | test_ood (table / contrast holdout) | ood_region |
|---|---|---|---|---|---|
| synthetic | 121 | 21 | 15 | 43 | 0 |
| real | 19 | 2 | 1 | 8 | 20 |

OOD flags (samples): region 20, table (`forest_bird`) 10, contrast (10⁴) 46, synth→real 1, scale 0 (no XL/XXL in mini).
The mini's 50 real tiles fall in 30 macro-cells; 20 of them are OOD-region because the pilot over-samples
Australasia and the held-out biomes — expected for a 50-tile pilot, not for v1.0 (§4 of the plan: ≈ 12 %).

## Findings

1. **5° blocks do not work with multi-resolution tiles**: with the footprint cascade, straddling XL/XXL
   tiles turned most of the globe into test (L: 317 test vs 71 train on the synthetic check). The
   adopted 20° macro-cells with sampler-side exclusion give 0 overlaps and ≈ 78/8/10 (`DECISIONS.md`).
2. **Block 1 at S costs 8.7× the Phase-5 T4 time** (53.7 vs 6.2 s), consistent with the fidelity study
   (52 s) and the plan's cost table (11 300 CPU-h for v1.0).
3. Omniscape's `project_name` directory is created in the cwd even without outputs — the source of
   505 empty directories in the repo root; fixed and verified (owner item).

## Gaps carried to later phases

- `MetaModel.resampling` is optional until the M–XXL extractor records it (Phase 6 real-tile work
  deferred: real tiles exist at S only; the resampling rules are specified in the plan §5.1).
- Parquet index columns for the download subsets (mini/core/full membership) come with Phase 7.
- Real-tile sampler must enforce the macro-cell straddling constraint (`split == "excluded"` today).
- Zarr export is a plain copy (no consolidated metadata); fine for the optional deliverable.

## Next step (Phase 7, on confirmation)

`ampscape.data.AmpScapeDataset(task, split, tier, root)` with lazy shard loading, normalisation stats on
train only, HF dataset layout by tier / task group, `scripts/push_to_hub.py --dry-run`, Croissant export,
and the dataset card — no push without confirmation.
