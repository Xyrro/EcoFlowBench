# Phase 7 report — loaders and Hugging Face integration

Date: 2026-09-06. Status: **code complete; the real push to `Xirro/AmpScape` is blocked on the login-node
Hugging Face token** (`hf auth whoami` → "Not logged in"). Same summary in `docs/status/latest.md`.

## Pre-Phase-7 check: macro-cells and latitude

The owner's concern was confirmed and fixed. Under the previous 20° lon/lat grid, **0 %** of random XXL
centres fit inside a cell at any latitude (2 226 km cells vs 2 048 km tiles leave no margin even at
the equator) and XL fit 27–58 %. Equal-width 30° bands would still confine XXL to |lat| 9–21°, 39–51°,
69–81°. Adopted: an **equal-width 20° grid** (104 cells, ≈ 2 200 km wide at every latitude) for the
assignment, with **XXL tiles as their own regions** (merged when overlapping) and inheritance for
contained tiles; straddling tiles are excluded and resampled. Placeability is now uniform in
latitude (XXL 100 %, S–L ≥ 88 %, XL 65–86 % of random centres), and the leakage test is geometric
(0 train/val–test box intersections over 1 640 random tiles, 5 tiers). Details in `docs/dataset_plan.md`
§5; `DECISIONS.md`.

## What was built

| Item | Location |
|---|---|
| `AmpScapeDataset(task, split, tier, root, subset, ood, normalize)` over builds or the HF layout; lazy per-process shard handles (DataLoader-safe); tensors per task (T1/T1W/T1R/T2/T3/T4); `torch()` adapter; `log1p_targets`; `compute_norm_stats` (train only) → `stats/norm_stats.json`; `load_from_hub` (pattern download of one tier × task group) | `ampscape/data/dataset.py` |
| HF layout export: per-tier × task-group shards (inputs duplicated, one configuration per shard), `index/<tier>.parquet` with `task_group`, `hf_path`, `subset_*`, nested `splits/<subset>/<split>.parquet` | `ampscape/io/hf_layout.py` |
| Subsets mini ⊂ core ⊂ full (hash-stratified by tier × family × split) | `ampscape/data/subsets.py` |
| Croissant 1.0 + RAI export, validated with `mlcroissant` (0 errors, 0 warnings) | `scripts/export_croissant.py` |
| Push script: create private repo, upload card/croissant/index/splits/stats, shard uploads sha256-verified against the Hub, `.uploaded` markers, manifest; dry-run default | `scripts/push_to_hub.py` |
| Dataset card draft with the honest framing | `docs/dataset_card.md` |
| Tests (4): loader contracts per task, split/QC/OOD filters + torch, train-only stats, layout export + subsets + loading from the layout | `tests/test_data_loader.py` |

## Mini release staged (`data/hf/AmpScape`)

38 files, 300 MB: `data/S/{T1,T1R,T1W,T3,T4}/shard-0000{0..4}.h5` (T1 56 MB, T1W 87 MB, T3 36 MB, T4 49 MB,
T1R 8 MB), `index/S.parquet` (1 270 rows), `splits/{mini,core,full}/*.parquet`, `stats/norm_stats.json`
(log-resistance mean 2.60, std 1.78 over 2.2 M train pixels), `croissant.json`, `README.md`.
Dry run of the push lists all files; `load_from_hub("T4", "S")` will fetch 49 MB + index only.

## ICE feasibility (owner question)

Recorded in `docs/compute_env.md` §8.2: no per-user CPU cap, `MaxSubmitPU = 500`, group cap 2 536 cores,
≈ 1 290 idle cores at inspection; 11 300 core-hours finish in ≈ 23 h at 500 concurrent 1-core jobs
or ≈ 4.7 days at 100 — technically well within two weeks. The blockers are policy (ICE = pipeline/
mini/dev; 500 CPU-h gate), the 300 GB scratch (needs the streaming sync), and etiquette.

## Blocked / next

- **Real push**: run `hf auth login` (write token) on the login node, then
  `python scripts/push_to_hub.py --layout data/hf/AmpScape --push`, followed by the
  `load_from_hub` download-alone check and the sync delete test on the smoke build.
- Phase 8 (licences/provenance) on confirmation.
