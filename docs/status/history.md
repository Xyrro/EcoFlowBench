# Status history

Newest entries at the bottom. Each entry mirrors `docs/status/latest.md` at the time of a stop.

---

# Status — 2026-09-05 (end of Phase 1)

Phase 1 complete (cluster inspected, toolchain isolated on scratch, scaffold, solver smoke test, prior art, task spec).
Blockers raised: 300 GB scratch vs v1.0 size; NeurIPS 2026 deadline passed; spec open questions (per-pair maps, CRS, OSM, Omniscape r/b, HF org).
Report: `docs/phase_01_report.md`.

---

# Status — 2026-09-05 (end of Phase 2)

**Phase 2 complete; waiting for confirmation to start Phase 3.**

- Synthetic generators done (7 families + overlays + NoData + contrast ladder), 38 tests, gallery in `docs/figures/synthetic_gallery.png`.
- Real-tile pipeline done; pilot extracted **60 tiles, 14 biomes, 5 realms** at tier S (128 px @ 100 m, local UTM), manifest + checksums, quicklooks. Map: `docs/figures/pilot_tiles_map.png`.
- Sources verified and downloaded (2.76 GB, sha256 manifest): WorldCover 2021 v200, Copernicus DEM GLO-30/90 (windowed COG reads), GRIP4 (regions 1,2,3,5,7), HydroRIVERS (5 regions), gHM v1, RESOLVE 2017. Licences in `docs/licenses.md`.
- Owner decisions applied (CHOLMOD, avg-conductance, UTM, GRIP4, K ≤ 4, Xirro, NeurIPS 2027, Croissant); storage plan in `docs/compute_env.md` §10.

**Decisions needed from the owner**
1. Hugging Face storage: free private quota is 100 GB; v1.0 ≈ 215 GB. Options: PRO account, make repo public, or cap v1.0 ≈ 90 GB while private.
2. Approve downloading the remaining GRIP4 regions 4 and 6 + HydroRIVERS as/si/ar/gr (~2.3 GB; cumulative sources ≈ 5.1 GB, crossing the 5 GB gate) for full-scale global tiling.

Full report: `docs/phase_02_report.md`.

---

# Status — 2026-09-05 (end of Phase 3)

**Phase 3 complete; waiting for confirmation to start Phase 4.**

- Resistance tables done: `generic_hm`, `large_mammal`, `amphibian`, `forest_bird`, `random_lm_20260905` (YAML + pydantic schema + citations). Values are AmpScape's own, literature-ordered.
- Applied to all 60 pilot tiles: 300 rasters, all in [1, r_max], masks consistent, 11 tests green (70 total in the repo). Gallery: `docs/figures/pilot_resistance_gallery.png`.
- Owner decisions applied: HF storage deferred to Phase 5; download gate = single > 5 GB or cumulative > 20 GB; full GRIP4 + HydroRIVERS set downloaded (4.87 GB cumulative, sha256 manifest); neighbour-rule wording corrected (average conductance **is** the Circuitscape default); GRIP4/OSM wording fixed; WDPA not downloaded (habitat-patch focal regions planned instead).

**Nothing blocking.** One note: `forest_bird` saturates on three Andean tiles above 3000 m (by design; will be QC-flagged in Phase 5).

Full report: `docs/phase_03_report.md`.

---

# Status — 2026-09-05 (end of Phase 4)

**Phase 4 complete; waiting for confirmation to start Phase 5 (solver pipeline + mini run).**

- Focal-node / source generators done (`ampscape/sources`): points with per-node mixed placement, wall-to-wall strips, habitat-patch regions, T3 source/ground, T4 Omniscape sources; every configuration checked on an **exact reconstruction of the Circuitscape graph** (unit-tested against the source formulas).
- Config with documented defaults: `configs/tasks/sources_default.yaml` (sha256 recorded per sample); tier scaling built in.
- Pilot build: **1865 configurations, all connected**; anywhere-placement fraction 0.312 (target 0.3); regions on 23/60 tiles. Gallery: `docs/figures/pilot_sources_gallery.png`.
- Verified in Circuitscape that repeated point-raster labels are one short-circuited region (T1W design settled).
- Tests: 21 new, 91 passing overall.

**Nothing blocking.** Carried forward: `frac_at_rmax` QC flag, Omniscape r/b timing, HF storage decision (all Phase 5).

Full report: `docs/phase_04_report.md`.

---

# Status — 2026-09-05 (end of Phase 5, MANDATORY GATE)

**Phase 5 complete; stopped at the gate. Nothing will be scaled until the owner decides on §9 of `docs/phase_05_report.md`.**

- Solver wrapper (`AmpScapeSolve.jl`) + Python driver + Slurm array template done; smoke (5), mini (250 samples, 1,270 configs, **100 % QC pass**), probes M/L/XL (5 each) and XXL (1) all solved with CHOLMOD.
- **Reference solver validated:** CHOLMOD vs CG+AMG on 10 samples agree to 6e-7 (cum current), 3e-8 (Reff), 4e-6 (T3), 4e-7 (T4); CHOLMOD residuals 1e-11–1e-14 vs 1e-6–3e-5 for CG; both bitwise deterministic single-threaded; CG+AMG aborted an Omniscape solve on a contrast-10⁴ landscape → CHOLMOD is the reference for T4 too.
- **Cost driver = Omniscape (T4)**: 6.2 s (S), 48 s (M), 178 s (L), 13 min (XL, block 17), 74 min (XXL, block 33) per solve; everything else is ≤ 13 s up to XL. Adopted XL block 33 / XXL block 65 (≈ 4× cheaper).
- **Budget:** brief's ladder ≈ **7,400 CPU-hours, 451 GB**. 500 CPU-hours buys ≈ 26.5k S / 1.9k M / 300 L / 200 XL / 15 XXL at 4 cores per job (43 GB), or ≈ 106k S / 7.8k M / 1.3k L / 600 XL / 60 XXL at 1 core per job (167 GB, needs a one-shard check). Storage: 0.93 MB (S) → 139 MB (XXL) per sample.
- QC exclusions: 0 hard failures; 3 configurations of one contrast-10⁴ landscape flagged `rmax_saturated` (kept in index, excluded from train/val). Residual threshold set to 1e-6 (double-precision floor ≈ 1e-8 at contrast 10⁴).
- Incident: a stale shared Julia cache across CPU types caused a > 20 min precompile hang; fixed with a portable `JULIA_CPU_TARGET` and login-node precompile in `generate.py submit`.

**Owner decisions needed:** (1) ladder/budget and whether T4 is solved on a subset; 1-core jobs for S/M; (2) HF storage plan; (3) confirm Omniscape XL r128/b33, XXL r256/b65; (4) the correct `HF_ORG` (placeholder "<correct name>" was not filled in).

Full report: `docs/phase_05_report.md`.

---

# Status — 2026-09-05 (dataset plan written; STOPPED, no v1.0 generation launched)

**`docs/dataset_plan.md` is complete.** Compute-agnostic v1.0 design; ICE is pipeline/mini/dev only.

- **Coverage:** 174,400 landscapes (S 100k, M 50k, L 20k, XL 4k, XXL 400; 60 % synthetic / 40 % real; 13,952 real tiles × 5 tables incl. one seeded random table per tile), **902,904 solves** across T1/T2/T1W/T1R/T3/T4 (+3,000 4-neighbour ablation solves). Every family × table × source configuration × task × tier cell is non-zero. Named `hard` stratum = 20 % of synthetic (contrast 10⁴, r_max-saturated, narrow corridors, large NoData).
- **Omniscape block size (fidelity study, 12 paired solves):** coarsening rejected (XL 17→33: 2.5–2.9 % rel. L2; XXL 33→65: 5.9–11.9 %); anchors vs the exact block-1 map: S b3 = 4.5 %, M b5 = 2.0 %. **Adopted rule: block = largest odd ≤ radius/10** (S 1, M 3, L 5, XL 11, XXL 25), the setting that stays within ≈ 1 % of exact by extrapolation; the Phase-5 blocks (S 3 … XXL 33) are kept as a priced option.
- **Cost (measured, CHOLMOD, single-threaded):** recommended ladder ≈ **11,309 CPU-hours, ≈ 812 GB** with the fidelity blocks (≈ 3,981 CPU-h with the Phase-5 blocks); brief's baseline ladder ≈ 7,586 CPU-h / 552 GB. Wall-clock ≈ 4.7 days at 100 cores, 0.9 d at 500, 0.5 d at 1,000. T4 > 90 % of compute; XXL needs 9.5 GB per job.
- **Recommended deviations from the brief:** L 10k→20k, XL 2k→4k, XXL 200→400 (per-cell OOD-scale statistics); named hard-case stratum; random table per tile; ≥ 150 real S tiles per biome; `forest_bird` as the held-out table; 2 biomes + 1 realm as held-out regions; Omniscape b/r ≤ 0.10.
- **Portability:** `configs/cluster/{ice,template}.yaml`, profile-aware `generate.py submit`, `docs/run_guide.md`; requirements: Julia 1.11.3, depot on scratch, `JULIA_CPU_TARGET`, Python from `uv.lock`, no network in jobs.
- **Gaps (Phase 6):** `narrow_corridor` generator, NoData prior extension, real tiles at M–XXL, per-tier seed ranges / cross-tier tile exclusion, per-tile random table, HDF5 schema + splits + sync tool, 4-neighbour duplicates in the planner; mini was generated with the Phase-5 blocks and would be regenerated under the fidelity rule.

Phase 5 decisions applied (CHOLMOD all tasks, residual 1e-6, T4 everywhere, HF_ORG = Xirro, HF storage deferred).

---

# Status — 2026-09-06 (end of Phase 6)

**Phase 6 complete; waiting for confirmation to start Phase 7.** Rename to AmpScape pushed (repo `Xyrro/AmpScape`).

- Schema v0.2 + validator (`ampscape/io/schema.py`, `docs/schema.md`), Zarr export, streaming sync (validate → upload → verify → delete; dry-run default, pushes gated).
- Splits: 20° macro-cells shared across tiers (5° rejected: footprint cascade turned the globe into test), seed-family for synthetic, holdouts/OOD flags, XL 25 % train/val; cross-tier overlap test passes with 0 overlaps.
- Mini regenerated under the radius/10 rule (S block 1): 250 samples, 1 270 configs, 232 MB, 100 % QC pass, all shards schema-valid; T4 53.7 s/solve at S (8.7× block 3). Old mini kept as `mini_phase5blocks`.
- Plan amendments C1–C5 applied to `configs/datasets/v1_0.yaml` and `docs/dataset_plan.md`; litter of empty Omniscape project dirs fixed and removed.
- Tests: 106 passing (8 new). Nothing blocking.

Full report: `docs/phase_06_report.md`.
