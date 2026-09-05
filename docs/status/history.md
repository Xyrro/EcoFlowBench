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

- Resistance tables done: `generic_hm`, `large_mammal`, `amphibian`, `forest_bird`, `random_lm_20260905` (YAML + pydantic schema + citations). Values are EcoFlowBench's own, literature-ordered.
- Applied to all 60 pilot tiles: 300 rasters, all in [1, r_max], masks consistent, 11 tests green (70 total in the repo). Gallery: `docs/figures/pilot_resistance_gallery.png`.
- Owner decisions applied: HF storage deferred to Phase 5; download gate = single > 5 GB or cumulative > 20 GB; full GRIP4 + HydroRIVERS set downloaded (4.87 GB cumulative, sha256 manifest); neighbour-rule wording corrected (average conductance **is** the Circuitscape default); GRIP4/OSM wording fixed; WDPA not downloaded (habitat-patch focal regions planned instead).

**Nothing blocking.** One note: `forest_bird` saturates on three Andean tiles above 3000 m (by design; will be QC-flagged in Phase 5).

Full report: `docs/phase_03_report.md`.

---

# Status — 2026-09-05 (end of Phase 4)

**Phase 4 complete; waiting for confirmation to start Phase 5 (solver pipeline + mini run).**

- Focal-node / source generators done (`ecoflowbench/sources`): points with per-node mixed placement, wall-to-wall strips, habitat-patch regions, T3 source/ground, T4 Omniscape sources; every configuration checked on an **exact reconstruction of the Circuitscape graph** (unit-tested against the source formulas).
- Config with documented defaults: `configs/tasks/sources_default.yaml` (sha256 recorded per sample); tier scaling built in.
- Pilot build: **1865 configurations, all connected**; anywhere-placement fraction 0.312 (target 0.3); regions on 23/60 tiles. Gallery: `docs/figures/pilot_sources_gallery.png`.
- Verified in Circuitscape that repeated point-raster labels are one short-circuited region (T1W design settled).
- Tests: 21 new, 91 passing overall.

**Nothing blocking.** Carried forward: `frac_at_rmax` QC flag, Omniscape r/b timing, HF storage decision (all Phase 5).

Full report: `docs/phase_04_report.md`.

---

# Status — 2026-09-05 (end of Phase 5, MANDATORY GATE)

**Phase 5 complete; stopped at the gate. Nothing will be scaled until the owner decides on §9 of `docs/phase_05_report.md`.**

- Solver wrapper (`EcoFlowBenchSolve.jl`) + Python driver + Slurm array template done; smoke (5), mini (250 samples, 1,270 configs, **100 % QC pass**), probes M/L/XL (5 each) and XXL (1) all solved with CHOLMOD.
- **Reference solver validated:** CHOLMOD vs CG+AMG on 10 samples agree to 6e-7 (cum current), 3e-8 (Reff), 4e-6 (T3), 4e-7 (T4); CHOLMOD residuals 1e-11–1e-14 vs 1e-6–3e-5 for CG; both bitwise deterministic single-threaded; CG+AMG aborted an Omniscape solve on a contrast-10⁴ landscape → CHOLMOD is the reference for T4 too.
- **Cost driver = Omniscape (T4)**: 6.2 s (S), 48 s (M), 178 s (L), 13 min (XL, block 17), 74 min (XXL, block 33) per solve; everything else is ≤ 13 s up to XL. Adopted XL block 33 / XXL block 65 (≈ 4× cheaper).
- **Budget:** brief's ladder ≈ **7,400 CPU-hours, 451 GB**. 500 CPU-hours buys ≈ 26.5k S / 1.9k M / 300 L / 200 XL / 15 XXL at 4 cores per job (43 GB), or ≈ 106k S / 7.8k M / 1.3k L / 600 XL / 60 XXL at 1 core per job (167 GB, needs a one-shard check). Storage: 0.93 MB (S) → 139 MB (XXL) per sample.
- QC exclusions: 0 hard failures; 3 configurations of one contrast-10⁴ landscape flagged `rmax_saturated` (kept in index, excluded from train/val). Residual threshold set to 1e-6 (double-precision floor ≈ 1e-8 at contrast 10⁴).
- Incident: a stale shared Julia cache across CPU types caused a > 20 min precompile hang; fixed with a portable `JULIA_CPU_TARGET` and login-node precompile in `generate.py submit`.

**Owner decisions needed:** (1) ladder/budget and whether T4 is solved on a subset; 1-core jobs for S/M; (2) HF storage plan; (3) confirm Omniscape XL r128/b33, XXL r256/b65; (4) the correct `HF_ORG` (placeholder "<correct name>" was not filled in).

Full report: `docs/phase_05_report.md`.
