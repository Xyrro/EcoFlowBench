# Status — 2026-09-05 (end of Phase 3)

**Phase 3 complete; waiting for confirmation to start Phase 4.**

- Resistance tables done: `generic_hm`, `large_mammal`, `amphibian`, `forest_bird`, `random_lm_20260905` (YAML + pydantic schema + citations). Values are EcoFlowBench's own, literature-ordered.
- Applied to all 60 pilot tiles: 300 rasters, all in [1, r_max], masks consistent, 11 tests green (70 total in the repo). Gallery: `docs/figures/pilot_resistance_gallery.png`.
- Owner decisions applied: HF storage deferred to Phase 5; download gate = single > 5 GB or cumulative > 20 GB; full GRIP4 + HydroRIVERS set downloaded (4.87 GB cumulative, sha256 manifest); neighbour-rule wording corrected (average conductance **is** the Circuitscape default); GRIP4/OSM wording fixed; WDPA not downloaded (habitat-patch focal regions planned instead).

**Nothing blocking.** One note: `forest_bird` saturates on three Andean tiles above 3000 m (by design; will be QC-flagged in Phase 5).

Full report: `docs/phase_03_report.md`.
