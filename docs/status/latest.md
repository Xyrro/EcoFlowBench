# Status — 2026-09-05 (end of Phase 4)

**Phase 4 complete; waiting for confirmation to start Phase 5 (solver pipeline + mini run).**

- Focal-node / source generators done (`ecoflowbench/sources`): points with per-node mixed placement, wall-to-wall strips, habitat-patch regions, T3 source/ground, T4 Omniscape sources; every configuration checked on an **exact reconstruction of the Circuitscape graph** (unit-tested against the source formulas).
- Config with documented defaults: `configs/tasks/sources_default.yaml` (sha256 recorded per sample); tier scaling built in.
- Pilot build: **1865 configurations, all connected**; anywhere-placement fraction 0.312 (target 0.3); regions on 23/60 tiles. Gallery: `docs/figures/pilot_sources_gallery.png`.
- Verified in Circuitscape that repeated point-raster labels are one short-circuited region (T1W design settled).
- Tests: 21 new, 91 passing overall.

**Nothing blocking.** Carried forward: `frac_at_rmax` QC flag, Omniscape r/b timing, HF storage decision (all Phase 5).

Full report: `docs/phase_04_report.md`.
