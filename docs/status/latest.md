# Status — 2026-09-06 (end of Phase 6)

**Phase 6 complete; waiting for confirmation to start Phase 7.** Rename to AmpScape pushed (repo `Xyrro/AmpScape`).

- Schema v0.2 + validator (`ampscape/io/schema.py`, `docs/schema.md`), Zarr export, streaming sync (validate → upload → verify → delete; dry-run default, pushes gated).
- Splits: 20° macro-cells shared across tiers (5° rejected: footprint cascade turned the globe into test), seed-family for synthetic, holdouts/OOD flags, XL 25 % train/val; cross-tier overlap test passes with 0 overlaps.
- Mini regenerated under the radius/10 rule (S block 1): 250 samples, 1 270 configs, 232 MB, 100 % QC pass, all shards schema-valid; T4 53.7 s/solve at S (8.7× block 3). Old mini kept as `mini_phase5blocks`.
- Plan amendments C1–C5 applied to `configs/datasets/v1_0.yaml` and `docs/dataset_plan.md`; litter of empty Omniscape project dirs fixed and removed.
- Tests: 106 passing (8 new). Nothing blocking.

Full report: `docs/phase_06_report.md`.
