# Notes for the manuscript

Facts to state explicitly in the paper (collected during generation).

- **Pilot artifact (2026-09-06):** the mini set's 50 real tiles skew toward held-out regions — 20 of 50 fall in
  `test_ood_region` (Australasia and the held-out biomes) because the Phase 2 pilot over-sampled those strata for
  coverage. This is a property of the 50-tile pilot, not of v1.0 (plan §4 targets ≈ 12 % of real tiles as
  OOD-region); the mini's real train/val/test_id counts (19 / 2 / 1) are therefore not representative.
- Real landscapes with synthesized resistance: resistance tables are literature-informed but the numeric values are
  AmpScape's own (`configs/resistance_tables/README.md`); the solver output is the ground truth, never post-processed.
