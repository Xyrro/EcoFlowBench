# Status — 2026-09-05 (dataset plan written; STOPPED, no v1.0 generation launched)

**`docs/dataset_plan.md` is complete.** Compute-agnostic v1.0 design; ICE is pipeline/mini/dev only.

- **Coverage:** 174,400 landscapes (S 100k, M 50k, L 20k, XL 4k, XXL 400; 60 % synthetic / 40 % real; 13,952 real tiles × 5 tables incl. one seeded random table per tile), **902,904 solves** across T1/T2/T1W/T1R/T3/T4 (+3,000 4-neighbour ablation solves). Every family × table × source configuration × task × tier cell is non-zero. Named `hard` stratum = 20 % of synthetic (contrast 10⁴, r_max-saturated, narrow corridors, large NoData).
- **Omniscape block size (fidelity study, 12 paired solves):** coarsening rejected (XL 17→33: 2.5–2.9 % rel. L2; XXL 33→65: 5.9–11.9 %); anchors vs the exact block-1 map: S b3 = 4.5 %, M b5 = 2.0 %. **Adopted rule: block = largest odd ≤ radius/10** (S 1, M 3, L 5, XL 11, XXL 25), the setting that stays within ≈ 1 % of exact by extrapolation; the Phase-5 blocks (S 3 … XXL 33) are kept as a priced option.
- **Cost (measured, CHOLMOD, single-threaded):** recommended ladder ≈ **11,309 CPU-hours, ≈ 812 GB** with the fidelity blocks (≈ 3,981 CPU-h with the Phase-5 blocks); brief's baseline ladder ≈ 7,586 CPU-h / 552 GB. Wall-clock ≈ 4.7 days at 100 cores, 0.9 d at 500, 0.5 d at 1,000. T4 > 90 % of compute; XXL needs 9.5 GB per job.
- **Recommended deviations from the brief:** L 10k→20k, XL 2k→4k, XXL 200→400 (per-cell OOD-scale statistics); named hard-case stratum; random table per tile; ≥ 150 real S tiles per biome; `forest_bird` as the held-out table; 2 biomes + 1 realm as held-out regions; Omniscape b/r ≤ 0.10.
- **Portability:** `configs/cluster/{ice,template}.yaml`, profile-aware `generate.py submit`, `docs/run_guide.md`; requirements: Julia 1.11.3, depot on scratch, `JULIA_CPU_TARGET`, Python from `uv.lock`, no network in jobs.
- **Gaps (Phase 6):** `narrow_corridor` generator, NoData prior extension, real tiles at M–XXL, per-tier seed ranges / cross-tier tile exclusion, per-tile random table, HDF5 schema + splits + sync tool, 4-neighbour duplicates in the planner; mini was generated with the Phase-5 blocks and would be regenerated under the fidelity rule.

Phase 5 decisions applied (CHOLMOD all tasks, residual 1e-6, T4 everywhere, HF_ORG = Xirro, HF storage deferred).
