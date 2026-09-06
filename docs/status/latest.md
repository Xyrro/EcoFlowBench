# Status — 2026-09-05 (dataset plan written; STOPPED, no v1.0 generation launched)

**`docs/dataset_plan.md` is complete.** Compute-agnostic v1.0 design; ICE is pipeline/mini/dev only.

- **Coverage:** 174,400 landscapes (S 100k, M 50k, L 20k, XL 4k, XXL 400; 60 % synthetic / 40 % real; 13,952 real tiles × 5 tables incl. one seeded random table per tile), **902,904 solves** across T1/T2/T1W/T1R/T3/T4 (+3,000 4-neighbour ablation solves). Every family × table × source configuration × task × tier cell is non-zero. Named `hard` stratum = 20 % of synthetic (contrast 10⁴, r_max-saturated, narrow corridors, large NoData).
- **Cost (measured, CHOLMOD, single-threaded):** ≈ **3,981 CPU-hours, ≈ 812 GB**; brief's baseline ladder ≈ 3,100 CPU-h / 690 GB. Wall-clock 1.7 days at 100 cores, 0.3 d at 500, 0.2 d at 1,000. T4 ≈ 90 % of compute; XXL needs 9.5 GB per job.
- **Omniscape block size (fidelity study, 12 paired solves):** coarser blocks rejected (XL b33 2.2–2.9 %, XXL b65 5.9–11.9 % relative L2). Recommended rule block = odd(radius/8): S 3, M 5, L 9, XL 17, XXL 33. Measured against the exact block-1 limit: S b3 deviates 2.9–7.6 %, M b5 1.7–2.5 %, at 6× / 22× lower cost. **Owner decision:** block 1 as exact anchor at tier S (+≈ 1,200 CPU-h) or the uniform rule.
- **Recommended deviations from the brief:** L 10k→20k, XL 2k→4k, XXL 200→400 (per-cell OOD-scale statistics); named hard-case stratum; random table per tile; ≥ 150 real S tiles per biome; `forest_bird` as the held-out table; 2 biomes + 1 realm as held-out regions.
- **Portability:** `configs/cluster/{ice,template}.yaml`, profile-aware `generate.py submit`, `docs/run_guide.md`; requirements: Julia 1.11.3, depot on scratch, `JULIA_CPU_TARGET`, Python from `uv.lock`, no network in jobs.
- **Gaps (Phase 6):** `narrow_corridor` generator, NoData prior extension, real tiles at M–XXL, per-tier seed ranges / cross-tier tile exclusion, per-tile random table, HDF5 schema + splits + sync tool, 4-neighbour duplicates in the planner.

Phase 5 decisions applied (CHOLMOD all tasks, residual 1e-6, T4 everywhere, HF_ORG = Xirro, HF storage deferred).
