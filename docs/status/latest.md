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
