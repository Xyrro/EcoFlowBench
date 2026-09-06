# Status — 2026-09-06 (Phase 7 code complete; HF push blocked on login)

- Macro-cell check: previous 20° lon/lat grid could not hold XXL tiles anywhere (0 % fit); adopted equal-width 20° grid + XXL footprints as regions; placeability uniform in latitude, 0 geometric cross-tier overlaps.
- Phase 7: `AmpScapeDataset` + `load_from_hub`, HF layout by tier × task group (any tier/task downloads alone), nested mini/core/full subsets, Croissant 1.0 + RAI (validated, 0 errors), push script with verified uploads, dataset card draft with the honest framing. Mini staged: 38 files, 300 MB. Tests: 110 passing.
- ICE feasibility: v1.0 (11 300 core-h) would finish in ≈ 1–5 days on ICE by the limits alone (`docs/compute_env.md` §8.2); policy and the 300 GB scratch are the real constraints.
- Paper note recorded: mini's real tiles skew to held-out regions (pilot artifact).

**Blocked:** the login node has no Hugging Face token (`hf auth whoami` → Not logged in). After `hf auth login`, the real push, the download-alone check and the sync delete test run.
