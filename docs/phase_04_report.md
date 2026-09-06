# Phase 4 report — focal nodes and source configurations

Date: 2026-09-05. Status: **complete, awaiting owner confirmation** before Phase 5 (solver pipeline).
Same summary in `docs/status/latest.md`.

## What was built

| Item | Location |
|---|---|
| Exact solver-graph reconstruction (8-neighbour, NoData removed, average conductance, diagonal /√2), Laplacian, component labels | `ampscape/sources/graph.py` |
| Config schema with documented defaults and per-tier pixel scaling | `ampscape/sources/config.py`, `configs/tasks/sources_default.yaml` |
| Generators: point focal nodes (K ∈ [2,8], min separation, per-node mixed placement), wall-to-wall strips (NS/EW), habitat-patch focal regions, T3 source + ground rasters, T4 Omniscape sources; `generate_all` (sub-seeded per kind) | `ampscape/sources/generators.py` |
| Build script over every (tile, table) resistance raster + 50 synthetic landscapes; NPZ per landscape (int32 masks, float32 sources, int8 grounds, JSON tables), `sources.parquet`, gallery | `scripts/build_sources.py` |
| Circuitscape check that repeated point labels act as one short-circuited region | `julia/AmpScapeSolve.jl/examples/region_check.jl` |
| Tests (21): graph formulas vs Circuitscape source, NoData/components/diagonal contact, point contract (range, separation, placement bookkeeping, low-R threshold), placement fraction ≈ 0.3, determinism, split-landscape safety, strips, regions (min size, trimming, absence), T3/T4 contracts, provenance, tier scaling | `tests/test_sources.py` |

## Owner requirements

1. **Exact graph.** `graph.py` reproduces `construct_graph` edge weights exactly (unit-tested on a
   3×3 case: cardinal (g_i+g_j)/2, diagonal /√2, 20 edges, symmetric, zero-row-sum Laplacian;
   4-neighbour and average-resistance variants also tested). Candidates come from the largest
   component; after placement every focal/source/ground pixel is verified to lie in one component.
   Result: **1865 / 1865 configurations connected**, 0 separation relaxations.
2. **Config values** with defaults in `configs/tasks/sources_default.yaml` (sha256 recorded per
   sample): strip width 2 px, min separation 16 px, habitat classes {10, 90, 95} with min patch
   50 px and max region 2000 px, K ∈ [2, 8] (regions K ∈ [2, 6]), T3 source = (1/R)^1 above the
   0.7 quantile normalised to Σ = 1 with grounds ∈ {edge, all_edges, patches}, T4 source = (1/R)
   above the 0.5 quantile scaled to max 1 with `source_threshold = 0`. Tier scaling ×2/×4/×8/×16.
3. **Placement.** Per node, P(anywhere) = 0.3 else low-resistance (R ≤ 0.25 quantile). Measured
   anywhere fraction over all point nodes: **0.312**. Per-sample summary:
   {'mixed': 230, 'low_resistance': 67, 'anywhere': 3}. Each node's placement and resistance is in the table.
4. **Storage.** int32 label raster + table (label, row, col, kind, placement, n_pixels, resistance;
   WKT polygon for regions/strips) in every NPZ and, from Phase 6, in HDF5 `inputs/focal_mask` +
   `inputs/focal_table`.

## Measured numbers (pilot: 60 tiles × 5 tables, plus 50 synthetic)

| Kind | real configs | synthetic | notes |
|---|---|---|---|
| points | 300 | 50 | K distribution (real): {2: 45, 3: 43, 4: 42, 5: 58, 6: 39, 7: 35, 8: 38} |
| wall_to_wall NS / EW | 300 / 300 | 50 / 50 | 256 strip pixels each (minus NoData) |
| regions | 115 | — | 23 of 60 tiles have ≥ 2 eligible, separated habitat patches |
| advanced (T3) | 300 | 50 | ground modes {'patches': 114, 'edge': 95, 'all_edges': 91}; mean 6362 source px |
| omniscape (T4) | 300 | 50 | mean 9319 source px |

Every landscape (real and synthetic) had exactly one graph component after NoData removal, so
no placement was ever forced away from the full raster. Build time: 25 s; NPZ total 13 MB.

## Findings

- **Region semantics verified in Circuitscape:** two 12-pixel strips give Reff 0.469 versus 1.037
  for their midpoints, and the cumulative current map shows the full unit current on *every*
  strip pixel. Consequence recorded in the spec: focal-region pixels are excluded from map metrics.
- Habitat-patch regions are available on 38 % of pilot tiles; they are an additional
  configuration, not a replacement for point samples.
- Synthetic landscapes never have NoData pixels touching both strips' worth of edge, but the
  single-component guarantee from Phase 2 means strips are never empty; the code still raises
  if a strip has no valid pixels.

## Open items carried to Phase 5

- QC flag `frac_at_rmax > 0.5` (exclude from train/val, keep in index) per owner decision.
- Omniscape radius / block size per tier are still proposals to be timed in the mini run.
- HF storage decision at the Phase 5 gate.

## Next step (Phase 5, on confirmation)

`AmpScapeSolve.jl` solve functions (pairwise / advanced / omniscape with `SolveStats`),
Python driver with QC and Slurm array template, then the **mandatory mini run** (200 synthetic +
50 real at tier S, all tasks) with quicklooks, timings and the full-budget estimate — stopping
at the gate before any scaling.
