# Changelog

## [Unreleased]

### Phase 0/1 — 2026-09-05
- Cluster inspection of PACE-ICE recorded in `docs/compute_env.md` (partitions, QoS, GPUs, storage, network).
- Isolated toolchain on scratch: `scripts/env.sh`, uv-managed Python 3.11.16 (`pyproject.toml`, `uv.lock`),
  Julia 1.11.3 depot with Circuitscape.jl 5.17.1 + Omniscape.jl 0.6.2 (`julia/EcoFlowBenchSolve.jl`),
  GDAL 3.9.3 CLI env (`scripts/setup_gdal_env.sh`).
- Repository scaffold per brief §2.2 (empty Python subpackages, config dirs, Julia package skeleton, tests).
- Solver smoke test (`julia/EcoFlowBenchSolve.jl/examples/smoke_test.jl`) verified against shipped reference outputs.
- `docs/prior_art.md` (42 verified references) and `docs/task_specification.md` (T1/T1W/T2/T3/T4/T5 tensor specs).
- `DECISIONS.md` entries for solver choice (CHOLMOD), neighbour combination, environment layout.
