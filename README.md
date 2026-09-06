# AmpScape

**A benchmark dataset and evaluation suite for learned surrogates of circuit-theoretic landscape connectivity (Circuitscape / Omniscape).**

AmpScape provides standardized (resistance landscape, source configuration) → (solver output) pairs for training and fairly comparing machine-learning surrogates of circuit-theory connectivity solvers, together with official splits, out-of-distribution test sets, evaluation metrics, baselines, and a fully reproducible generation pipeline.

> Status: under development. See `docs/TASK_BRIEF.md` for the full project specification.

## Tasks
- **T1** Pairwise current mapping
- **T2** Effective resistance prediction
- **T3** Advanced-mode (source/ground) flow
- **T4** Omnidirectional connectivity (Omniscape)

## Repository layout
See `docs/TASK_BRIEF.md` §2.2. Phase 1 documents: `docs/compute_env.md`, `docs/prior_art.md`,
`docs/task_specification.md`, `docs/phase_01_report.md`.

## Development setup (PACE-ICE)
```bash
source scripts/env.sh          # isolates from ~/.bashrc conda env, points caches at scratch
uv sync --extra dev            # Python 3.11 venv on scratch (pinned in uv.lock)
julia --project=julia/AmpScapeSolve.jl -e 'using Pkg; Pkg.instantiate()'
bash scripts/setup_gdal_env.sh # optional GDAL CLI tools
python -m pytest -q
```

## License
Code: MIT. Data: CC BY 4.0 (subject to upstream data licenses; see `docs/licenses.md`).

## Citation
To be added.
