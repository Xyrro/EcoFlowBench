# AmpScape — project conventions for Claude Code

## Identity
- Project name: **AmpScape**. Python package `ampscape`, Julia package `AmpScapeSolve.jl`, Hugging Face dataset `Xirro/AmpScape` (private during development).
- Master specification: `docs/TASK_BRIEF.md`. Read it fully before doing anything. It is the source of truth for scope, tasks, formats, and acceptance criteria.
- Owner: the user. Communicate in English. Be concise in chat; be thorough in files.

## How to work
- Work phase by phase, in the order given in the task brief. Do not skip ahead.
- At the end of each phase: write `docs/phase_XX_report.md`, run tests, commit with a clear message, then **STOP and wait for confirmation** before starting the next phase.
- Every time you stop to report, write the same summary to `docs/status/latest.md` and append it to `docs/status/history.md` (owner reads these on GitHub).
- Log every design decision that refines or deviates from the brief in `DECISIONS.md` with a one-line rationale.
- Keep `CHANGELOG.md` up to date.
- Prefer small, reviewable commits. Never force-push.
- Never add `Co-Authored-By`, "Generated with Claude Code", or `Claude-Session` trailers to commits or PRs (a `commit-msg` hook strips them; `~/.claude/settings.json` sets `attribution` to empty).
- Do not fabricate data sources, URLs, licenses, or citations. If something cannot be verified, say so in the report.

## Compute environment: Georgia Tech PACE-ICE (Slurm)
- Before writing any job script, inspect the cluster (`sinfo`, `scontrol show partition`, `sacctmgr show qos`, `module avail`, `pace-quota`, `nvidia-smi` on a GPU node if available) and record findings in `docs/compute_env.md`.
- Home directory quota is small. Put all data, the Julia depot (`JULIA_DEPOT_PATH`), package caches, and Python environments under the scratch directory. Never write large files to `$HOME`.
- Scratch may be purged periodically and is capped at 300 GB. Finished shards are synced to the private HF dataset repo `Xirro/AmpScape` (validate → upload → checksum-verify → delete locally); keep the local working set under 150 GB.
- Assume compute nodes have **no internet access**. Downloads, package installs, and Hugging Face pushes run on the login node only, never inside `sbatch` jobs.
- Login nodes are shared: do not run heavy computation on them. Anything more than a few minutes of CPU goes through Slurm.
- Generation jobs must be resumable: Slurm job arrays, ≤ 4 hours per job, one shard per job, skip shards that already exist and validate.
- Use `module load` for Julia, GDAL, Python/Anaconda if available; otherwise install to scratch.
- Baseline training uses GPU partitions; keep mini-scale runs short. Full-scale baselines will be scheduled separately after discussion.

## Gates (must ask the owner before)
- Downloading source data: any single download > 5 GB, or cumulative source downloads > 20 GB (running total in `data/sources/manifest.json`).
- Launching more than 500 CPU-hours or 20 GPU-hours of jobs.
- Pushing anything to Hugging Face or creating any public artifact.
- Deleting data.

## Environment variables (owner fills in)
- `AMPSCAPE_SCRATCH=/storage/ice1/1/8/yxiao413/EcoFlowBench` path to the scratch working directory
- `HF_ORG=Xirro` Hugging Face organization or username
