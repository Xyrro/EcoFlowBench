#!/usr/bin/env python
"""Dataset build driver (Phase 5): plan -> prepare -> solve -> finalize, one shard per Slurm job.

    python scripts/generate.py plan     --dataset mini --out data/builds/mini [--n-synthetic 200 --n-real 50 --tier S --shard-size 50]
    python scripts/generate.py prepare  --build data/builds/mini [--shard 0]
    python scripts/generate.py solve    --build data/builds/mini --shard 0            # local (compute node)
    python scripts/generate.py submit   --build data/builds/mini [--time 04:00:00 --cpus 4 --mem 16G --partition coc-cpu]
    python scripts/generate.py finalize --build data/builds/mini [--shard 0] [--quicklooks]
    python scripts/generate.py status   --build data/builds/mini

Resumable at every stage: existing inputs / complete outputs / final shards are skipped unless --force.
Solving never needs the network (inputs are materialised in `prepare`, on the login node).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
JULIA_PKG = ROOT / "julia" / "EcoFlowBenchSolve.jl"
SBATCH_TEMPLATE = ROOT / "scripts" / "slurm" / "solve_shard.sbatch"


def load_manifest(build: pathlib.Path) -> pd.DataFrame:
    return pd.read_parquet(build / "manifest.parquet")


def shard_paths(build: pathlib.Path, shard: int) -> dict[str, pathlib.Path]:
    s = f"shard-{shard:05d}"
    return {"inputs": build / "inputs" / f"{s}.inputs.h5", "outputs": build / "outputs" / f"{s}.outputs.h5",
            "final": build / "shards" / f"{s}.h5", "index": build / "index" / f"{s}.parquet",
            "quicklooks": build / "quicklooks" / s}


def cmd_plan(a) -> None:
    from ecoflowbench.solve.manifest import plan_real, plan_synthetic, to_frame

    build = pathlib.Path(a.out)
    build.mkdir(parents=True, exist_ok=True)
    configs = a.configs.split(",") if a.configs else None
    from ecoflowbench.solve.manifest import DEFAULT_CONFIGS

    specs = plan_synthetic(a.dataset, a.n_synthetic, a.tier, a.seed0, configs=configs or DEFAULT_CONFIGS, shard_size=a.shard_size)
    if a.k_override:
        for sp in specs:
            sp.extra = json.dumps({"k_override": a.k_override})
    n_shards = (len(specs) + a.shard_size - 1) // a.shard_size if specs else 0
    if a.n_real:
        specs += plan_real(a.dataset, str(ROOT / a.pilot / "resistance.parquet"), str(ROOT / a.pilot / "sources.parquet"),
                           a.n_real, a.tier, a.seed0, configs=configs or DEFAULT_CONFIGS, shard_size=a.shard_size, shard0=n_shards)
    df = to_frame(specs)
    df.to_parquet(build / "manifest.parquet", index=False)
    cfg = {"dataset_id": a.dataset, "tier": a.tier, "seed0": a.seed0, "shard_size": a.shard_size, "pilot": a.pilot,
           "n_synthetic": a.n_synthetic, "n_real": a.n_real, "source_config": a.source_config,
           "solver_preset": a.solver_preset, "dataset_version": a.dataset_version}
    (build / "build.json").write_text(json.dumps(cfg, indent=1))
    print(f"planned {len(df)} samples in {df.shard.nunique()} shards -> {build}")
    print(df.groupby(["family", "shard"]).size().to_string())
    if "generator" in df:
        print("generators:", df[df.family == "synthetic"].generator.value_counts().to_dict())
        print("tables:", df[df.family == "real"].table_id.value_counts().to_dict())


def cmd_prepare(a) -> None:
    from ecoflowbench.solve.manifest import from_frame
    from ecoflowbench.solve.prepare import prepare_shard
    from ecoflowbench.sources import SourceConfig

    build = pathlib.Path(a.build)
    cfg = json.loads((build / "build.json").read_text())
    df = load_manifest(build)
    scfg = SourceConfig.from_yaml(ROOT / cfg["source_config"])
    shards = [a.shard] if a.shard is not None else sorted(df.shard.unique())
    for sh in shards:
        p = shard_paths(build, int(sh))
        specs = from_frame(df[df.shard == sh])
        n = prepare_shard(specs, str(p["inputs"]), scfg, pilot_root=str(ROOT / cfg["pilot"]), overwrite=a.force)
        print(f"shard {sh}: prepared {n} samples -> {p['inputs'].name}" if n else f"shard {sh}: inputs already present")


def julia_cmd(inputs, outputs, tmp, solver, fallback, osolver, max_n=None, configs=None, force=False) -> list[str]:
    cmd = ["julia", f"--project={JULIA_PKG}", str(JULIA_PKG / "scripts" / "solve_shard.jl"), str(inputs), str(outputs),
           "--tmp", tmp, "--solver", solver, "--fallback", fallback, "--omniscape-solver", osolver]
    if max_n:
        cmd += ["--max", str(max_n)]
    if configs:
        cmd += ["--configs", configs]
    if force:
        cmd += ["--force", "true"]
    return cmd


def cmd_solve(a) -> None:
    build = pathlib.Path(a.build)
    p = shard_paths(build, a.shard)
    p["outputs"].parent.mkdir(parents=True, exist_ok=True)
    tmp = os.environ.get("TMPDIR", "/tmp")
    cmd = julia_cmd(p["inputs"], p["outputs"], tmp, a.solver, a.fallback, a.omniscape_solver, a.max,
                    configs=a.configs, force=a.force)
    print(" ".join(cmd))
    sys.exit(subprocess.run(cmd, check=False).returncode)


def cmd_submit(a) -> None:
    build = pathlib.Path(a.build).resolve()
    df = load_manifest(build)
    shards = sorted(int(s) for s in df.shard.unique())
    todo = [s for s in shards if not shard_paths(build, s)["final"].exists()] if not a.force else shards
    if not todo:
        print("nothing to submit")
        return
    arr = ",".join(str(s) for s in todo)
    (build / "logs").mkdir(exist_ok=True)
    cmd = ["sbatch", f"--array={arr}%{a.max_concurrent}", f"--time={a.time}", f"--cpus-per-task={a.cpus}", f"--mem={a.mem}",
           f"--partition={a.partition}", f"--job-name=efb-{build.name}", f"--output={build}/logs/%A_%a.out",
           f"--export=ALL,EFB_BUILD={build},EFB_SOLVER={a.solver},EFB_FALLBACK={a.fallback},EFB_OSOLVER={a.omniscape_solver},"
           f"EFB_CONFIGS={a.configs or ''},EFB_FORCE={'1' if a.force_solve else '0'}",
           str(SBATCH_TEMPLATE)]
    print(" ".join(cmd))
    if not a.dry_run:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        print(out)
        (build / "logs" / "submissions.txt").open("a").write(out + f"  shards={arr}\n")


def cmd_finalize(a) -> None:
    import yaml

    from ecoflowbench.solve.finalize import finalize_shard
    from ecoflowbench.solve.quicklook import shard_quicklooks

    build = pathlib.Path(a.build)
    cfg = json.loads((build / "build.json").read_text())
    df = load_manifest(build)
    preset = yaml.safe_load(open(ROOT / cfg["solver_preset"]))
    preset = {k: v for k, v in preset.items() if k != "ablation_four_neighbour"}
    shards = [a.shard] if a.shard is not None else sorted(int(s) for s in df.shard.unique())
    for sh in shards:
        p = shard_paths(build, sh)
        if not p["outputs"].exists():
            print(f"shard {sh}: no outputs yet")
            continue
        if p["final"].exists() and not a.force:
            print(f"shard {sh}: final exists")
            continue
        idx = finalize_shard(str(p["inputs"]), str(p["outputs"]), str(p["final"]), cfg["dataset_version"], preset)
        p["index"].parent.mkdir(parents=True, exist_ok=True)
        idx.to_parquet(p["index"], index=False)
        n_fail = int((~idx.qc_pass).sum())
        print(f"shard {sh}: {idx.sample_id.nunique()} samples, {len(idx)} configs, {n_fail} QC failures, "
              f"{p['final'].stat().st_size/1e6:.1f} MB")
        if a.quicklooks:
            shard_quicklooks(str(p["final"]), str(p["quicklooks"]))
    parts = sorted((build / "index").glob("shard-*.parquet"))
    if parts:
        full = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
        full.to_parquet(build / "index.parquet", index=False)
        (build / "stats").mkdir(exist_ok=True)
        full[["sample_id", "config", "kind", "family", "tier", "H", "W", "K", "solver", "solve_time_s", "maxrss_mb"]].to_parquet(
            build / "stats" / "solve_times.parquet", index=False)
        print(f"index: {len(full)} rows, {full.sample_id.nunique()} samples, qc_pass={full.qc_pass.mean():.3f}")


def cmd_status(a) -> None:
    build = pathlib.Path(a.build)
    df = load_manifest(build)
    rows = []
    for sh in sorted(int(s) for s in df.shard.unique()):
        p = shard_paths(build, sh)
        done = 0
        if p["outputs"].exists():
            import h5py

            with h5py.File(p["outputs"], "r") as f:
                done = sum(1 for s in f["samples"] if "complete" in f["samples"][s].attrs)
        rows.append({"shard": sh, "samples": int((df.shard == sh).sum()), "inputs": p["inputs"].exists(),
                     "solved": done, "final": p["final"].exists()})
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-synthetic", type=int, default=200)
    p.add_argument("--n-real", type=int, default=50)
    p.add_argument("--tier", default="S")
    p.add_argument("--seed0", type=int, default=20260905)
    p.add_argument("--shard-size", type=int, default=50)
    p.add_argument("--pilot", default="data/tiles/pilot")
    p.add_argument("--source-config", default="configs/tasks/sources_default.yaml")
    p.add_argument("--solver-preset", default="configs/solver/circuitscape_reference.yaml")
    p.add_argument("--dataset-version", default="0.1.0-mini")
    p.add_argument("--configs", default=None, help="comma-separated subset of configs (default: all)")
    p.add_argument("--k-override", type=int, default=None, help="fix K for the points config (scaling probe)")
    p.set_defaults(func=cmd_plan)
    for name, fn in [("prepare", cmd_prepare), ("finalize", cmd_finalize)]:
        q = sub.add_parser(name)
        q.add_argument("--build", required=True)
        q.add_argument("--shard", type=int, default=None)
        q.add_argument("--force", action="store_true")
        if name == "finalize":
            q.add_argument("--quicklooks", action="store_true")
        q.set_defaults(func=fn)
    s = sub.add_parser("solve")
    s.add_argument("--build", required=True)
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--solver", default="cholmod")
    s.add_argument("--fallback", default="cg+amg")
    s.add_argument("--omniscape-solver", default="cholmod")
    s.add_argument("--max", type=int, default=None)
    s.add_argument("--configs", default=None, help="comma-separated configs to (re-)solve")
    s.add_argument("--force", action="store_true", help="re-solve listed configs even if the sample is complete")
    s.set_defaults(func=cmd_solve)
    b = sub.add_parser("submit")
    b.add_argument("--build", required=True)
    b.add_argument("--time", default="04:00:00")
    b.add_argument("--cpus", type=int, default=4)
    b.add_argument("--mem", default="16G")
    b.add_argument("--partition", default="coc-cpu")
    b.add_argument("--max-concurrent", type=int, default=20)
    b.add_argument("--solver", default="cholmod")
    b.add_argument("--fallback", default="cg+amg")
    b.add_argument("--omniscape-solver", default="cholmod")
    b.add_argument("--force", action="store_true", help="resubmit shards that already have final files")
    b.add_argument("--configs", default=None, help="only (re-)solve these configs inside the job")
    b.add_argument("--force-solve", action="store_true", help="re-solve listed configs even for complete samples")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_submit)
    t = sub.add_parser("status")
    t.add_argument("--build", required=True)
    t.set_defaults(func=cmd_status)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
