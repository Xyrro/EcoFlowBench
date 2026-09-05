#!/usr/bin/env python
"""Summaries for the Phase 5 report: solve time / memory distributions, storage per sample, QC.

Usage: python scripts/analyze_build.py --builds data/builds/mini data/builds/probe_M ... --out docs/figures --prefix phase05
Prints Markdown tables and writes <prefix>_solve_times.png.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import pandas as pd


def storage_per_sample(build: pathlib.Path) -> dict:
    """Compressed bytes per sample (final shard size / samples) and per kind of dataset."""
    tot = 0
    n = 0
    per_kind: dict[str, int] = {}
    for fh in sorted((build / "shards").glob("shard-*.h5")):
        tot += fh.stat().st_size
        with h5py.File(fh, "r") as f:
            for sid in f:
                n += 1
                for c in f[sid]["configs"]:
                    kind = f[sid]["configs"][c].attrs["kind"]
                    g = f[sid]["configs"][c]["outputs"]
                    per_kind[kind] = per_kind.get(kind, 0) + sum(g[d].id.get_storage_size() for d in g)
    return {"samples": n, "bytes_total": tot, "bytes_per_sample": tot / max(n, 1),
            "output_bytes_per_kind": {k: v / max(n, 1) for k, v in per_kind.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", required=True)
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--prefix", default="phase05")
    args = ap.parse_args()
    frames = []
    storage = {}
    for b in args.builds:
        b = pathlib.Path(b)
        if not (b / "index.parquet").exists():
            print(f"[skip] {b}: no index.parquet")
            continue
        df = pd.read_parquet(b / "index.parquet")
        df["build"] = b.name
        frames.append(df)
        storage[b.name] = storage_per_sample(b)
    df = pd.concat(frames, ignore_index=True)
    df["task"] = df["task_ids"]
    # --- solve time / memory per task and tier ---
    g = df.groupby(["tier", "task", "solver"]).agg(
        n=("solve_time_s", "size"), t_median=("solve_time_s", "median"), t_p90=("solve_time_s", lambda x: x.quantile(0.9)),
        t_max=("solve_time_s", "max"), rss_max_mb=("maxrss_mb", "max"), conv=("converged", "mean"), qc=("qc_pass", "mean"))
    print("\n### Solve time (s) and peak RSS (MB) per tier × task × solver\n")
    print(g.round(3).to_markdown())
    # --- per-sample totals (all configs) ---
    per_sample = df.groupby(["tier", "family", "sample_id"]).solve_time_s.sum().reset_index()
    ps = per_sample.groupby(["tier", "family"]).solve_time_s.agg(["count", "median", "mean", "max"])
    print("\n### Wall time per sample, all configs summed (s)\n")
    print(ps.round(2).to_markdown())
    # --- QC ---
    print("\n### QC flags\n")
    flags = df[df.qc_flags != ""].groupby(["tier", "kind", "qc_flags"]).size()
    print(flags.to_markdown() if len(flags) else "no flags")
    print("\nqc_pass rate:", round(df.qc_pass.mean(), 4), "| qc_trainval rate:", round(df.qc_trainval.mean(), 4))
    # --- storage ---
    print("\n### Storage (compressed HDF5)\n")
    rows = []
    for k, v in storage.items():
        rows.append({"build": k, "samples": v["samples"], "total_MB": v["bytes_total"] / 1e6,
                     "MB_per_sample": v["bytes_per_sample"] / 1e6,
                     **{f"out_{kk}_MB": vv / 1e6 for kk, vv in v["output_bytes_per_kind"].items()}})
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))
    # --- figure ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tiers = [t for t in ["S", "M", "L", "XL", "XXL"] if t in set(df.tier)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for task, sub in df.groupby("task"):
        med = [sub[sub.tier == t].solve_time_s.median() for t in tiers]
        axes[0].plot(tiers, med, "o-", label=task)
        rss = [sub[sub.tier == t].maxrss_mb.max() for t in tiers]
        axes[1].plot(tiers, rss, "o-", label=task)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median solve time (s)")
    axes[0].set_title("solve time vs tier")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("peak RSS (MB, process high-water)")
    axes[1].set_title("memory vs tier")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = pathlib.Path(args.out) / f"{args.prefix}_solve_times.png"
    fig.savefig(out, dpi=110)
    print("\nwrote", out)
    (pathlib.Path(args.out) / f"{args.prefix}_summary.json").write_text(json.dumps(
        {"time": g.reset_index().to_dict("records"), "storage": storage}, indent=1, default=float))


if __name__ == "__main__":
    main()
