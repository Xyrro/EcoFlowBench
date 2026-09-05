#!/usr/bin/env python
"""Full-budget extrapolation from measured per-sample solve times and storage (Phase 5 report).

Reads the build indexes (mini + probes), fits per-tier per-config median wall times, and prints
CPU-hour + storage totals for (a) the brief's original ladder (§4.3) and (b) a proposed ladder
that fits a target CPU-hour budget. Tiers without measurements are extrapolated by a power law in
pixel count fitted on the measured tiers.

Usage: python scripts/budget_extrapolation.py --builds data/builds/mini data/builds/probe_M ... \
          --target-cpu-hours 500 --cpus-per-job 4 --overhead 0.15
"""
from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np
import pandas as pd

TIERS = ["S", "M", "L", "XL", "XXL"]
PIX = {"S": 128**2, "M": 256**2, "L": 512**2, "XL": 1024**2, "XXL": 2048**2}
BRIEF_LADDER = {"S": 100_000, "M": 50_000, "L": 10_000, "XL": 2_000, "XXL": 200}
# configs solved per landscape in the mini design (points, NS, EW, T3, T4; regions on ~40 % of real tiles)
CONFIGS_PER_SAMPLE = {"points": 1.0, "wall_to_wall": 2.0, "advanced": 1.0, "omniscape": 1.0, "regions": 0.2}


def fit_powerlaw(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    b, a = np.polyfit(x, y, 1) if len(x) > 1 else (1.0, y[0] - x[0])
    return float(np.exp(a)), float(b)


def storage_per_sample(build: pathlib.Path) -> float | None:
    files = sorted((build / "shards").glob("*.h5"))
    if not files:
        return None
    n = 0
    tot = 0
    for fh in files:
        tot += fh.stat().st_size
        with h5py.File(fh, "r") as f:
            n += len(f)
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", required=True)
    ap.add_argument("--target-cpu-hours", type=float, default=500.0)
    ap.add_argument("--cpus-per-job", type=int, default=4)
    ap.add_argument("--overhead", type=float, default=0.15, help="fraction added for JIT, I/O, QC, finalize")
    ap.add_argument("--out", default="docs/figures/phase05_budget.json")
    ap.add_argument("--omniscape-scale", default="", help="per-tier multipliers for the omniscape time, e.g. XL=0.27,XXL=0.26 "
                    "(adopted block sizes vs the measured ones)")
    args = ap.parse_args()
    frames, stor = [], {}
    for b in args.builds:
        b = pathlib.Path(b)
        if (b / "index.parquet").exists():
            df = pd.read_parquet(b / "index.parquet")
            frames.append(df)
            s = storage_per_sample(b)
            if s:
                stor.setdefault(df.tier.iloc[0], []).append(s)
    df = pd.concat(frames, ignore_index=True)
    # median wall time per (tier, kind)
    med = df.groupby(["tier", "kind"]).solve_time_s.median().unstack("kind")
    print("\n### Measured median solve time per config (s)\n")
    print(med.round(3).to_markdown())
    # per-tier seconds per landscape = Σ configs × multiplicity; missing kinds at a tier are extrapolated
    oscale = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.omniscape_scale.split(",") if kv}
    if oscale and "omniscape" in med:
        for t, f in oscale.items():
            if t in med.index:
                med.loc[t, "omniscape"] = med.loc[t, "omniscape"] * f
        print(f"\n(omniscape times scaled for adopted window sizes: {oscale})")
    per_tier_s = {}
    fits = {}
    for kind, mult in CONFIGS_PER_SAMPLE.items():
        if kind not in med:
            continue
        have = med[kind].dropna()
        xs = [PIX[t] for t in have.index]
        a, bexp = fit_powerlaw(xs, have.values)
        fits[kind] = (a, bexp)
        for t in TIERS:
            val = have[t] if t in have.index else a * PIX[t] ** bexp
            per_tier_s[t] = per_tier_s.get(t, 0.0) + mult * val
    print("\n### Power-law fits t = a·pixels^b per config (b ≈ 1 means linear in pixels)\n")
    for k, (_a, bexp) in fits.items():
        print(f"- {k}: b = {bexp:.2f}")
    # storage per sample (compressed MB) with fallback power law
    st = {t: float(np.mean(v)) for t, v in stor.items()}
    if len(st) >= 2:
        a_s, b_s = fit_powerlaw([PIX[t] for t in st], [st[t] for t in st])
    else:
        a_s, b_s = (st.get("S", 1.2e6) / PIX["S"], 1.0)
    st_all = {t: st.get(t, a_s * PIX[t] ** b_s) for t in TIERS}

    def ladder_table(ladder: dict, title: str):
        rows = []
        tot_h = tot_gb = 0.0
        for t, n in ladder.items():
            sec = per_tier_s[t] * n * (1 + args.overhead)
            cpu_h = sec * args.cpus_per_job / 3600.0
            gb = st_all[t] * n / 1e9
            tot_h += cpu_h
            tot_gb += gb
            rows.append({"tier": t, "samples": n, "s_per_sample": round(per_tier_s[t], 1), "job_h": round(sec / 3600, 1),
                         "cpu_h": round(cpu_h, 1), "MB_per_sample": round(st_all[t] / 1e6, 2), "GB": round(gb, 1)})
        print(f"\n### {title}\n")
        print(pd.DataFrame(rows).to_markdown(index=False))
        print(f"\n**Total: {tot_h:,.0f} CPU-hours ({args.cpus_per_job} cores/job, +{args.overhead:.0%} overhead), {tot_gb:,.0f} GB**")
        return tot_h, tot_gb

    brief_h, brief_gb = ladder_table(BRIEF_LADDER, "Brief's original ladder (§4.3)")
    # proposed: keep the brief's tier proportions of *compute* roughly balanced; scale to target
    weights = {"S": 0.45, "M": 0.25, "L": 0.15, "XL": 0.10, "XXL": 0.05}   # share of the CPU budget per tier
    proposed = {}
    for t, w in weights.items():
        sec_per = per_tier_s[t] * (1 + args.overhead) * args.cpus_per_job / 3600.0
        n = int(args.target_cpu_hours * w / sec_per)
        proposed[t] = int(round(n, -2)) if n >= 100 else max(n, 1)
    prop_h, prop_gb = ladder_table(proposed, f"Proposed ladder fitting ≈ {args.target_cpu_hours:.0f} CPU-hours (budget shares S/M/L/XL/XXL = 45/25/15/10/5 %)")
    pathlib.Path(args.out).write_text(json.dumps({"per_tier_seconds_per_sample": per_tier_s, "storage_bytes_per_sample": st_all,
                                                  "fits": fits, "brief": {"cpu_h": brief_h, "gb": brief_gb, "ladder": BRIEF_LADDER},
                                                  "proposed": {"cpu_h": prop_h, "gb": prop_gb, "ladder": proposed}}, indent=1))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
