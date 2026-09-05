#!/usr/bin/env python
"""Expand configs/datasets/v1_0.yaml into the coverage matrix and cost tables for docs/dataset_plan.md.

Usage: python scripts/dataset_plan_tables.py [--config configs/datasets/v1_0.yaml] [--ladder recommended|brief]
Prints Markdown; `--json out.json` also dumps the numbers.
"""
from __future__ import annotations

import argparse
import json
import math

import pandas as pd
import yaml

TIERS = ["S", "M", "L", "XL", "XXL"]
PIX = {"S": 128**2, "M": 256**2, "L": 512**2, "XL": 1024**2, "XXL": 2048**2}


def per_solve_seconds(cfg: dict, config: str, tier: str) -> float:
    m = cfg["cost_measured"][config]
    if tier in m:
        return float(m[tier])
    # scale from S linearly in pixels (measured exponents 1.00–1.07 for these configs)
    return float(m["S"]) * PIX[tier] / PIX["S"]


def landscape_counts(cfg: dict, ladder: dict) -> pd.DataFrame:
    rows = []
    fs = cfg["family_share"]
    n_tables = len(cfg["resistance_tables"])
    for t in TIERS:
        n = ladder[t]
        n_syn = int(round(n * fs["synthetic"]))
        n_real = n - n_syn
        n_tiles = math.ceil(n_real / n_tables)
        for gen, frac in cfg["generator_mix"].items():
            if gen == "hard":
                for hc, hfrac in cfg["hard_cases"].items():
                    rows.append({"tier": t, "family": "synthetic", "stratum": f"hard:{hc}", "table": "-", "landscapes": int(round(n_syn * frac * hfrac))})
            else:
                rows.append({"tier": t, "family": "synthetic", "stratum": gen, "table": "-", "landscapes": int(round(n_syn * frac))})
        for tab in cfg["resistance_tables"]:
            rows.append({"tier": t, "family": "real", "stratum": f"tiles={n_tiles}", "table": tab, "landscapes": n_tiles})
    return pd.DataFrame(rows)


def config_counts(cfg: dict, land: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, fam), g in land.groupby(["tier", "family"]):
        n = int(g.landscapes.sum())
        for c, spec in cfg["configs_per_landscape"].items():
            if fam not in spec["families"]:
                continue
            rows.append({"tier": t, "family": fam, "config": c, "tasks": ",".join(spec["tasks"]), "solves": int(round(n * spec["per_landscape"]))})
    abl = cfg["four_neighbour_ablation"]
    n_syn_s = int(land[(land.tier == abl["tier"]) & (land.family == "synthetic")].landscapes.sum())
    rows.append({"tier": abl["tier"], "family": "synthetic", "config": "points (4-neighbour ablation)", "tasks": "T1,T2", "solves": int(round(n_syn_s * abl["fraction_of_synthetic"]))})
    return pd.DataFrame(rows)


def cost_table(cfg: dict, land: pd.DataFrame, conf: pd.DataFrame, omni_scale: dict | None = None) -> pd.DataFrame:
    oh = 1 + cfg["cost_measured"]["overhead_fraction"]
    rows = []
    for t in TIERS:
        n_syn = int(land[(land.tier == t) & (land.family == "synthetic")].landscapes.sum())
        n_real = int(land[(land.tier == t) & (land.family == "real")].landscapes.sum())
        sec = 0.0
        for r in conf[conf.tier == t].itertuples():
            key = "points" if r.config.startswith("points") else ("wall_to_wall" if r.config.startswith("wall") else r.config)
            s = per_solve_seconds(cfg, key, t)
            if key == "omniscape" and omni_scale and t in omni_scale:
                s *= omni_scale[t]
            sec += s * r.solves
        sec *= oh
        gb = (n_syn * cfg["storage_mb_per_landscape"]["synthetic"][t] + n_real * cfg["storage_mb_per_landscape"]["real"][t]) / 1000
        per_land = sec / max(n_syn + n_real, 1)
        rows.append({"tier": t, "landscapes": n_syn + n_real, "cpu_s_per_landscape": round(per_land, 1),
                     "peak_rss_gb": round(cfg["peak_rss_mb"][t] / 1000, 1), "cpu_hours": round(sec / 3600, 0),
                     "storage_gb": round(gb, 1)})
    df = pd.DataFrame(rows)
    tot = {"tier": "total", "landscapes": int(df.landscapes.sum()), "cpu_s_per_landscape": None, "peak_rss_gb": None,
           "cpu_hours": float(df.cpu_hours.sum()), "storage_gb": round(float(df.storage_gb.sum()), 1)}
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)


def wallclock(cpu_hours: float) -> pd.DataFrame:
    rows = [{"concurrent_cores": c, "wall_days": round(cpu_hours / c / 24, 1)} for c in (100, 500, 1000)]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasets/v1_0.yaml")
    ap.add_argument("--ladder", default="recommended", choices=["recommended", "brief"])
    ap.add_argument("--omniscape-scale", default="", help="e.g. XL=0.265,XXL=0.258 to price a coarser block")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ladder = cfg[f"{args.ladder}_ladder"]
    land = landscape_counts(cfg, ladder)
    conf = config_counts(cfg, land)
    oscale = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.omniscape_scale.split(",") if kv}
    cost = cost_table(cfg, land, conf, oscale or None)
    print(f"## Ladder: {args.ladder} {ladder}\n")
    print("### Landscapes per tier × family × stratum/table\n")
    piv = land.pivot_table(index=["family", "stratum", "table"], columns="tier", values="landscapes", aggfunc="sum", fill_value=0)[TIERS]
    piv["total"] = piv.sum(axis=1)
    print(piv.to_markdown())
    print("\n### Solves per tier × family × source configuration (task instances)\n")
    piv2 = conf.pivot_table(index=["family", "config", "tasks"], columns="tier", values="solves", aggfunc="sum", fill_value=0)[TIERS]
    piv2["total"] = piv2.sum(axis=1)
    print(piv2.to_markdown())
    print(f"\nTotal landscapes: {int(land.landscapes.sum()):,}; total solves: {int(conf.solves.sum()):,}; "
          f"real tiles: {land[land.family == 'real'].groupby('tier').landscapes.first().to_dict()}")
    print("\n### Cost (measured Phase 5, CHOLMOD, single-threaded solves, +15 % overhead)\n")
    print(cost.to_markdown(index=False))
    total_h = float(cost[cost.tier == "total"].cpu_hours.iloc[0])
    print("\n### Wall-clock for the full ladder\n")
    print(wallclock(total_h).to_markdown(index=False))
    if args.json:
        json.dump({"ladder": ladder, "landscapes": land.to_dict("records"), "solves": conf.to_dict("records"),
                   "cost": cost.to_dict("records"), "total_cpu_hours": total_h}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
