#!/usr/bin/env python
"""Build docs/dataset_plan.md from configs/datasets/v1_0.yaml (via dataset_plan_tables.py) and the
Omniscape block-study results in data/builds/block_study/*.json. Re-run after any change.
Usage: python scripts/write_dataset_plan.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE = pathlib.Path(__file__).with_name("dataset_plan_template.md")


def tables(ladder: str, out_json: str, option: str | None = None) -> str:
    cmd = [sys.executable, "scripts/dataset_plan_tables.py", "--ladder", ladder, "--json", out_json]
    if option:
        cmd += ["--omniscape-option", option]
    return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=ROOT).stdout


def sect(text: str, title: str) -> str:
    i = text.find(title)
    j = text.find("\n### ", i + 1)
    return "\n".join(text[i:j if j > 0 else None].strip().splitlines()[1:]).strip()


def block_study() -> tuple[str, str]:
    files = sorted(glob.glob(str(ROOT / "data/builds/block_study/*.json")))
    if not files:
        return "*Pending: no block-study results yet.*", ""
    rows = []
    for f in files:
        d = json.load(open(f))
        c = d["cum_current"]
        rows.append({"tier": d["tier"], "radius": d["radius"], "blocks": f"{d['blocks'][0]} vs {d['blocks'][1]}",
                     "b/r": f"{d['blocks'][0] / d['radius']:.3f} vs {d['blocks'][1] / d['radius']:.3f}",
                     "sample": d["sample"][:8], "rel_L2_cum": c["rel_l2"], "max_diff/max_cum": c["max_abs_diff_over_max"],
                     "pearson_cum": c["pearson"], "rel_L2_normalized": d["normalized"]["rel_l2"],
                     "t_fine_s": round(d["t_a"]), "t_coarse_s": round(d["t_b"])})
    df = pd.DataFrame(rows)
    order = {"S": 0, "M": 1, "L": 2, "XL": 3, "XXL": 4}
    df = df.sort_values(["tier", "sample"], key=lambda c: c.map(order) if c.name == "tier" else c)
    verdict = []
    for (tier, blocks), g in sorted(df.groupby(["tier", "blocks"]), key=lambda kv: order[kv[0][0]]):
        a, b = (int(x) for x in blocks.split(" vs "))
        r = int(g.radius.iloc[0])
        mean, mx, pr = g.rel_L2_cum.mean(), g.rel_L2_cum.max(), g.pearson_cum.min()
        speed = g.t_fine_s.sum() / max(g.t_coarse_s.sum(), 1)
        if b < a:   # anchor: alternative is the finer (exact when b == 1) setting
            verdict.append(f"- **{tier} anchor, block {a} vs {b} (radius {r}, b/r {a / r:.2f} vs {b / r:.2f})**: the standard block {a} "
                           f"deviates from block {b} by {mean:.2%} mean / {mx:.2%} max relative L2 (Pearson ≥ {pr:.4f}); "
                           f"block {b} costs {1 / speed:.1f}× more.")
        else:       # coarsening test
            ok = mx < 0.01
            verdict.append(f"- **{tier} coarsening, block {a} vs {b} (radius {r}, b/r {a / r:.2f} vs {b / r:.2f})**: relative L2 "
                           f"{mean:.2%} mean / {mx:.2%} max (Pearson ≥ {pr:.4f}); block {b} is {speed:.1f}× cheaper → "
                           f"{'negligible (< 1 %): acceptable' if ok else 'NOT negligible (≥ 1 %): rejected'}.")
    return df.round(4).to_markdown(index=False), "\n".join(verdict)


def main() -> None:
    rec = tables("recommended", "docs/figures/dataset_plan_recommended.json")
    brief = tables("brief", "docs/figures/dataset_plan_brief.json")
    rec_p5 = tables("recommended", "docs/figures/dataset_plan_recommended_phase5.json", "phase5")
    jr = json.load(open(ROOT / "docs/figures/dataset_plan_recommended.json"))
    jb = json.load(open(ROOT / "docs/figures/dataset_plan_brief.json"))
    nl = [x for x in jr["landscapes"] if x["stratum"] != "(distinct tiles)"]
    nb = [x for x in jb["landscapes"] if x["stratum"] != "(distinct tiles)"]
    hc = {k: sum(x["landscapes"] for x in nl if x["stratum"] == f"hard:{k}")
          for k in ("high_contrast_1e4", "rmax_saturated", "narrow_corridor", "large_nodata")}
    study, verdict = block_study()
    out = TEMPLATE.read_text().format(
        land_rec=sect(rec, "### Landscapes per tier"), solves_rec=sect(rec, "### Solves per tier"),
        land_brief=sect(brief, "### Landscapes per tier"),
        tot_land_r=sum(x["landscapes"] for x in nl), tot_solves_r=sum(x["solves"] for x in jr["solves"]),
        tot_land_b=sum(x["landscapes"] for x in nb), tot_solves_b=sum(x["solves"] for x in jb["solves"]),
        cost_rec=sect(rec, "### Cost"), wall_rec=sect(rec, "### Wall-clock"),
        cost_brief=sect(brief, "### Cost"), wall_brief=sect(brief, "### Wall-clock"),
        cost_rec_p5=sect(rec_p5, "### Cost"), wall_rec_p5=sect(rec_p5, "### Wall-clock"),
        block_study=study, block_verdict=verdict,
        **{f"hc_{k}": f"{v:,}" for k, v in hc.items()})
    (ROOT / "docs/dataset_plan.md").write_text(out)
    print("wrote docs/dataset_plan.md")


if __name__ == "__main__":
    main()
