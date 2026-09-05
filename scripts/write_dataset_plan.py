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


def tables(ladder: str, out_json: str) -> str:
    return subprocess.run([sys.executable, "scripts/dataset_plan_tables.py", "--ladder", ladder, "--json", out_json],
                          capture_output=True, text=True, check=True, cwd=ROOT).stdout


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
                     "sample": d["sample"][:8], "rel_L2_cum": c["rel_l2"], "max_diff/max_cum": c["max_abs_diff_over_max"],
                     "pearson_cum": c["pearson"], "rel_L2_normalized": d["normalized"]["rel_l2"],
                     "t_fine_s": round(d["t_a"]), "t_coarse_s": round(d["t_b"])})
    df = pd.DataFrame(rows)
    verdict = []
    for (tier, blocks), g in df.groupby(["tier", "blocks"]):
        ok = g.rel_L2_cum.max() < 0.01
        verdict.append(f"- **{tier}, block {blocks} (radius {int(g.radius.iloc[0])})**: relative L2 of `cum_current` "
                       f"{g.rel_L2_cum.mean():.2%} mean / {g.rel_L2_cum.max():.2%} max over {len(g)} samples, Pearson ≥ "
                       f"{g.pearson_cum.min():.4f}; the coarse block is {g.t_fine_s.sum() / max(g.t_coarse_s.sum(), 1):.1f}× cheaper → "
                       f"{'negligible (< 1 %): coarse block acceptable' if ok else 'NOT negligible (≥ 1 %): keep the fine block'}.")
    return df.round(4).to_markdown(index=False), "\n".join(verdict)


def main() -> None:
    rec = tables("recommended", "docs/figures/dataset_plan_recommended.json")
    brief = tables("brief", "docs/figures/dataset_plan_brief.json")
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
        block_study=study, block_verdict=verdict,
        **{f"hc_{k}": f"{v:,}" for k, v in hc.items()})
    (ROOT / "docs/dataset_plan.md").write_text(out)
    print("wrote docs/dataset_plan.md")


if __name__ == "__main__":
    main()
