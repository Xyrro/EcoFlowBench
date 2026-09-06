#!/usr/bin/env python
"""Generate every source configuration for every (tile, table) resistance raster (Phase 4).

Writes <out>/sources/<table_id>/<tile_id>.npz (focal masks int32, source/ground rasters, tables as
JSON) and <out>/sources.parquet (one row per tile × table × kind with connectivity results and
placement summaries). Also runs the same generators on N synthetic landscapes for a sanity set.

Usage: python scripts/build_sources.py --tiles data/tiles/pilot --out data/tiles/pilot \
           --config configs/tasks/sources_default.yaml --seed 20260905 --synthetic 50 \
           --figure docs/figures/pilot_sources_gallery.png
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import rasterio

from ampscape.landscapes.real import read_tile
from ampscape.landscapes.synthetic import sample_landscape
from ampscape.sources import SourceConfig, generate_all


def flatten(rows, tile_id, table_id, family, samples):
    for kind, s in samples.items():
        m = s.meta
        rows.append({
            "tile_id": tile_id, "table_id": table_id, "family": family, "kind": kind, "k": s.k,
            "connected": m["connected"], "n_components_touched": m["n_components_touched"],
            "n_graph_components": m["n_graph_components"], "placement": m.get("placement"),
            "n_anywhere": m.get("n_anywhere"), "n_low_resistance": m.get("n_low_resistance"),
            "separation_relaxations": m.get("separation_relaxations"),
            "min_separation_px_used": m.get("min_separation_px_used"),
            "n_source_pixels": m.get("n_source_pixels"), "n_ground_pixels": m.get("n_ground_pixels"),
            "ground_mode": (m.get("ground") or {}).get("mode"), "n_eligible_patches": m.get("n_eligible_patches"),
            "seed": m["seed"], "source_config_sha256": m["source_config"]["sha256"],
        })


def save_npz(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, meta = {}, {}
    for kind, s in samples.items():
        if s.focal_mask is not None:
            arrays[f"{kind}/focal_mask"] = s.focal_mask
        if s.source_strength is not None:
            arrays[f"{kind}/source_strength"] = s.source_strength
        if s.ground is not None:
            arrays[f"{kind}/ground"] = s.ground
        meta[kind] = {"focal_table": s.focal_table, "meta": s.meta}
    np.savez_compressed(path, **arrays, meta=json.dumps(meta, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/tasks/sources_default.yaml")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--synthetic", type=int, default=50)
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()
    cfg = SourceConfig.from_yaml(args.config)
    tiles = pathlib.Path(args.tiles)
    out = pathlib.Path(args.out)
    res = pd.read_parquet(tiles / "resistance.parquet")
    tdf = pd.read_parquet(tiles / "tiles.parquet").set_index("tile_id")
    rows, gallery = [], []
    ss = np.random.SeedSequence(args.seed)
    for i, r in enumerate(res.itertuples()):
        with rasterio.open(tiles / r.path) as src:
            R = src.read(1)
            nd = src.read(2) > 0.5
        cov, _ = read_tile(str(tiles / tdf.loc[r.tile_id, "path"]))
        tier_cfg = cfg.for_tier(str(tdf.loc[r.tile_id, "tier"]))
        seed = int(np.random.default_rng(ss.spawn(1)[0]).integers(0, 2**31 - 1)) if False else args.seed * 1000 + i
        samples = generate_all(R, nd, tier_cfg, seed, landcover=cov["landcover"])
        save_npz(out / "sources" / r.table_id / f"{r.tile_id}.npz", samples)
        flatten(rows, r.tile_id, r.table_id, "real", samples)
        if len(gallery) < 4 and r.table_id == "large_mammal":
            gallery.append((r.tile_id, R, nd, samples))
    for j in range(args.synthetic):
        ls = sample_landscape(args.seed + j, (128, 128))
        samples = generate_all(ls.resistance, ls.nodata_mask, cfg, args.seed * 1000 + 100000 + j)
        flatten(rows, f"synthetic_{j:04d}", "synthetic", "synthetic", samples)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "sources.parquet", index=False)
    print(f"{len(df)} configurations; all connected: {bool(df['connected'].all())}")
    print(df.groupby(["family", "kind"]).agg(n=("kind", "size"), connected=("connected", "mean"), k_mean=("k", "mean"),
                                             relax=("separation_relaxations", "mean")).round(3).to_string())
    pts = df[df.kind == "points"]
    print("points placement:", pts["placement"].value_counts().to_dict(),
          "| anywhere fraction of nodes:", round(pts["n_anywhere"].sum() / pts["k"].sum(), 3))
    print("regions available on", int((df.kind == "regions").sum()), "of", int((df.kind == "points").sum()), "landscapes")
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kinds = ["points", "wall_to_wall_NS", "regions", "advanced", "omniscape"]
        fig, axes = plt.subplots(len(gallery), len(kinds) + 1, figsize=(2.3 * (len(kinds) + 1), 2.4 * len(gallery)))
        for i, (tile_id, R, nd, samples) in enumerate(gallery):
            axes[i, 0].imshow(np.ma.masked_array(np.log10(R), nd), cmap="magma", interpolation="nearest")
            axes[i, 0].set_ylabel(f"{tile_id}\nlarge_mammal", fontsize=7)
            for j, kind in enumerate(kinds, start=1):
                ax = axes[i, j]
                ax.imshow(np.log10(R), cmap="gray", alpha=0.6, interpolation="nearest")
                s = samples.get(kind)
                if s is None:
                    ax.text(0.5, 0.5, "no eligible\npatches", ha="center", transform=ax.transAxes, fontsize=7)
                elif s.focal_mask is not None:
                    ax.imshow(np.ma.masked_equal(s.focal_mask, 0), cmap="tab10", interpolation="nearest", vmin=1, vmax=10)
                    for t in s.focal_table:
                        ax.plot(t["col"], t["row"], "w+", ms=6)
                else:
                    ax.imshow(np.ma.masked_equal(s.source_strength, 0), cmap="viridis", interpolation="nearest")
                    if s.ground is not None:
                        ax.imshow(np.ma.masked_equal(s.ground, 0), cmap="autumn", interpolation="nearest")
                if i == 0:
                    ax.set_title(kind, fontsize=8)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.figure, dpi=90)
        print("wrote", args.figure)


if __name__ == "__main__":
    main()
