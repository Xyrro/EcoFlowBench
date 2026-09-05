#!/usr/bin/env python
"""Figures for the real-tile manifest: world map of tile centres by biome, and a covariate gallery.

Usage: python scripts/plot_tiles.py --manifest data/tiles/pilot/tiles.parquet --out docs/figures --prefix pilot
"""
from __future__ import annotations

import argparse
import pathlib

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecoflowbench.landscapes import real  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="pilot")
    ap.add_argument("--ecoregions", default="data/sources/Ecoregions2017/Ecoregions2017.shp")
    ap.add_argument("--n-gallery", type=int, default=6)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.manifest)
    ok = df[df["qc_accept"]].copy()

    # --- map ---
    fig, ax = plt.subplots(figsize=(11, 5.2))
    try:
        eco = gpd.read_file(args.ecoregions)[["BIOME_NUM", "geometry"]]
        eco["geometry"] = eco.geometry.simplify(0.1)
        eco.plot(ax=ax, column="BIOME_NUM", cmap="tab20", alpha=0.25, linewidth=0)
    except Exception as e:  # noqa: BLE001
        print("ecoregion background skipped:", e)
    biomes = sorted(ok["biome_num"].unique())
    cmap = plt.get_cmap("tab20")
    for i, b in enumerate(biomes):
        sub = ok[ok["biome_num"] == b]
        ax.scatter(sub["lon"], sub["lat"], s=28, color=cmap(i % 20), edgecolor="k", linewidth=0.4,
                   label=f"{b:02d} {sub['biome_name'].iloc[0][:34]} ({len(sub)})", zorder=3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title(f"EcoFlowBench {args.prefix} tiles: {len(ok)} accepted, {ok['biome_num'].nunique()} biomes, "
                 f"{ok['realm'].nunique()} realms, {ok['stratum'].nunique()} strata")
    ax.legend(fontsize=6, loc="lower left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / f"{args.prefix}_tiles_map.png", dpi=120)
    plt.close(fig)

    # --- gallery ---
    rng = np.random.default_rng(0)
    pick = ok.sample(min(args.n_gallery, len(ok)), random_state=int(rng.integers(1 << 31)))
    root = pathlib.Path(args.manifest).parent
    fig, axes = plt.subplots(len(pick), 4, figsize=(10, 2.4 * len(pick)))
    for row, (_, r) in zip(axes, pick.iterrows(), strict=True):
        ch, _ = real.read_tile(str(root / r["path"]))
        panels = [("landcover", ch["landcover"], "tab20"), ("elevation", ch["elevation"], "terrain"),
                  ("log10 road dist", np.log10(1 + ch["road_distance"]), "viridis"), ("ghm", ch["ghm"], "magma")]
        for ax, (name, a, cm) in zip(row, panels, strict=True):
            ax.imshow(a, cmap=cm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, fontsize=7)
        row[0].set_ylabel(f"{r['tile_id']}\n{r['biome_name'][:28]}\n({r['lat']:.1f}, {r['lon']:.1f})", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / f"{args.prefix}_tiles_gallery.png", dpi=90)
    print("wrote", out / f"{args.prefix}_tiles_map.png", "and gallery")
    # stratum table
    tab = ok.groupby(["realm", "biome_num", "ghm_tercile"]).size().rename("n").reset_index()
    tab.to_csv(out / f"{args.prefix}_strata.csv", index=False)


if __name__ == "__main__":
    main()
