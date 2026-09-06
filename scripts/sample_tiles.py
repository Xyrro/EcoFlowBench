#!/usr/bin/env python
"""Stratified sampling of real-tile centres (brief §4.2).

Draws uniform-on-sphere land points, joins RESOLVE biome/realm, samples gHM, assigns gHM
terciles, and balances across biome × realm × tercile strata. Writes:
  <out>/candidates.parquet   all attributed candidates (for resampling rejected tiles)
  <out>/tile_specs.json      the selected tiles + a reserve list, plus tercile edges and seed

Usage (login node, needs data/sources unzipped):
  python scripts/sample_tiles.py --out data/tiles/pilot --n 50 --reserve 100 --seed 20260905 \
      --realms Nearctic Neotropic Afrotropic Australasia Oceania Palearctic --grip-regions 1 2 3 5 7
"""
from __future__ import annotations

import argparse
import json
import pathlib

import geopandas as gpd
import numpy as np

from ampscape.landscapes import sampling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="data/sources")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--reserve", type=int, default=100)
    ap.add_argument("--candidates", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--realms", nargs="*", default=None)
    ap.add_argument("--grip-regions", nargs="*", type=int, default=None,
                    help="only keep candidates whose GRIP4 region file is downloaded")
    ap.add_argument("--min-biomes", type=int, default=5)
    ap.add_argument("--tier", default="S")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--pixel-m", type=float, default=100.0)
    args = ap.parse_args()

    src = pathlib.Path(args.sources)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    eco = gpd.read_file(src / "Ecoregions2017" / "Ecoregions2017.shp")
    pts = sampling.sample_land_points(eco, args.candidates, rng)
    if args.realms:
        pts = pts[pts["REALM"].isin(args.realms)].reset_index(drop=True)
    ghm_path = next((src / "gHM").rglob("gHM.tif"))
    pts["ghm"] = sampling.sample_ghm(str(ghm_path), pts["lat"].to_numpy(), pts["lon"].to_numpy())
    pts = pts[np.isfinite(pts["ghm"])].reset_index(drop=True)
    terc, edges = sampling.assign_terciles(pts["ghm"].to_numpy())
    pts["ghm_tercile"] = terc
    pts["grip_region"] = [sampling.grip_region(r, la, lo) for r, la, lo in zip(pts["REALM"], pts["lat"], pts["lon"], strict=True)]
    if args.grip_regions:
        pts = pts[pts["grip_region"].isin(args.grip_regions)].reset_index(drop=True)

    cands = [sampling.Candidate(float(r.lat), float(r.lon), int(r.BIOME_NUM), str(r.BIOME_NAME), str(r.REALM),
                                int(r.ECO_ID), str(r.ECO_NAME), float(r.ghm), int(r.ghm_tercile),
                                None if r.grip_region is None or np.isnan(r.grip_region) else int(r.grip_region))
             for r in pts.itertuples()]
    chosen = sampling.balance_strata(cands, args.n + args.reserve, rng, min_biomes=args.min_biomes)
    selected, reserve = chosen[: args.n], chosen[args.n:]

    def spec(i: int, c: sampling.Candidate) -> dict:
        return {"tile_id": f"{args.tier}_{c.realm[:3].lower()}_b{c.biome_num:02d}_{i:04d}", "lat": c.lat, "lon": c.lon,
                "tier": args.tier, "size": args.size, "pixel_m": args.pixel_m, "stratum": c.to_dict()}

    specs = {
        "seed": args.seed, "ghm_tercile_edges": edges, "n_candidates_attributed": len(cands),
        "selected": [spec(i, c) for i, c in enumerate(selected)],
        "reserve": [spec(1000 + i, c) for i, c in enumerate(reserve)],
    }
    (out / "tile_specs.json").write_text(json.dumps(specs, indent=1))
    pts.drop(columns="geometry").to_parquet(out / "candidates.parquet", index=False)
    from collections import Counter

    print(f"candidates attributed: {len(cands)}; selected {len(selected)} + reserve {len(reserve)}")
    print("tercile edges:", edges)
    print("biomes:", sorted(Counter(c.biome_name for c in selected).items()))
    print("realms:", sorted(Counter(c.realm for c in selected).items()))
    print("terciles:", sorted(Counter(c.ghm_tercile for c in selected).items()))
    print("strata:", len({c.stratum for c in selected}))


if __name__ == "__main__":
    main()
