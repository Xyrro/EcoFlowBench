#!/usr/bin/env python
"""Extract covariate stacks for sampled tiles and maintain the tile manifest.

Reads <specs>/tile_specs.json, extracts each selected tile (network: WorldCover + Copernicus DEM
COG windows; local: gHM, GRIP4, HydroRIVERS), applies the usability QC, replaces rejected tiles
from the reserve list (same stratum preferred), writes GeoTIFFs + quicklooks, and appends rows to
<out>/tiles.parquet. Idempotent: existing accepted tiles are skipped.

Usage (login node): python scripts/extract_tiles.py --specs data/tiles/pilot --out data/tiles/pilot --workers 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess

import numpy as np

from ampscape.landscapes import real

for k, v in real.env_gdal().items():
    os.environ.setdefault(k, v)


def source_versions(sources_dir: pathlib.Path) -> dict:
    manifest = json.loads((sources_dir / "manifest.json").read_text()) if (sources_dir / "manifest.json").exists() else {}
    return {
        "worldcover": "ESA WorldCover 10m 2021 v200 (s3://esa-worldcover)",
        "copdem": "Copernicus DEM GLO-30 (s3://copernicus-dem-30m), GLO-90 fallback",
        "ghm": "gHM v1 1km (figshare 7283087 v1)",
        "grip4": "GRIP4 regional shapefiles (PBL, 2018)",
        "hydrorivers": "HydroRIVERS v1.0",
        "downloads": {k: {"sha256": v["sha256"], "downloaded_utc": v["downloaded_utc"]} for k, v in manifest.items()},
    }


def quicklook(path: str, channels: dict, spec: real.TileSpec, qc: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, name in zip(axes.ravel(), real.CHANNELS, strict=True):
        a = channels[name].astype(float)
        cmap = "tab20" if name in ("landcover", "road_class", "river_order") else "viridis"
        if name in ("road_distance", "river_distance"):
            a = np.log10(1 + a)
        ax.imshow(a, cmap=cmap, interpolation="nearest")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"{spec.tile_id}  ({spec.lat:.2f}, {spec.lon:.2f})  {spec.stratum.get('biome_name','')}  "
                 f"unusable={qc['frac_unusable']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def process(spec_d: dict, sources: real.SourcePaths, out: pathlib.Path, versions: dict) -> dict:
    spec = real.TileSpec(**spec_d)
    tif = out / "tiles" / spec.tier / f"{spec.tile_id}.tif"
    channels, grid, qc = real.extract_tile(spec, sources)
    row = {"tile_id": spec.tile_id, "lat": spec.lat, "lon": spec.lon, "tier": spec.tier, "size": spec.size,
           "pixel_m": spec.pixel_m, "epsg": grid.epsg, "transform": json.dumps(list(grid.transform)[:6]),
           **{f"qc_{k}": v for k, v in qc.items()},
           **{k: v for k, v in spec.stratum.items() if k in ("biome_num", "biome_name", "realm", "ecoregion_id",
                                                              "ecoregion_name", "ghm", "ghm_tercile", "stratum", "grip_region")},
           "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "path": None, "sha256": None}
    if qc["accept"]:
        row["sha256"] = real.write_tile(str(tif), channels, grid, spec, qc, versions)
        row["path"] = str(tif.relative_to(out))
        ql = out / "quicklooks" / f"{spec.tile_id}.png"
        ql.parent.mkdir(parents=True, exist_ok=True)
        quicklook(str(ql), channels, spec, qc)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="data/sources")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh", action="store_true",
                    help="re-extract every tile already in the manifest (same tile set, new code)")
    ap.add_argument("--retry-rejected", action="store_true",
                    help="drop previously rejected rows from the manifest and extract them again")
    args = ap.parse_args()
    import pandas as pd
    from joblib import Parallel, delayed

    out = pathlib.Path(args.out)
    specs = json.loads((pathlib.Path(args.specs) / "tile_specs.json").read_text())
    sources = real.local_sources_from_dir(args.sources)
    versions = source_versions(pathlib.Path(args.sources))
    versions["pipeline_git_sha"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest_path = out / "tiles.parquet"
    done = pd.read_parquet(manifest_path) if manifest_path.exists() else pd.DataFrame()
    if args.retry_rejected and len(done):
        done = done[done["qc_accept"]].reset_index(drop=True)
    done_ids = set(done["tile_id"]) if len(done) else set()

    if args.refresh and len(done):
        by_id = {s["tile_id"]: s for s in specs["selected"] + specs["reserve"]}
        todo = [by_id[t] for t in done["tile_id"] if t in by_id]
        done = done.iloc[0:0]
        done_ids = set()
    else:
        todo = [s for s in specs["selected"] if s["tile_id"] not in done_ids]
    reserve = [s for s in specs["reserve"] if s["tile_id"] not in done_ids]
    if args.limit:
        todo = todo[: args.limit]
    rows = list(done.to_dict("records")) if len(done) else []
    n_target = max(len(specs["selected"]), len(todo))
    round_ = 0
    while todo:
        round_ += 1
        print(f"round {round_}: extracting {len(todo)} tiles with {args.workers} workers")
        new = Parallel(n_jobs=args.workers, prefer="threads")(delayed(process)(s, sources, out, versions) for s in todo)
        rows.extend(new)
        pd.DataFrame(rows).to_parquet(manifest_path, index=False)
        rejected = [r for r in new if not r["qc_accept"]]
        for r in rejected:
            print(f"  rejected {r['tile_id']}: unusable={r['qc_frac_unusable']:.2f} dem_nan={r['qc_frac_dem_nan']:.2f}")
        n_ok = sum(1 for r in rows if r["qc_accept"])
        need = n_target - n_ok
        todo = []
        if need > 0 and reserve:
            # prefer reserve tiles from the same strata as the rejected ones
            wanted = [r["stratum"] for r in rejected]
            pick = [s for s in reserve if s["stratum"]["stratum"] in wanted][:need]
            pick += [s for s in reserve if s not in pick][: need - len(pick)]
            reserve = [s for s in reserve if s not in pick]
            todo = pick
        if args.limit:
            break
    df = pd.DataFrame(rows)
    ok = df[df["qc_accept"]]
    print(f"accepted {len(ok)} / {len(df)} tiles; biomes={ok['biome_num'].nunique()} realms={ok['realm'].nunique()} "
          f"strata={ok['stratum'].nunique()}")
    print(ok.groupby(["biome_name"]).size().to_string())


if __name__ == "__main__":
    main()
