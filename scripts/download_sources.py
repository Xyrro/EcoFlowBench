#!/usr/bin/env python
"""Download upstream source archives to $AMPSCAPE_DATA/sources with a sha256 manifest.

Login node only (network). Idempotent: existing files with matching size are skipped.
Every download is recorded in sources/manifest.json (url, bytes, sha256, utc timestamp) so the
datasheet can state exactly which upstream files were used.

Usage:
  python scripts/download_sources.py --set pilot      # RESOLVE, gHM, HydroRIVERS + GRIP4 for pilot regions
  python scripts/download_sources.py --set full       # all GRIP4 / HydroRIVERS regions (asks: > 5 GB gate)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

SOURCES = {
    "resolve_ecoregions_2017": {
        "url": "https://storage.googleapis.com/teow2016/Ecoregions2017.zip",
        "file": "Ecoregions2017.zip", "license": "CC BY 4.0", "sets": ["pilot", "full"]},
    "ghm_v1_1km": {
        "url": "https://ndownloader.figshare.com/files/13448294",
        "file": "gHM.zip", "license": "CC BY 4.0", "sets": ["pilot", "full"]},
}
HYDRO = "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_{r}_shp.zip"
for r in ["na", "sa", "af", "au", "eu", "as", "ar", "si", "gr"]:
    SOURCES[f"hydrorivers_v10_{r}"] = {
        "url": HYDRO.format(r=r), "file": f"HydroRIVERS_v10_{r}_shp.zip", "license": "CC BY 4.0",
        "sets": ["full"] + (["pilot"] if r in ("na", "sa", "af", "au", "eu") else [])}
GRIP = "https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_Region{n}_vector_shp.zip"
for n in range(1, 8):
    SOURCES[f"grip4_region{n}"] = {
        "url": GRIP.format(n=n), "file": f"GRIP4_Region{n}_vector_shp.zip", "license": "CC0 / CC BY 4.0 (see docs/licenses.md)",
        "sets": ["full"] + (["pilot"] if n in (1, 2, 3, 5, 7) else [])}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--only", nargs="*", help="subset of source keys")
    ap.add_argument("--dest", default=os.path.join(os.environ.get("AMPSCAPE_DATA", "data"), "sources"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    keys = [k for k, v in SOURCES.items() if args.set in v["sets"]]
    if args.only:
        keys = [k for k in keys if k in args.only]
    for k in keys:
        src = SOURCES[k]
        out = dest / src["file"]
        if out.exists() and k in manifest and manifest[k].get("bytes") == out.stat().st_size:
            print(f"[skip] {k}: {out.name} ({out.stat().st_size/1e6:.0f} MB)")
            continue
        print(f"[get ] {k}: {src['url']}")
        if args.dry_run:
            continue
        tmp = out.with_suffix(out.suffix + ".part")
        cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10", "-C", "-",
               "-A", "Mozilla/5.0 (AmpScape downloader)", "-o", str(tmp), src["url"]]
        subprocess.run(cmd, check=True)
        shutil.move(tmp, out)
        manifest[k] = {
            "url": src["url"], "file": out.name, "bytes": out.stat().st_size, "sha256": sha256(out),
            "license": src["license"], "downloaded_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"[done] {k}: {out.stat().st_size/1e6:.0f} MB sha256={manifest[k]['sha256'][:12]}")
    total = sum(m["bytes"] for m in manifest.values()) / 1e9
    print(f"manifest: {len(manifest)} files, {total:.2f} GB total in {dest}")
    if total > 5:
        print("WARNING: total exceeds the 5 GB gate", file=sys.stderr)


if __name__ == "__main__":
    main()
