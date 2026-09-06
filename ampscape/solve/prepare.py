"""Materialise the inputs of one shard into an HDF5 file the Julia batch solver understands.

Layout (see AmpScapeSolve.solve_shard): /samples/<sid>/inputs/{resistance, nodata_mask,
covariates?, focal tables...}, /samples/<sid>/configs/<cname>/{focal_mask | source_strength [+ ground]}
with attrs ``kind``; per-sample attrs ``meta`` (JSON) carrying everything needed for the final
metadata (generator params, table provenance, source config provenance, focal tables).
"""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import yaml

from ampscape.landscapes.real import CHANNELS
from ampscape.solve.manifest import SampleSpec
from ampscape.sources import SourceConfig, generate_all

OMNI_TIERS = {"S": (16, 3), "M": (32, 5), "L": (64, 9), "XL": (128, 17), "XXL": (256, 33)}


def omni_params(tier: str, solver_yaml: str = "configs/solver/omniscape_reference.yaml") -> tuple[int, int]:
    try:
        d = yaml.safe_load(open(solver_yaml))["tiers"][tier]
        return int(d["radius"]), int(d["block_size"])
    except Exception:  # noqa: BLE001
        return OMNI_TIERS[tier]


def load_landscape(spec: SampleSpec, pilot_root: str | None):
    """Return (R float32, nodata bool, landcover int16|None, covariates dict|None, meta dict)."""
    if spec.family == "synthetic":
        from ampscape.landscapes.synthetic import sample_landscape

        ls = sample_landscape(spec.seed, (spec.size, spec.size))
        meta = {"generator": ls.generator, "generator_params": ls.params, "contrast": ls.contrast,
                "resistance_table_id": None}
        return ls.resistance, ls.nodata_mask, None, None, meta
    import pandas as pd
    import rasterio

    from ampscape.landscapes.real import read_tile

    root = pathlib.Path(pilot_root)
    res = pd.read_parquet(root / "resistance.parquet")
    row = res[(res.tile_id == spec.tile_id) & (res.table_id == spec.table_id)].iloc[0]
    with rasterio.open(root / row.path) as src:
        R = src.read(1).astype(np.float32)
        nd = src.read(2) > 0.5
        r_max = float(src.tags().get("r_max", "nan"))
    tiles = pd.read_parquet(root / "tiles.parquet").set_index("tile_id")
    t = tiles.loc[spec.tile_id]
    cov, _ = read_tile(str(root / t["path"]))
    meta = {"generator": "real", "resistance_table_id": spec.table_id, "table_version": int(row.table_version),
            "table_sha256": row.table_sha256, "r_max": r_max, "tile_id": spec.tile_id, "lat": float(t["lat"]),
            "lon": float(t["lon"]), "crs": f"EPSG:{int(t['epsg'])}", "transform": json.loads(t["transform"]),
            "biome_num": int(t["biome_num"]), "biome_name": t["biome_name"], "realm": t["realm"],
            "ghm_tercile": int(t["ghm_tercile"]), "stratum": t["stratum"],
            "frac_at_rmax": float(row.frac_at_rmax), "contrast": float(10 ** row.log10_contrast)}
    return R, nd, cov["landcover"], cov, meta


def prepare_shard(specs: list[SampleSpec], out_h5: str, source_cfg: SourceConfig, pilot_root: str | None = None,
                  overwrite: bool = False) -> int:
    """Write inputs for all specs into out_h5; returns number of samples written."""
    p = pathlib.Path(out_h5)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        with h5py.File(p, "r") as f:
            if all(s.sample_id in f["samples"] for s in specs):
                return 0
    n = 0
    with h5py.File(p, "w") as f:
        g_all = f.create_group("samples")
        for spec in specs:
            R, nd, lc, cov, meta = load_landscape(spec, pilot_root)
            cfg = source_cfg.for_tier(spec.tier)
            extra = json.loads(spec.extra or "{}")
            samples = generate_all(R, nd, cfg, spec.seed, landcover=lc)
            wanted = spec.config_list()
            if "k_override" in extra:
                from ampscape.sources import sample_points

                samples["points"] = sample_points(R, nd, cfg, np.random.default_rng(spec.seed), k=int(extra["k_override"]))
            g = g_all.create_group(spec.sample_id)
            radius, bs = omni_params(spec.tier)
            g.attrs["tier"] = spec.tier
            g.attrs["omni_radius"] = radius
            g.attrs["omni_block_size"] = bs
            gi = g.create_group("inputs")
            gi.create_dataset("resistance", data=R.astype(np.float32), compression="gzip", compression_opts=4)
            gi.create_dataset("nodata_mask", data=nd.astype(np.uint8), compression="gzip", compression_opts=4)
            if cov is not None:
                stack = np.stack([np.nan_to_num(cov[c].astype(np.float32), nan=-9999.0) for c in CHANNELS])
                d = gi.create_dataset("covariates", data=stack, compression="gzip", compression_opts=4)
                d.attrs["channels"] = ",".join(CHANNELS)
            gc_all = g.create_group("configs")
            src_meta = {}
            for cname in wanted:
                s = samples.get(cname)
                if s is None:
                    continue
                gc = gc_all.create_group(cname)
                gc.attrs["kind"] = s.kind
                if s.focal_mask is not None:
                    gc.create_dataset("focal_mask", data=s.focal_mask.astype(np.int32), compression="gzip", compression_opts=4)
                if s.source_strength is not None:
                    gc.create_dataset("source_strength", data=s.source_strength.astype(np.float32), compression="gzip", compression_opts=4)
                if s.ground is not None:
                    gc.create_dataset("ground", data=s.ground.astype(np.int8), compression="gzip", compression_opts=4)
                if s.kind == "omniscape":
                    gc.attrs["source_threshold"] = float(s.meta.get("source_threshold", 0.0))
                src_meta[cname] = {"focal_table": s.focal_table, "meta": s.meta}
            meta.update({"sample_id": spec.sample_id, "dataset_id": spec.dataset_id, "family": spec.family,
                         "tier": spec.tier, "H": spec.size, "W": spec.size, "pixel_size_m": spec.pixel_m,
                         "seed": spec.seed, "source_config": cfg.provenance(), "omniscape": {"radius": radius, "block_size": bs},
                         "configs": src_meta, "graph_connectivity": 8})
            g.attrs["meta"] = json.dumps(meta, default=float)
            n += 1
    return n
