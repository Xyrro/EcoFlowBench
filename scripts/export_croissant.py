#!/usr/bin/env python
"""Croissant 1.0 metadata for the AmpScape HF layout (NeurIPS D&B requirement; core + RAI fields).

Usage: python scripts/export_croissant.py --layout data/hf/AmpScape --out data/hf/AmpScape/croissant.json [--validate]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib

import pandas as pd

CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs", "column": "cr:column", "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/", "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"}, "dct": "http://purl.org/dc/terms/", "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject", "fileSet": "cr:fileSet",
    "format": "cr:format", "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath", "key": "cr:key",
    "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path", "recordSet": "cr:recordSet", "references": "cr:references",
    "regex": "cr:regex", "repeated": "cr:repeated", "replace": "cr:replace", "sc": "https://schema.org/", "separator": "cr:separator",
    "source": "cr:source", "subField": "cr:subField", "transform": "cr:transform",
}
INDEX_FIELDS = {
    "sample_id": ("sc:Text", "UUID5 of (dataset, family, key); one landscape"), "config": ("sc:Text", "source configuration name"),
    "kind": ("sc:Text", "points | wall_to_wall | regions | advanced | omniscape"), "task_ids": ("sc:Text", "tasks solved by this configuration"),
    "family": ("sc:Text", "synthetic | real"), "tier": ("sc:Text", "S/M/L/XL/XXL"), "H": ("sc:Integer", "raster height"), "W": ("sc:Integer", "raster width"),
    "generator": ("sc:Text", "synthetic generator or 'real'"), "resistance_table_id": ("sc:Text", "resistance table (real)"),
    "tile_id": ("sc:Text", "real tile id"), "biome_num": ("sc:Float", "RESOLVE biome number"), "realm": ("sc:Text", "RESOLVE realm"),
    "contrast": ("sc:Float", "max/min resistance"), "K": ("sc:Integer", "number of focal nodes"), "seed": ("sc:Integer", "landscape seed"),
    "solver": ("sc:Text", "linear solver used"), "solve_time_s": ("sc:Float", "solver wall time (s)"), "maxrss_mb": ("sc:Float", "peak RSS (MB)"),
    "residual_rel": ("sc:Float", "Kirchhoff residual (exact graph, full precision)"), "qc_flags": ("sc:Text", "comma-separated QC flags"),
    "qc_pass": ("sc:Boolean", "passes all hard QC checks"), "qc_trainval": ("sc:Boolean", "usable for train/val"), "split": ("sc:Text", "train | val | test_id | test_ood | ood_region | excluded"),
    "test_ood_region": ("sc:Boolean", "held-out region"), "test_ood_scale": ("sc:Boolean", "held-out scale"), "test_ood_table": ("sc:Boolean", "held-out resistance table"),
    "test_ood_contrast": ("sc:Boolean", "held-out contrast"), "test_ood_synth2real": ("sc:Boolean", "synthetic-to-real evaluation subset"),
    "task_group": ("sc:Text", "T1 | T1W | T1R | T3 | T4"), "hf_path": ("sc:Text", "shard path in this repository"),
    "subset_mini": ("sc:Boolean", "in the mini subset"), "subset_core": ("sc:Boolean", "in the core subset"), "subset_full": ("sc:Boolean", "in the full subset"),
}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build(layout: pathlib.Path, repo_id: str, version: str) -> dict:
    tiers = sorted(p.name for p in (layout / "data").iterdir() if p.is_dir())
    distribution = []
    for tier in tiers:
        for grp in sorted(p.name for p in (layout / "data" / tier).iterdir() if p.is_dir()):
            distribution.append({"@type": "cr:FileSet", "@id": f"shards-{tier}-{grp}", "name": f"shards-{tier}-{grp}",
                                 "description": f"HDF5 shards, tier {tier}, task group {grp} (inputs + solver outputs; schema in docs/schema.md)",
                                 "encodingFormat": "application/x-hdf5", "includes": f"data/{tier}/{grp}/*.h5"})
    for f in sorted((layout / "index").glob("*.parquet")):
        distribution.append({"@type": "cr:FileObject", "@id": f"index-{f.stem}", "name": f"index-{f.stem}", "contentUrl": f"index/{f.name}",
                             "encodingFormat": "application/x-parquet", "sha256": sha256(f)})
    record_sets = []
    for f in sorted((layout / "index").glob("*.parquet")):
        cols = set(pd.read_parquet(f).columns)
        fields = [{"@type": "cr:Field", "@id": f"records-{f.stem}/{c}", "name": f"records-{f.stem}/{c}", "description": d, "dataType": t,
                   "source": {"fileObject": {"@id": f"index-{f.stem}"}, "extract": {"column": c}}}
                  for c, (t, d) in INDEX_FIELDS.items() if c in cols]
        record_sets.append({"@type": "cr:RecordSet", "@id": f"records-{f.stem}", "name": f"records-{f.stem}", "key": {"@id": f"records-{f.stem}/sample_id"},
                            "description": f"One row per (sample, configuration) of tier {f.stem}; shard membership via hf_path.", "field": fields})
    return {
        "@context": CONTEXT, "@type": "sc:Dataset", "name": "AmpScape", "conformsTo": "http://mlcommons.org/croissant/1.0",
        "description": ("Benchmark of circuit-theoretic landscape connectivity (Circuitscape.jl 5.17.1 / Omniscape.jl 0.6.2 as exact reference "
                        "solvers) for learned surrogates: resistance rasters + source configurations -> current-density maps, voltage maps, "
                        "effective resistances, omnidirectional connectivity. Synthetic and real landscapes; official spatial splits and OOD sets."),
        "url": f"https://huggingface.co/datasets/{repo_id}", "license": "https://creativecommons.org/licenses/by/4.0/", "version": version,
        "datePublished": dt.date.today().isoformat(), "creator": {"@type": "sc:Organization", "name": "AmpScape contributors"},
        "citeAs": "AmpScape: a benchmark for learned surrogates of circuit-theoretic landscape connectivity (in preparation). See CITATION.cff.",
        "keywords": ["landscape connectivity", "Circuitscape", "Omniscape", "surrogate models", "neural operators", "ecology"],
        "isLiveDataset": False,
        # RAI fields (Croissant RAI vocabulary)
        "rai:dataCollection": ("Synthetic landscapes from neutral landscape models and Gaussian random fields with documented priors; real "
                               "landscapes are covariate stacks from ESA WorldCover 2021, Copernicus DEM GLO-30, GRIP4, HydroRIVERS and gHM, "
                               "sampled stratified by biome x realm x human-modification tercile; resistance from literature-informed tables "
                               "with AmpScape's own numeric values; outputs computed by the reference solvers and QC-checked."),
        "rai:dataCollectionType": ["Synthetic data", "Derived data"],
        "rai:dataPreprocessingProtocol": ["Resampling per source layer (majority/mean/min) to the tile grid", "Resistance = table(covariates) clipped to [1, r_max]",
                                          "No normalisation or clipping of solver outputs"],
        "rai:dataUseCases": ["Training and benchmarking surrogate models of connectivity solvers", "Physics-informed and operator learning research",
                             "Not a source of ecological truth for any species (resistance values are illustrative)"],
        "rai:dataLimitations": ["Resistance tables are not species-calibrated", "Omniscape block size is a documented approximation of the per-pixel solve",
                                "Mini release covers tier S only; real tiles in the mini over-represent held-out regions"],
        "rai:dataBiases": ["Real tiles are sampled to balance biome x realm x human modification, not population or area"],
        "rai:personalSensitiveInformation": "None; all inputs are public land-cover/terrain products at >= 100 m resolution; no protected-area boundaries are stored.",
        "rai:dataReleaseMaintenancePlan": "Versioned releases on the Hugging Face Hub; issues via the code repository; Zenodo DOI at the full release.",
        "distribution": distribution, "recordSet": record_sets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repo", default="Xirro/AmpScape")
    ap.add_argument("--version", default="0.2.0-mini")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    layout = pathlib.Path(a.layout)
    meta = build(layout, a.repo, a.version)
    out = pathlib.Path(a.out or layout / "croissant.json")
    out.write_text(json.dumps(meta, indent=1))
    print("wrote", out, f"({len(meta['distribution'])} distributions, {len(meta['recordSet'])} record sets)")
    if a.validate:
        try:
            import mlcroissant as mlc

            ds = mlc.Dataset(jsonld=str(out))
            issues = ds.metadata.issues
            print("mlcroissant validation:", "OK" if not issues.errors else f"{len(issues.errors)} errors", "|", len(issues.warnings), "warnings")
            for e in list(issues.errors)[:10]:
                print("  ERROR", e)
            for w in list(issues.warnings)[:5]:
                print("  warn", w)
        except ImportError:
            print("mlcroissant not installed; structural checks only")
        assert meta["conformsTo"].endswith("/1.0") and meta["distribution"] and meta["recordSet"]


if __name__ == "__main__":
    main()
