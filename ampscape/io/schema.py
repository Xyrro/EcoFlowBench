"""AmpScape HDF5 shard schema (v0.2) and validator (brief §8.1, docs/schema.md).

Shard layout (one file per shard, one group per sample)::

    /                              attrs: dataset_version, schema_version, pipeline_git_sha, created_at
    /<sample_id>/                  attrs: meta (JSON, see MetaModel)
        inputs/resistance          float32 (H, W)   R in [1, r_max], 1.0 at NoData
        inputs/nodata_mask         uint8   (H, W)   1 = NoData
        inputs/covariates          float32 (C, H, W)  real tiles only; attrs.channels
        configs/<cname>/           attrs: kind, task_ids, focal_table (JSON), source_meta (JSON)
            inputs/focal_mask      int32   (H, W)   points / wall_to_wall / regions
            inputs/source_strength float32 (H, W)   advanced / omniscape
            inputs/ground          int8    (H, W)   advanced
            outputs/...            solver outputs, raw (see CONFIG_OUTPUTS); attrs: solver_stats, qc_flags, qc_pass

The validator is used by `scripts/generate.py finalize` (every shard) and by the tests. It checks
structure, dtypes, shapes, value ranges, metadata fields and consistency between inputs and outputs.
"""

from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np
from pydantic import BaseModel, Field, ValidationError

SCHEMA_VERSION = "0.2"
KINDS = {"points", "wall_to_wall", "regions", "advanced", "omniscape"}
KIND_TASK = {"points": "T1,T2", "wall_to_wall": "T1W", "regions": "T1R", "advanced": "T3", "omniscape": "T4"}
CONFIG_INPUTS = {
    "points": {"focal_mask": np.int32}, "wall_to_wall": {"focal_mask": np.int32}, "regions": {"focal_mask": np.int32},
    "advanced": {"source_strength": np.float32, "ground": np.int8}, "omniscape": {"source_strength": np.float32},
}
CONFIG_OUTPUTS = {
    "points": {"cum_current": np.float32, "reff": np.float64, "labels": np.int32, "pair_index": np.int32},
    "wall_to_wall": {"cum_current": np.float32, "reff": np.float64, "labels": np.int32, "pair_index": np.int32},
    "regions": {"cum_current": np.float32, "reff": np.float64, "labels": np.int32, "pair_index": np.int32},
    "advanced": {"current": np.float32, "voltage": np.float32},
    "omniscape": {"cum_current": np.float32, "flow_potential": np.float32, "normalized": np.float32},
}
OPTIONAL_OUTPUTS = {"pairwise_current": np.float32, "voltage": np.float32}   # pairwise kinds, K <= 4


class MetaModel(BaseModel):
    """Required per-sample metadata (docs/task_specification.md §4.3)."""

    sample_id: str
    dataset_id: str
    family: str
    tier: str
    H: int
    W: int
    pixel_size_m: float
    seed: int
    generator: str | None = None
    generator_params: dict | None = None
    contrast: float | None = None
    resistance_table_id: str | None = None
    tile_id: str | None = None
    lat: float | None = None
    lon: float | None = None
    crs: str | None = None
    graph_connectivity: int = 8
    source_config: dict
    omniscape: dict
    solver_name: str
    solver_versions: dict
    solver_preset: dict
    qc_flags: list[str] = Field(default_factory=list)
    created_at: str
    pipeline_git_sha: str
    dataset_version: str
    resampling: dict | None = None      # real tiles (owner amendment C2); required from schema 0.3


class ValidationReport(BaseModel):
    path: str
    n_samples: int = 0
    n_configs: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_ds(g: h5py.Group, name: str, dtype, shape: tuple | None, errs: list[str], where: str) -> h5py.Dataset | None:
    if name not in g:
        errs.append(f"{where}: missing dataset {name}")
        return None
    d = g[name]
    if d.dtype != np.dtype(dtype):
        errs.append(f"{where}/{name}: dtype {d.dtype} != {np.dtype(dtype)}")
    if shape is not None and tuple(d.shape) != tuple(shape):
        errs.append(f"{where}/{name}: shape {d.shape} != {shape}")
    return d


def validate_sample(gs: h5py.Group, errs: list[str], warns: list[str], max_samples_arrays: bool = True) -> int:
    sid = gs.name.strip("/")
    try:
        meta = MetaModel.model_validate(json.loads(gs.attrs["meta"]))
    except (KeyError, ValidationError, json.JSONDecodeError) as e:
        errs.append(f"{sid}: meta invalid: {str(e)[:200]}")
        return 0
    if meta.sample_id != sid:
        errs.append(f"{sid}: meta.sample_id mismatch")
    H, W = meta.H, meta.W
    gi = gs.get("inputs")
    if gi is None:
        errs.append(f"{sid}: missing inputs")
        return 0
    R = _check_ds(gi, "resistance", np.float32, (H, W), errs, sid)
    M = _check_ds(gi, "nodata_mask", np.uint8, (H, W), errs, sid)
    if R is not None and M is not None:
        r = R[...]
        m = M[...] > 0
        if not np.isfinite(r).all():
            errs.append(f"{sid}: non-finite resistance")
        if r[~m].size and (r[~m].min() < 1.0):
            errs.append(f"{sid}: resistance < 1 on valid pixels")
        if not np.all(r[m] == 1.0):
            errs.append(f"{sid}: resistance at NoData must be 1.0")
        if meta.family == "real" and "covariates" not in gi:
            errs.append(f"{sid}: real sample without covariates")
    n_cfg = 0
    for cname, gc in gs.get("configs", {}).items():
        kind = gc.attrs.get("kind")
        if kind not in KINDS:
            errs.append(f"{sid}/{cname}: unknown kind {kind}")
            continue
        if gc.attrs.get("task_ids") != KIND_TASK[kind]:
            errs.append(f"{sid}/{cname}: task_ids {gc.attrs.get('task_ids')} != {KIND_TASK[kind]}")
        for name, dt in CONFIG_INPUTS[kind].items():
            _check_ds(gc["inputs"], name, dt, (H, W), errs, f"{sid}/{cname}")
        go = gc.get("outputs")
        if go is None:
            errs.append(f"{sid}/{cname}: missing outputs")
            continue
        for name, dt in CONFIG_OUTPUTS[kind].items():
            shape = (H, W) if name in ("cum_current", "current", "voltage", "flow_potential", "normalized") else None
            _check_ds(go, name, dt, shape, errs, f"{sid}/{cname}")
        for name, dt in OPTIONAL_OUTPUTS.items():
            if kind in ("points", "wall_to_wall", "regions") and name in go:
                d = go[name]
                if d.dtype != np.dtype(dt) or d.ndim != 3 or d.shape[1:] != (H, W):
                    errs.append(f"{sid}/{cname}/{name}: bad dtype/shape")
        for a in ("solver_stats", "qc_flags", "qc_pass"):
            if a not in go.attrs:
                errs.append(f"{sid}/{cname}: missing attr {a}")
        if kind in ("points", "wall_to_wall", "regions") and "reff" in go and "focal_mask" in gc["inputs"]:
            K = len(np.unique(gc["inputs"]["focal_mask"][...])) - 1
            if go["reff"].shape != (K, K):
                errs.append(f"{sid}/{cname}: reff shape {go['reff'].shape} != ({K}, {K})")
            if K > 4 and "pairwise_current" in go:
                warns.append(f"{sid}/{cname}: pairwise maps stored for K={K} > 4")
        for name in CONFIG_OUTPUTS[kind]:
            if name in go and go[name].dtype.kind == "f" and not np.isfinite(go[name][...]).all():
                errs.append(f"{sid}/{cname}/{name}: non-finite values")
        n_cfg += 1
    if n_cfg == 0:
        errs.append(f"{sid}: no configurations")
    return n_cfg


def validate_shard(path: str, max_samples: int | None = None) -> ValidationReport:
    rep = ValidationReport(path=str(path))
    with h5py.File(path, "r") as f:
        for a in ("dataset_version", "schema_version", "pipeline_git_sha", "created_at"):
            if a not in f.attrs:
                rep.errors.append(f"root: missing attr {a}")
        if f.attrs.get("schema_version") not in (SCHEMA_VERSION, "0.1"):
            rep.warnings.append(f"root: schema_version {f.attrs.get('schema_version')} (validator {SCHEMA_VERSION})")
        for i, sid in enumerate(f):
            if max_samples and i >= max_samples:
                break
            rep.n_samples += 1
            rep.n_configs += validate_sample(f[sid], rep.errors, rep.warnings)
    return rep


def schema_markdown() -> str:
    """Render docs/schema.md from the definitions above."""
    lines = ["# AmpScape shard schema", "", f"Schema version **{SCHEMA_VERSION}** (`ampscape.io.schema`). One HDF5 file per shard, one group per sample.", "",
             "## Root attributes", "", "`dataset_version`, `schema_version`, `pipeline_git_sha`, `created_at`.", "",
             "## Sample group `/<sample_id>/`", "", "Attribute `meta` (JSON) with the fields of `MetaModel`:", "",
             "| field | type | note |", "|---|---|---|"]
    for name, fld in MetaModel.model_fields.items():
        lines.append(f"| `{name}` | `{fld.annotation}` | {'required' if fld.is_required() else 'optional'} |")
    lines += ["", "### `inputs/`", "", "| dataset | dtype | shape |", "|---|---|---|",
              "| `resistance` | float32 | (H, W) — R ∈ [1, r_max], 1.0 at NoData |",
              "| `nodata_mask` | uint8 | (H, W) — 1 = NoData |",
              "| `covariates` | float32 | (C, H, W) — real tiles; `attrs.channels` |", ""]
    lines += ["### `configs/<name>/`", "", "Attributes `kind`, `task_ids`, `focal_table` (JSON), `source_meta` (JSON).", "",
              "| kind | tasks | inputs | outputs (raw solver output) |", "|---|---|---|---|"]
    for k in ("points", "wall_to_wall", "regions", "advanced", "omniscape"):
        ins = ", ".join(f"`{n}` {np.dtype(d).name}" for n, d in CONFIG_INPUTS[k].items())
        outs = ", ".join(f"`{n}` {np.dtype(d).name}" for n, d in CONFIG_OUTPUTS[k].items())
        if k in ("points", "wall_to_wall", "regions"):
            outs += "; K ≤ 4 also `pairwise_current`, `voltage` float32 (P, H, W)"
        lines.append(f"| {k} | {KIND_TASK[k]} | {ins} | {outs} |")
    lines += ["", "Output group attributes: `solver_stats` (JSON `SolveStats`), `qc_flags` (JSON list), `qc_pass` (bool).", "",
              "Conventions: north-up row-major rasters; pair (i, j): node i grounded, 1 A injected at j; NoData pixels",
              "hold 0 in output maps; nothing is normalised or clipped. See `docs/task_specification.md`.", "",
              "## Parquet index", "", "One row per (sample, config): identifiers, family/tier/generator/table/tile, K, placement, seed,",
              "solver, timings, residuals, `qc_flags`, `qc_pass`, `qc_trainval`, split and OOD flags, shard file.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import pathlib

    ap = argparse.ArgumentParser(description="Validate AmpScape HDF5 shards")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--write-schema-md", default=None)
    a = ap.parse_args(argv)
    if a.write_schema_md:
        pathlib.Path(a.write_schema_md).write_text(schema_markdown())
        print("wrote", a.write_schema_md)
    rc = 0
    for p in a.paths:
        rep = validate_shard(p)
        print(f"{p}: {rep.n_samples} samples, {rep.n_configs} configs, {len(rep.errors)} errors, {len(rep.warnings)} warnings")
        for e in rep.errors[:20]:
            print("  ERROR", e)
        rc |= int(not rep.ok)
    return rc


def as_dict(rep: ValidationReport) -> dict[str, Any]:
    return rep.model_dump()


if __name__ == "__main__":
    raise SystemExit(main())
