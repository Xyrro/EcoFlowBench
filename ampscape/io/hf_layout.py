"""Hugging Face repository layout (dataset plan §5.2, owner amendment C4).

    data/<tier>/<task_group>/shard-NNNNN.h5     per-task-group shards: inputs + ONE configuration
    index/<tier>.parquet                        one row per (sample, config) with split / OOD / subset columns
    splits/<subset>/<split>.parquet             sample ids, nested mini ⊂ core ⊂ full
    croissant.json, README.md (dataset card)

Task groups map 1:1 onto source configurations so that any single tier or task can be downloaded
alone: T1 = `points` (T1 + T2: cumulative current, per-pair maps for K ≤ 4, Reff), T1W = wall-to-wall
strips, T1R = habitat-patch regions, T3 = `advanced`, T4 = `omniscape`. Inputs (resistance, NoData
mask, covariates) are duplicated into every task-group shard (≈ 0.1 MB per S sample) so each shard is
self-contained.
"""

from __future__ import annotations

import pathlib

import h5py
import pandas as pd

TASK_GROUPS = {"T1": ["points"], "T1W": ["wall_to_wall_NS", "wall_to_wall_EW"], "T1R": ["regions"], "T3": ["advanced"], "T4": ["omniscape"]}
CONFIG_TO_GROUP = {c: g for g, cs in TASK_GROUPS.items() for c in cs}
GZIP = {"compression": "gzip", "compression_opts": 4, "shuffle": True}


def _copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            _copy_group(item, dst.create_group(name))
        else:
            d = dst.create_dataset(name, data=item[...], chunks=True if item.ndim >= 2 else None, **(GZIP if item.ndim >= 2 else {}))
            for k, v in item.attrs.items():
                d.attrs[k] = v


def split_shard_by_task_group(final_h5: str | pathlib.Path, out_root: str | pathlib.Path, tier: str) -> dict[str, pathlib.Path]:
    """Write one shard per task group under out_root/data/<tier>/<group>/<same shard name>."""
    final_h5 = pathlib.Path(final_h5)
    outs: dict[str, h5py.File] = {}
    paths: dict[str, pathlib.Path] = {}
    with h5py.File(final_h5, "r") as f:
        root_attrs = dict(f.attrs)
        for sid in f:
            gs = f[sid]
            for cname in gs["configs"]:
                grp = CONFIG_TO_GROUP.get(cname)
                if grp is None:
                    continue
                if grp not in outs:
                    p = pathlib.Path(out_root) / "data" / tier / grp / final_h5.name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    outs[grp] = h5py.File(p, "w")
                    for k, v in root_attrs.items():
                        outs[grp].attrs[k] = v
                    outs[grp].attrs["task_group"] = grp
                    paths[grp] = p
                fo = outs[grp]
                if sid not in fo:
                    g = fo.create_group(sid)
                    for k, v in gs.attrs.items():
                        g.attrs[k] = v
                    _copy_group(gs["inputs"], g.create_group("inputs"))
                    g.create_group("configs")
                _copy_group(gs["configs"][cname], fo[sid]["configs"].create_group(cname))
    for fo in outs.values():
        fo.close()
    return paths


def export_build(build: str | pathlib.Path, out_root: str | pathlib.Path, tier: str, subsets: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """Export a finalized build into the HF layout; returns the per-tier index with `shard` re-pointed."""
    build = pathlib.Path(build)
    out_root = pathlib.Path(out_root)
    idx = pd.read_parquet(build / "index.parquet")
    for sh in sorted((build / "shards").glob("shard-*.h5")):
        if not sh.with_suffix(".ok").exists():
            raise RuntimeError(f"{sh} has no .ok marker (run the validator first)")
        split_shard_by_task_group(sh, out_root, tier)
    idx["task_group"] = idx.config.map(CONFIG_TO_GROUP)
    idx["hf_path"] = [f"data/{tier}/{g}/{s}" for g, s in zip(idx.task_group, idx.shard, strict=True)]
    for name, ids in (subsets or {}).items():
        idx[f"subset_{name}"] = idx.sample_id.isin(ids)
    (out_root / "index").mkdir(parents=True, exist_ok=True)
    idx.to_parquet(out_root / "index" / f"{tier}.parquet", index=False)
    for sub in (subsets or {"full": set(idx.sample_id)}):
        d = out_root / "splits" / sub
        d.mkdir(parents=True, exist_ok=True)
        part = idx[idx[f"subset_{sub}"]] if f"subset_{sub}" in idx else idx
        for split, g in part.groupby("split"):
            g[["sample_id"]].drop_duplicates().to_parquet(d / f"{split}.parquet", index=False)
    return idx
