"""Sample manifest for a dataset build (Phase 5).

A *sample* is one landscape (synthetic seed or real tile × table) plus the set of source
configurations solved on it. The manifest is a Parquet table with one row per sample and is the
single source of truth for shards: ``shard`` (int) groups samples into one Julia job each.

Sample ids are UUID5 of ``(dataset_id, family, key)`` so re-planning the same dataset gives the
same ids.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TIER_SIZES = {"S": 128, "M": 256, "L": 512, "XL": 1024, "XXL": 2048}
TIER_PIXEL_M = {"S": 100.0, "M": 100.0, "L": 200.0, "XL": 500.0, "XXL": 1000.0}
NS = uuid.UUID("6d2a5a4e-6c3b-4f2e-9a7e-1e0c3b3a5e11")
DEFAULT_CONFIGS = ["points", "wall_to_wall_NS", "wall_to_wall_EW", "advanced", "omniscape"]


@dataclass
class SampleSpec:
    sample_id: str
    dataset_id: str
    family: str                 # synthetic | real
    tier: str
    size: int
    pixel_m: float
    seed: int                   # landscape seed (synthetic) / source seed (both)
    configs: str                # JSON list of config names to solve
    tile_id: str | None = None
    table_id: str | None = None
    generator: str | None = None
    contrast: float | None = None
    shard: int = 0
    extra: str = "{}"           # JSON, free-form (e.g. K override for probes)

    def config_list(self) -> list[str]:
        return json.loads(self.configs)


def sample_uuid(dataset_id: str, family: str, key: str) -> str:
    return str(uuid.uuid5(NS, f"{dataset_id}|{family}|{key}"))


def plan_synthetic(dataset_id: str, n: int, tier: str, seed0: int, configs=DEFAULT_CONFIGS,
                   shard_size: int = 50, shard0: int = 0) -> list[SampleSpec]:
    """n synthetic landscapes from the documented prior (seeds seed0..seed0+n-1)."""
    from ampscape.landscapes.synthetic import sample_landscape

    out = []
    for i in range(n):
        seed = seed0 + i
        ls = sample_landscape(seed, (TIER_SIZES[tier],) * 2)
        out.append(SampleSpec(sample_uuid(dataset_id, "synthetic", f"{tier}:{seed}"), dataset_id, "synthetic", tier,
                              TIER_SIZES[tier], TIER_PIXEL_M[tier], seed, json.dumps(list(configs)),
                              generator=ls.generator, contrast=ls.contrast, shard=shard0 + i // shard_size))
    return out


def plan_real(dataset_id: str, resistance_parquet: str, sources_parquet: str, n: int, tier: str, seed: int,
              configs=DEFAULT_CONFIGS, shard_size: int = 50, shard0: int = 0, one_per_tile: bool = True) -> list[SampleSpec]:
    """n real (tile, table) samples balanced over tables; regions config added where available."""
    res = pd.read_parquet(resistance_parquet)
    src = pd.read_parquet(sources_parquet)
    has_regions = set(zip(src[src.kind == "regions"].tile_id, src[src.kind == "regions"].table_id, strict=True))
    rng = np.random.default_rng(seed)
    tables = sorted(res.table_id.unique())
    per_table = int(np.ceil(n / len(tables)))
    used_tiles: set[str] = set()
    chosen = []
    for t in tables:
        cand = res[res.table_id == t].tile_id.tolist()
        rng.shuffle(cand)
        k = 0
        for tile in cand:
            if one_per_tile and tile in used_tiles:
                continue
            chosen.append((tile, t))
            used_tiles.add(tile)
            k += 1
            if k >= per_table:
                break
    chosen = chosen[:n]
    out = []
    for i, (tile, table) in enumerate(chosen):
        cfgs = list(configs) + (["regions"] if (tile, table) in has_regions and "regions" not in configs else [])
        out.append(SampleSpec(sample_uuid(dataset_id, "real", f"{tier}:{tile}:{table}"), dataset_id, "real", tier,
                              TIER_SIZES[tier], TIER_PIXEL_M[tier], seed * 1000 + i, json.dumps(cfgs), tile_id=tile,
                              table_id=table, shard=shard0 + i // shard_size))
    return out


def to_frame(specs: list[SampleSpec]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in specs])


def from_frame(df: pd.DataFrame) -> list[SampleSpec]:
    return [SampleSpec(**{k: (None if pd.isna(v) else v) for k, v in r.items()}) for r in df.to_dict("records")]
