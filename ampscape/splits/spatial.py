"""Spatial block splits shared across tiers (dataset plan §5, owner amendment C1).

The globe is partitioned into ``size_deg`` × ``size_deg`` lon/lat macro-cells (default 20°, larger
than an XXL tile so that every tile of every tier can lie inside one cell). Each cell receives exactly
one assignment — ``train``, ``val``, ``test_id`` or ``ood_region`` — from the dataset seed, and the
assignment is applied at every tier. Tiles whose footprint straddles cells with different assignments
are **excluded** (the tile sampler resamples them, ``split = "excluded"``); as a safety net the
footprint rule (test-most assignment wins) plus fix-point propagation is still applied to whatever
remains, so a test region at any resolution never overlaps a training region at any other resolution.

Order of precedence (test-most first): ood_region > test_id > val > train.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

RANK = {"ood_region": 3, "test_id": 2, "val": 1, "train": 0}
ASSIGNMENTS = ["train", "val", "test_id"]


@dataclass(frozen=True)
class BlockGrid:
    size_deg: float = 20.0

    def block_id(self, lat: float, lon: float) -> str:
        lon = ((lon + 180.0) % 360.0) - 180.0
        i = int(math.floor((lat + 90.0) / self.size_deg))
        j = int(math.floor((lon + 180.0) / self.size_deg))
        return f"b{i:03d}_{j:03d}"

    def footprint_blocks(self, lat: float, lon: float, half_extent_m: float) -> list[str]:
        """All blocks touched by a square tile of half-extent ``half_extent_m`` centred at (lat, lon)."""
        dlat = half_extent_m / 111_320.0
        dlon = half_extent_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.05))
        i0 = int(math.floor((max(lat - dlat, -90.0) + 90.0) / self.size_deg))
        i1 = int(math.floor((min(lat + dlat, 89.999) + 90.0) / self.size_deg))
        j0 = int(math.floor((lon - dlon + 180.0) / self.size_deg))
        j1 = int(math.floor((lon + dlon + 180.0) / self.size_deg))
        n_lon = int(round(360.0 / self.size_deg))
        ids = {f"b{i:03d}_{j % n_lon:03d}" for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}
        return sorted(ids)


def assign_blocks(block_ids: list[str], seed: int, fractions: dict[str, float] | None = None,
                  strata: dict[str, str] | None = None) -> dict[str, str]:
    """One seeded assignment per block, stratified (optionally) by ``strata[block_id]`` (e.g. realm).

    Within each stratum blocks are shuffled deterministically and cut by the fractions in order
    train / val / test_id, so every stratum contributes to every split.
    """
    fractions = fractions or {"train": 0.8, "val": 0.1, "test_id": 0.1}
    out: dict[str, str] = {}
    groups: dict[str, list[str]] = {}
    for b in sorted(set(block_ids)):
        groups.setdefault(strata.get(b, "all") if strata else "all", []).append(b)
    for stratum, blocks in groups.items():
        keyed = sorted(blocks, key=lambda b: hashlib.sha256(f"{seed}|{stratum}|{b}".encode()).hexdigest())
        n = len(keyed)
        n_val = int(round(n * fractions["val"]))
        n_test = int(round(n * fractions["test_id"]))
        n_train = n - n_val - n_test
        for k, b in enumerate(keyed):
            out[b] = "train" if k < n_train else ("val" if k < n_train + n_val else "test_id")
    return out


def tile_split(assign: dict[str, str], blocks: list[str], ood_blocks: set[str] | None = None) -> str:
    """Footprint rule: the test-most assignment among the blocks a tile touches."""
    best = "train"
    for b in blocks:
        a = "ood_region" if ood_blocks and b in ood_blocks else assign.get(b, "train")
        if RANK[a] > RANK[best]:
            best = a
    return best


def assign_tiles(tiles: pd.DataFrame, grid: BlockGrid, seed: int, ood_blocks: set[str] | None = None,
                 strata_col: str | None = "realm", fractions: dict[str, float] | None = None,
                 exclude_straddling: bool = True) -> pd.DataFrame:
    """Add ``block_id``, ``footprint_blocks`` and ``split`` columns to a tile table.

    ``tiles`` needs ``tile_id, lat, lon, size, pixel_m`` (+ ``realm`` when stratifying). Blocks are
    assigned once from all tiles of all tiers, then applied per tile with the footprint rule.
    """
    t = tiles.copy()
    t["block_id"] = [grid.block_id(a, b) for a, b in zip(t.lat, t.lon, strict=True)]
    t["footprint_blocks"] = [grid.footprint_blocks(a, b, s * p / 2.0) for a, b, s, p in
                             zip(t.lat, t.lon, t["size"], t.pixel_m, strict=True)]
    strata = None
    if strata_col and strata_col in t:
        strata = t.groupby("block_id")[strata_col].agg(lambda x: x.mode().iloc[0]).to_dict()
    all_blocks = sorted({b for fb in t.footprint_blocks for b in fb})
    assign = assign_blocks(all_blocks, seed, fractions, strata)

    def kind(b: str) -> str:
        return "ood_region" if ood_blocks and b in ood_blocks else assign.get(b, "train")

    t["straddles"] = [len({kind(b) for b in fb}) > 1 for fb in t.footprint_blocks]
    if exclude_straddling:
        keep = ~t.straddles
        t_keep = t[keep].copy()
    else:
        t_keep = t
    # Fix-point propagation of the footprint rule: a block touched by any test/OOD tile becomes a
    # test block (test-most assignment), then tile splits are re-derived, until nothing changes.
    # This guarantees that no train/val tile at any tier shares a block with a test tile at any tier.
    for _ in range(20):
        splits = [tile_split(assign, fb, ood_blocks) for fb in t_keep.footprint_blocks]
        changed = False
        for fb, sp in zip(t_keep.footprint_blocks, splits, strict=True):
            if RANK[sp] >= RANK["test_id"]:
                for b in fb:
                    if RANK[assign.get(b, "train")] < RANK[sp] and not (ood_blocks and b in ood_blocks):
                        assign[b] = sp if sp != "ood_region" else "test_id"
                        changed = True
        if not changed:
            break
    t["split"] = [("excluded" if (exclude_straddling and st) else tile_split(assign, fb, ood_blocks))
                  for fb, st in zip(t.footprint_blocks, t.straddles, strict=True)]
    t.attrs["block_assignment"] = assign
    return t


def check_no_cross_tier_overlap(t: pd.DataFrame) -> list[tuple[str, str]]:
    """Return pairs (train/val tile, test/OOD tile) whose footprints share a block, across all tiers."""
    trainval = t[t.split.isin(["train", "val"])]
    test = t[~t.split.isin(["train", "val", "excluded"])]
    by_block: dict[str, list[str]] = {}
    for r in test.itertuples():
        for b in r.footprint_blocks:
            by_block.setdefault(b, []).append(r.tile_id)
    bad = []
    for r in trainval.itertuples():
        for b in r.footprint_blocks:
            for tid in by_block.get(b, []):
                bad.append((r.tile_id, tid))
    return bad


def synthetic_split(seed_family: int, dataset_seed: int, fractions: dict[str, float] | None = None) -> str:
    """Deterministic split for a synthetic seed family."""
    fractions = fractions or {"train": 0.8, "val": 0.1, "test_id": 0.1}
    h = int(hashlib.sha256(f"{dataset_seed}|synthetic|{seed_family}".encode()).hexdigest()[:8], 16) / 2**32
    if h < fractions["train"]:
        return "train"
    if h < fractions["train"] + fractions["val"]:
        return "val"
    return "test_id"


def ood_flags(row: dict, cfg: dict) -> dict[str, bool]:
    """OOD-set membership flags for one index row (dataset plan §4)."""
    o = cfg["ood"]
    fam = row.get("family")
    return {
        "test_ood_region": row.get("split") == "ood_region",
        "test_ood_scale": row.get("tier") in o["test_ood_scale"]["tiers"] and row.get("split") not in ("train", "val"),
        "test_ood_table": fam == "real" and row.get("resistance_table_id") == o["test_ood_table"]["table"],
        "test_ood_contrast": fam == "synthetic" and float(row.get("contrast") or 0) >= o["test_ood_contrast"]["contrast"],
        "test_ood_synth2real": fam == "real" and row.get("split") == "test_id",
    }


def apply_holdouts(split: str, flags: dict[str, bool]) -> str:
    """Held-out table / contrast landscapes are never train/val: demote them to test."""
    if split in ("train", "val") and (flags["test_ood_table"] or flags["test_ood_contrast"]):
        return "test_ood"
    return split


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    return t.groupby(["tier", "split"]).size().unstack("split", fill_value=0)


__all__ = ["BlockGrid", "assign_blocks", "assign_tiles", "check_no_cross_tier_overlap", "synthetic_split",
           "ood_flags", "apply_holdouts", "summarize", "np"]
