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
    """Macro-cell grid used as the unit of split assignment.

    ``equal_width=True`` (default): latitude bands of ``size_deg`` and, per band, ``n_lon`` longitude
    cells chosen so that the cell's ground width at the band centre ≈ ``size_deg`` × 111 km at every
    latitude (a reduced / equal-width grid, ≈ equal-area). With 30° bands a cell is ≈ 3 340 km wide at
    its centre and ≥ 2 360 km at the poleward edge of the 30–60° band, so an XXL tile (2 048 km) fits
    inside one cell at every latitude that has land tiles (|lat| ≤ 72°).
    ``equal_width=False``: plain lon/lat cells (the first design; XXL fits only within ±23° — rejected).
    """

    size_deg: float = 30.0
    equal_width: bool = True

    # ---- band / cell arithmetic ----
    def n_bands(self) -> int:
        return int(round(180.0 / self.size_deg))

    def band(self, lat: float) -> int:
        return min(max(int(math.floor((lat + 90.0) / self.size_deg)), 0), self.n_bands() - 1)

    def n_lon(self, band: int) -> int:
        if not self.equal_width:
            return int(round(360.0 / self.size_deg))
        lat_c = -90.0 + (band + 0.5) * self.size_deg
        return max(1, int(round(360.0 * math.cos(math.radians(lat_c)) / self.size_deg)))

    def lon_index(self, band: int, lon: float) -> int:
        n = self.n_lon(band)
        lon = ((lon + 180.0) % 360.0) - 180.0
        return int(math.floor((lon + 180.0) / (360.0 / n))) % n

    def block_id(self, lat: float, lon: float) -> str:
        b = self.band(lat)
        return f"b{b:03d}_{self.lon_index(b, lon):03d}"

    def cell_bounds(self, block_id: str) -> tuple[float, float, float, float]:
        """(lat_min, lat_max, lon_min, lon_max) of a cell."""
        b, j = int(block_id[1:4]), int(block_id[5:8])
        w = 360.0 / self.n_lon(b)
        return (-90.0 + b * self.size_deg, -90.0 + (b + 1) * self.size_deg, -180.0 + j * w, -180.0 + (j + 1) * w)

    def footprint_blocks(self, lat: float, lon: float, half_extent_m: float) -> list[str]:
        """All cells touched by a square tile of half-extent ``half_extent_m`` centred at (lat, lon)."""
        dlat = half_extent_m / 111_320.0
        lat0, lat1 = max(lat - dlat, -90.0), min(lat + dlat, 89.999)
        ids = set()
        for b in range(self.band(lat0), self.band(lat1) + 1):
            # widest longitude span within this band = at its most poleward latitude inside the tile
            blat0 = max(lat0, -90.0 + b * self.size_deg)
            blat1 = min(lat1, -90.0 + (b + 1) * self.size_deg)
            worst = max(abs(blat0), abs(blat1))
            dlon = half_extent_m / (111_320.0 * max(math.cos(math.radians(min(worst, 89.0))), 0.02))
            n = self.n_lon(b)
            w = 360.0 / n
            j0 = int(math.floor((lon - dlon + 180.0) / w))
            j1 = int(math.floor((lon + dlon + 180.0) / w))
            if j1 - j0 + 1 >= n:
                ids.update(f"b{b:03d}_{j:03d}" for j in range(n))
            else:
                ids.update(f"b{b:03d}_{j % n:03d}" for j in range(j0, j1 + 1))
        return sorted(ids)

    def fits(self, lat: float, lon: float, half_extent_m: float) -> bool:
        return len(self.footprint_blocks(lat, lon, half_extent_m)) == 1

    def interior_bounds(self, block_id: str, half_extent_m: float) -> tuple[float, float, float, float] | None:
        """Lat/lon box of tile centres whose tile fits inside the cell (for the sampler); None if none."""
        lat0, lat1, lon0, lon1 = self.cell_bounds(block_id)
        dlat = half_extent_m / 111_320.0
        worst = max(abs(lat0), abs(lat1))
        dlon = half_extent_m / (111_320.0 * max(math.cos(math.radians(min(worst, 89.0))), 0.02))
        box = (lat0 + dlat, lat1 - dlat, lon0 + dlon, lon1 - dlon)
        return box if box[0] < box[1] and box[2] < box[3] else None


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


def tile_box(lat: float, lon: float, half_extent_m: float) -> tuple[float, float, float, float]:
    """Conservative lat/lon bounding box (lat0, lat1, lon0, lon1) of a square tile."""
    dlat = half_extent_m / 111_320.0
    worst = min(max(abs(lat - dlat), abs(lat + dlat)), 89.0)
    dlon = half_extent_m / (111_320.0 * max(math.cos(math.radians(worst)), 0.02))
    return (lat - dlat, lat + dlat, lon - dlon, lon + dlon)


def _boxes_intersect(a, b) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _box_inside(inner, outer) -> bool:
    return inner[0] >= outer[0] and inner[1] <= outer[1] and inner[2] >= outer[2] and inner[3] <= outer[3]


def assign_tiles(tiles: pd.DataFrame, grid: BlockGrid, seed: int, ood_blocks: set[str] | None = None,
                 strata_col: str | None = "realm", fractions: dict[str, float] | None = None,
                 exclude_straddling: bool = True, parent_tier: str = "XXL") -> pd.DataFrame:
    """Add ``block_id``, ``footprint_blocks``, ``region``, ``straddles`` and ``split`` columns to a tile table.

    Hierarchical regions (dataset plan §5): tiles of ``parent_tier`` (XXL) are their own assignment
    regions (overlapping parents are merged and share one assignment); every other tile inherits the
    assignment of the parent footprint that fully contains it, or otherwise of the single grid cell
    that contains it. Tiles straddling two regions are excluded (``split = "excluded"``) for the
    sampler to resample. Held-out cells (``ood_blocks``) and parents centred in them are ``ood_region``.
    ``tiles`` needs ``tile_id, tier, lat, lon, size, pixel_m`` (+ ``realm`` for stratification).
    """
    fractions = fractions or {"train": 0.8, "val": 0.1, "test_id": 0.1}
    t = tiles.copy()
    t["half_m"] = t["size"] * t["pixel_m"] / 2.0
    t["box"] = [tile_box(a, b, h) for a, b, h in zip(t.lat, t.lon, t.half_m, strict=True)]
    t["block_id"] = [grid.block_id(a, b) for a, b in zip(t.lat, t.lon, strict=True)]
    t["footprint_blocks"] = [grid.footprint_blocks(a, b, h) for a, b, h in zip(t.lat, t.lon, t.half_m, strict=True)]
    strata = None
    if strata_col and strata_col in t:
        strata = t.groupby("block_id")[strata_col].agg(lambda x: x.mode().iloc[0]).to_dict()
    all_blocks = sorted({b for fb in t.footprint_blocks for b in fb})
    assign = assign_blocks(all_blocks, seed, fractions, strata)

    def cell_kind(b: str) -> str:
        return "ood_region" if ood_blocks and b in ood_blocks else assign.get(b, "train")

    # --- parent regions: XXL footprints, merged when they overlap (union-find), one assignment each ---
    parents = t[t.tier == parent_tier]
    pid = {tid: tid for tid in parents.tile_id}

    def find(x):
        while pid[x] != x:
            pid[x] = pid[pid[x]]
            x = pid[x]
        return x

    prow = list(parents.itertuples())
    for i in range(len(prow)):
        for j in range(i + 1, len(prow)):
            if _boxes_intersect(prow[i].box, prow[j].box):
                pid[find(prow[i].tile_id)] = find(prow[j].tile_id)
    comp_of = {r.tile_id: find(r.tile_id) for r in prow}
    comps = sorted(set(comp_of.values()))
    # parents are few (tens): assign them unstratified so that val/test do not round to zero
    comp_assign = assign_blocks(comps, seed + 1, fractions, None)
    for r in prow:
        if ood_blocks and r.block_id in ood_blocks:
            comp_assign[comp_of[r.tile_id]] = "ood_region"
    comp_box = {}
    for r in prow:
        c = comp_of[r.tile_id]
        b = r.box
        comp_box[c] = b if c not in comp_box else (min(comp_box[c][0], b[0]), max(comp_box[c][1], b[1]),
                                                   min(comp_box[c][2], b[2]), max(comp_box[c][3], b[3]))

    # --- assignment per tile ---
    regions, splits, straddles = [], [], []
    for r in t.itertuples():
        if r.tier == parent_tier:
            c = comp_of[r.tile_id]
            regions.append(f"parent:{c}")
            splits.append(comp_assign[c])
            straddles.append(False)
            continue
        hits = [c for c, b in comp_box.items() if _boxes_intersect(r.box, b)]
        if hits:
            inside = [c for c in hits if _box_inside(r.box, comp_box[c])]
            if len(hits) == 1 and inside:
                regions.append(f"parent:{hits[0]}")
                splits.append(comp_assign[hits[0]])
                straddles.append(False)
            else:
                regions.append("straddle:parent")
                splits.append("excluded")
                straddles.append(True)
            continue
        kinds = {cell_kind(b) for b in r.footprint_blocks}
        if len(kinds) == 1:
            regions.append(f"cell:{r.footprint_blocks[0]}")
            splits.append(kinds.pop())
            straddles.append(False)
        else:
            regions.append("straddle:cell")
            splits.append("excluded" if exclude_straddling else tile_split(assign, r.footprint_blocks, ood_blocks))
            straddles.append(True)
    t["region"] = regions
    t["split"] = splits
    t["straddles"] = straddles
    t.attrs["block_assignment"] = assign
    t.attrs["parent_assignment"] = comp_assign
    return t


def check_no_cross_tier_overlap(t: pd.DataFrame) -> list[tuple[str, str]]:
    """Pairs (train/val tile, test/OOD tile) whose bounding boxes intersect, across all tiers (geometric)."""
    if "box" not in t:
        t = t.copy()
        t["box"] = [tile_box(a, b, s * p / 2.0) for a, b, s, p in zip(t.lat, t.lon, t["size"], t.pixel_m, strict=True)]
    trainval = t[t.split.isin(["train", "val"])]
    test = t[~t.split.isin(["train", "val", "excluded"])]
    bad = []
    tb = list(zip(test.tile_id, test.box, strict=True))
    for r in trainval.itertuples():
        for tid, box in tb:
            if _boxes_intersect(r.box, box):
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
           "ood_flags", "apply_holdouts", "summarize", "tile_box", "np"]
