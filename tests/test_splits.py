"""Spatial block splits: determinism, stratification, footprint rule, and NO cross-tier overlap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ampscape.splits import (
    BlockGrid,
    apply_holdouts,
    assign_blocks,
    assign_tiles,
    check_no_cross_tier_overlap,
    ood_flags,
    synthetic_split,
)

TIERS = {"S": (128, 100.0), "M": (256, 100.0), "L": (512, 200.0), "XL": (1024, 500.0), "XXL": (2048, 1000.0)}


def _tiles(n_per_tier=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    realms = ["Nearctic", "Neotropic", "Afrotropic", "Palearctic", "Australasia"]
    for tier, (size, pix) in TIERS.items():
        n = n_per_tier if tier != "XXL" else 40
        for i in range(n):
            rows.append({"tile_id": f"{tier}_{i}", "tier": tier, "lat": float(rng.uniform(-55, 70)),
                         "lon": float(rng.uniform(-180, 180)), "size": size, "pixel_m": pix,
                         "realm": realms[i % len(realms)]})
    return pd.DataFrame(rows)


def test_block_ids_and_footprints():
    g = BlockGrid(20.0, equal_width=False)
    assert g.block_id(0.0, 0.0) == "b004_009" and g.block_id(-90, -180) == "b000_000"
    assert g.block_id(10, 179.9) != g.block_id(10, -179.9)
    ew = BlockGrid(20.0)                                     # equal-width: fewer lon cells toward the poles
    assert ew.n_lon(4) == 18 and ew.n_lon(0) == 3 and ew.n_lon(8) == 3
    assert len(ew.footprint_blocks(15.0, 10.0, 6_400)) == 1  # S tile inside one cell (band 10..30)
    assert len(ew.footprint_blocks(0.0, 0.0, 1_024_000)) == 2  # XXL on a longitude boundary touches two cells
    assert ew.interior_bounds(ew.block_id(15, 10), 256_000) is not None   # XL fits inside a 20° cell
    assert ew.interior_bounds(ew.block_id(15, 10), 1_024_000) is None      # XXL does not (parent regions instead)


def test_assignment_deterministic_and_stratified():
    blocks = [f"b{i:03d}_{j:03d}" for i in range(10, 30) for j in range(0, 20)]
    strata = {b: ("A" if int(b[1:4]) < 20 else "B") for b in blocks}
    a = assign_blocks(blocks, 42, strata=strata)
    b = assign_blocks(blocks, 42, strata=strata)
    c = assign_blocks(blocks, 43, strata=strata)
    assert a == b and a != c
    for s in ("A", "B"):
        counts = pd.Series([a[k] for k in blocks if strata[k] == s]).value_counts(normalize=True)
        assert abs(counts["train"] - 0.8) < 0.03 and abs(counts["val"] - 0.1) < 0.03


def test_no_cross_tier_overlap():
    tiles = _tiles()
    grid = BlockGrid(20.0)
    ood = {grid.block_id(-25, 135)}                          # one Australasian cell as OOD region
    t = assign_tiles(tiles, grid, seed=20260906, ood_blocks=ood)
    assert set(t.split) <= {"train", "val", "test_id", "ood_region", "excluded"}
    assert check_no_cross_tier_overlap(t) == []             # geometric: no train/val box meets a test box
    kept = t[t.split != "excluded"]
    assert (t[t.tier == "XXL"].split == "excluded").sum() == 0          # XXL tiles are never excluded
    assert (t[t.tier == "S"].split == "excluded").mean() < 0.05
    assert (t[t.tier == "XL"].split == "excluded").mean() < 0.6
    small = kept[kept.tier.isin(["S", "M", "L"])]
    frac = small.split.value_counts(normalize=True)
    assert 0.6 < frac.get("train", 0) < 0.92 and frac.get("test_id", 0) > 0.03
    # tiles inside an XXL footprint inherit its assignment
    inh = kept[kept.region.str.startswith("parent:") & (kept.tier != "XXL")]
    par = kept[kept.tier == "XXL"].drop_duplicates("region").set_index("region").split
    assert len(inh) > 0 and all(par[r] == s for r, s in zip(inh.region, inh.split, strict=True))
    touching = kept[kept.footprint_blocks.apply(lambda fb: any(b in ood for b in fb)) & kept.region.str.startswith("cell:")]
    assert (touching.split == "ood_region").all()
    t2 = assign_tiles(tiles, grid, seed=20260906, ood_blocks=ood)
    assert list(t.split) == list(t2.split)


def test_synthetic_split_and_holdouts():
    s = [synthetic_split(i, 20260906) for i in range(2000)]
    frac = pd.Series(s).value_counts(normalize=True)
    assert abs(frac["train"] - 0.8) < 0.03 and synthetic_split(7, 1) == synthetic_split(7, 1)
    cfg = {"ood": {"test_ood_scale": {"tiers": ["XL", "XXL"]}, "test_ood_table": {"table": "forest_bird"},
                   "test_ood_contrast": {"contrast": 10000}}}
    f = ood_flags({"family": "real", "tier": "S", "split": "train", "resistance_table_id": "forest_bird"}, cfg)
    assert f["test_ood_table"] and apply_holdouts("train", f) == "test_ood"
    f = ood_flags({"family": "synthetic", "tier": "XL", "split": "test_id", "contrast": 10000}, cfg)
    assert f["test_ood_scale"] and f["test_ood_contrast"] and apply_holdouts("test_id", f) == "test_id"
    f = ood_flags({"family": "real", "tier": "M", "split": "test_id", "resistance_table_id": "amphibian"}, cfg)
    assert f["test_ood_synth2real"] and not f["test_ood_table"]
