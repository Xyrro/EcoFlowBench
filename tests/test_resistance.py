"""Tests for resistance tables: schema, value ranges, mask consistency, term behaviour, determinism."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml

from ampscape.landscapes.real import CHANNELS, WORLDCOVER_CLASSES
from ampscape.resistance import ResistanceTable, apply_table, load_tables, perturb_table

ROOT = pathlib.Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "configs" / "resistance_tables"
PILOT = ROOT / "data" / "tiles" / "pilot"


def synthetic_cov(seed: int = 0, n: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.array(list(WORLDCOVER_CLASSES) + [0])
    cov = {
        "landcover": rng.choice(classes, (n, n)).astype(np.int16),
        "elevation": rng.uniform(0, 4000, (n, n)).astype(np.float32),
        "slope": rng.uniform(0, 60, (n, n)).astype(np.float32),
        "road_distance": rng.uniform(0, 3000, (n, n)).astype(np.float32),
        "road_class": rng.integers(0, 6, (n, n)).astype(np.int16),
        "river_distance": rng.uniform(0, 3000, (n, n)).astype(np.float32),
        "river_order": rng.integers(0, 6, (n, n)).astype(np.int16),
        "ghm": rng.uniform(0, 1, (n, n)).astype(np.float32),
    }
    cov["ghm"][0, 0] = np.nan
    cov["elevation"][0, 1] = np.nan
    assert set(cov) == set(CHANNELS)
    return cov


@pytest.fixture(scope="module")
def tables():
    t = load_tables(TABLE_DIR)
    assert len(t) >= 4
    return t


def test_tables_load_and_have_citations(tables):
    for t in tables.values():
        assert t.citations, t.table_id
        assert t.sha256 and len(t.sha256) == 64
        assert t.r_max > 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ranges_and_mask(tables, seed):
    cov = synthetic_cov(seed)
    for t in tables.values():
        r, nodata, stats = apply_table(t, cov)
        assert r.dtype == np.float32 and r.shape == cov["landcover"].shape and nodata.dtype == bool
        assert np.isfinite(r).all()
        assert r[~nodata].min() >= 1.0 and r[~nodata].max() <= t.r_max
        assert np.all(r[nodata] == 1.0)
        expect_nd = cov["landcover"] == 0
        if t.water.mode == "nodata":
            expect_nd |= cov["landcover"] == 80
        np.testing.assert_array_equal(nodata, expect_nd)
        assert 0 <= stats["frac_nodata"] <= 1


def test_deterministic(tables):
    cov = synthetic_cov(5)
    for t in tables.values():
        a, _, _ = apply_table(t, cov)
        b, _, _ = apply_table(t, cov)
        np.testing.assert_array_equal(a, b)


def test_term_behaviour_large_mammal(tables):
    t = tables["large_mammal"]
    n = 8
    cov = {k: np.zeros((n, n), np.float32) for k in CHANNELS}
    cov["landcover"] = np.full((n, n), 10, np.int16)          # forest
    cov["road_class"] = np.zeros((n, n), np.int16)
    cov["road_distance"] = np.full((n, n), 5000, np.float32)
    base, _, _ = apply_table(t, cov)
    assert np.all(base == 1.0)
    # highway pixels get the class-1 penalty
    cov2 = {k: v.copy() for k, v in cov.items()}
    cov2["road_class"][2, :] = 1
    cov2["road_distance"][2, :] = 0
    r2, _, _ = apply_table(t, cov2)
    assert np.allclose(r2[2], 1 + t.roads.by_class[1]) and np.all(r2[3] == 1.0)
    # slope multiplies
    cov3 = {k: v.copy() for k, v in cov.items()}
    cov3["slope"][:] = 30
    r3, _, _ = apply_table(t, cov3)
    assert np.allclose(r3, 1 + 0.03 * 30)
    # water is a barrier regardless of class value; built-up is high; nodata is masked
    cov4 = {k: v.copy() for k, v in cov.items()}
    cov4["landcover"][0, :] = 80
    cov4["landcover"][1, :] = 50
    cov4["landcover"][2, :] = 0
    r4, nd4, _ = apply_table(t, cov4)
    assert np.all(r4[0] == 800) and np.all(r4[1] == 500) and nd4[2].all() and not nd4[3].any()
    # ordering forest < crop < built
    assert t.landcover[10] < t.landcover[40] < t.landcover[50]


def test_generic_hm_curve(tables):
    t = tables["generic_hm"]
    n = 4
    cov = {k: np.zeros((n, n), np.float32) for k in CHANNELS}
    cov["landcover"] = np.full((n, n), 30, np.int16)
    cov["ghm"] = np.array([[0.0, 0.5, 1.0, np.nan]] * n, np.float32)
    r, _, _ = apply_table(t, cov)
    assert np.allclose(r[:, 0], 1.0) and np.allclose(r[:, 2], 1000.0) and np.allclose(r[:, 3], 1.0)
    assert np.allclose(r[:, 1], 1 + 999 * 0.25)


def test_forest_bird_elevation_bands(tables):
    t = tables["forest_bird"]
    n = 4
    cov = {k: np.zeros((n, n), np.float32) for k in CHANNELS}
    cov["landcover"] = np.full((n, n), 30, np.int16)   # grass = 15
    cov["elevation"] = np.array([[500, 2000, 4000, 9000]] * n, np.float32)
    cov["road_distance"][:] = 9999
    r, _, _ = apply_table(t, cov)
    assert np.allclose(r[:, 0], 15) and np.allclose(r[:, 1], 30) and np.allclose(r[:, 2], 60) and np.allclose(r[:, 3], 60)


def test_schema_rejects_bad_tables(tables):
    d = yaml.safe_load((TABLE_DIR / "large_mammal.yaml").read_text())
    bad = dict(d)
    bad["landcover"] = {k: v for k, v in d["landcover"].items() if k != 95}
    with pytest.raises(ValueError):
        ResistanceTable.model_validate(bad)
    bad = dict(d)
    bad["landcover"] = {**d["landcover"], 50: 5000}
    with pytest.raises(ValueError):
        ResistanceTable.model_validate(bad)
    bad = dict(d)
    bad["roads"] = {"by_class": {9: 1}}
    with pytest.raises(ValueError):
        ResistanceTable.model_validate(bad)


def test_random_table_reproducible(tables):
    base = tables["large_mammal"]
    a = perturb_table(base, seed=7, log_sd=0.5)
    b = perturb_table(base, seed=7, log_sd=0.5)
    c = perturb_table(base, seed=8, log_sd=0.5)
    assert a.landcover == b.landcover and a.landcover != c.landcover
    assert all(1 <= v <= base.r_max for v in a.landcover.values())
    assert a.random["seed"] == 7 and a.random["base_table_id"] == "large_mammal"
    cov = synthetic_cov(3)
    r, nd, _ = apply_table(a, cov)
    assert r[~nd].min() >= 1 and r[~nd].max() <= a.r_max


@pytest.mark.skipif(not (PILOT / "tiles.parquet").exists(), reason="pilot tiles not on this machine")
def test_all_tables_on_all_pilot_tiles(tables):
    import pandas as pd

    from ampscape.landscapes.real import read_tile

    df = pd.read_parquet(PILOT / "tiles.parquet")
    df = df[df["qc_accept"]]
    assert len(df) >= 50
    for _, row in df.iterrows():
        cov, _ = read_tile(str(PILOT / row["path"]))
        for t in tables.values():
            r, nd, stats = apply_table(t, cov)
            assert np.isfinite(r).all()
            assert r[~nd].min() >= 1.0 and r[~nd].max() <= t.r_max
            assert nd.mean() < 0.9, (row["tile_id"], t.table_id)
            np.testing.assert_array_equal(nd, cov["landcover"] == 0)
