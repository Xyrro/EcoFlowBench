"""Tests for ampscape.sources: exact graph, connectivity, points/strips/regions, T3/T4 fields."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from ampscape.landscapes.synthetic import sample_landscape
from ampscape.sources import (
    SourceConfig,
    build_conductance_graph,
    component_labels,
    generate_all,
    laplacian,
    sample_advanced,
    sample_omniscape,
    sample_points,
    sample_regions,
    sample_wall_to_wall,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = SourceConfig.from_yaml(ROOT / "configs" / "tasks" / "sources_default.yaml")


def test_graph_matches_circuitscape_formulas():
    r = np.array([[1.0, 2.0, 4.0], [1.0, 1.0, 1.0], [8.0, 1.0, 2.0]])
    nd = np.zeros((3, 3), bool)
    G, idx = build_conductance_graph(r, nd)
    assert G.shape == (9, 9) and G.nnz // 2 == 20            # 12 cardinal + 8 diagonal edges
    assert np.isclose(G[idx[0, 0], idx[0, 1]], (1 + 0.5) / 2)   # cond_avg
    assert np.isclose(G[idx[0, 0], idx[1, 1]], (1 + 1) / 2 / np.sqrt(2))
    assert np.isclose(G[idx[0, 1], idx[1, 0]], (0.5 + 1) / 2 / np.sqrt(2))
    assert np.isclose(G[idx[2, 0], idx[2, 1]], (1 / 8 + 1) / 2)
    assert (G != G.T).nnz == 0
    G4, _ = build_conductance_graph(r, nd, four_neighbors=True)
    assert G4.nnz // 2 == 12
    Ga, _ = build_conductance_graph(r, nd, avg_resistances=True)
    assert np.isclose(Ga[idx[0, 0], idx[0, 1]], 1 / ((1 + 2) / 2))   # res_avg
    L = laplacian(G)
    assert np.allclose(np.asarray(L.sum(axis=1)).ravel(), 0.0)


def test_graph_excludes_nodata_and_finds_components():
    r = np.ones((6, 6))
    nd = np.zeros((6, 6), bool)
    nd[:, 3] = True                       # full barrier column -> two components
    lab = component_labels(r, nd)
    assert set(np.unique(lab)) == {-1, 0, 1}
    assert (lab[:, :3] == 0).all() and (lab[:, 4:] == 1).all() and (lab[:, 3] == -1).all()
    nd[2, 3] = False                      # one gap reconnects everything (8-neighbour)
    assert component_labels(r, nd).max() == 0
    # diagonal-only contact counts as connected (8-neighbour rule)
    nd = np.ones((3, 3), bool)
    nd[0, 0] = False
    nd[1, 1] = False
    assert component_labels(np.ones((3, 3)), nd).max() == 0


def _landscape(seed, shape=(128, 128)):
    ls = sample_landscape(seed, shape)
    return ls.resistance, ls.nodata_mask


@pytest.mark.parametrize("seed", range(6))
def test_points_contract(seed):
    R, nd = _landscape(seed)
    s = sample_points(R, nd, CFG, np.random.default_rng(seed))
    k = s.k
    assert CFG.points.k_range[0] <= k <= CFG.points.k_range[1]
    assert s.focal_mask.dtype == np.int32 and sorted(np.unique(s.focal_mask).tolist()) == list(range(k + 1))
    rc = [(t["row"], t["col"]) for t in s.focal_table]
    for i, a in enumerate(rc):
        assert not nd[a] and s.focal_mask[a] == s.focal_table[i]["label"]
        for b in rc[i + 1:]:
            assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= s.meta["min_separation_px_used"]
    assert s.meta["connected"] and s.meta["n_components_touched"] == 1
    assert {t["placement"] for t in s.focal_table} <= {"anywhere", "low_resistance"}
    assert s.meta["n_anywhere"] + s.meta["n_low_resistance"] == k
    for t in s.focal_table:
        if t["placement"] == "low_resistance":
            assert t["resistance"] <= s.meta["low_resistance_threshold"]
    assert s.meta["placement"] in ("anywhere", "low_resistance", "mixed")


def test_points_placement_fraction_documented():
    R, nd = _landscape(1)
    n_any = n_tot = 0
    for seed in range(40):
        s = sample_points(R, nd, CFG, np.random.default_rng(seed), k=8)
        n_any += s.meta["n_anywhere"]
        n_tot += s.k
    assert abs(n_any / n_tot - CFG.points.frac_anywhere) < 0.1


def test_points_deterministic():
    R, nd = _landscape(2)
    a = sample_points(R, nd, CFG, np.random.default_rng(11))
    b = sample_points(R, nd, CFG, np.random.default_rng(11))
    np.testing.assert_array_equal(a.focal_mask, b.focal_mask)
    assert a.focal_table == b.focal_table


def test_points_on_split_landscape_stay_in_one_component():
    R = np.ones((64, 64), np.float32)
    nd = np.zeros((64, 64), bool)
    nd[:, 30:34] = True                   # two islands
    for seed in range(5):
        s = sample_points(R, nd, CFG, np.random.default_rng(seed), k=4)
        assert s.meta["connected"]
        cols = [t["col"] for t in s.focal_table]
        assert all(c < 30 for c in cols) or all(c >= 34 for c in cols)


def test_wall_to_wall():
    R, nd = _landscape(3)
    for o, axis in (("NS", 0), ("EW", 1)):
        s = sample_wall_to_wall(R, nd, CFG, o)
        m = s.focal_mask
        w = CFG.wall_to_wall.strip_width_px
        assert s.k == 2 and s.meta["connected"]
        if axis == 0:
            assert (m[w:-w, :] == 0).all() and (m[:w][~nd[:w]] == 1).all() and (m[-w:][~nd[-w:]] <= 2).all()
        else:
            assert (m[:, w:-w] == 0).all()
        assert all(t["wkt"].startswith(("POLYGON", "MULTIPOLYGON")) for t in s.focal_table)
        assert all(t["n_pixels"] > 0 for t in s.focal_table)


def test_regions_from_habitat_patches():
    R = np.ones((128, 128), np.float32)
    nd = np.zeros((128, 128), bool)
    lc = np.full((128, 128), 30, np.int16)                 # grass
    lc[10:30, 10:30] = 10                                  # forest patch A (400 px)
    lc[80:110, 70:110] = 10                                # forest patch B (1200 px)
    lc[60:64, 60:64] = 90                                  # tiny wetland (16 px, below min)
    s = sample_regions(R, nd, lc, CFG, np.random.default_rng(0))
    assert s is not None and s.k == 2 and s.meta["n_eligible_patches"] == 2 and s.meta["connected"]
    assert sorted(t["n_pixels"] for t in s.focal_table) == [400, 1200]
    assert (s.focal_mask[60:64, 60:64] == 0).all()
    assert all(t["kind"] == "region" and t["placement"] == "habitat_patch" for t in s.focal_table)
    assert sample_regions(R, nd, np.full((128, 128), 30, np.int16), CFG, np.random.default_rng(0)) is None


def test_regions_trim_large_patch():
    R = np.ones((128, 128), np.float32)
    nd = np.zeros((128, 128), bool)
    lc = np.full((128, 128), 10, np.int16)                 # everything forest = one huge patch
    lc[:, 60:68] = 40                                      # split by a crop band -> two patches of 7680 px
    s = sample_regions(R, nd, lc, CFG, np.random.default_rng(0))
    assert s is not None and all(t["n_pixels"] <= CFG.regions.max_region_px for t in s.focal_table)


@pytest.mark.parametrize("seed", range(4))
def test_advanced_contract(seed):
    R, nd = _landscape(seed)
    s = sample_advanced(R, nd, CFG, np.random.default_rng(seed))
    src, gnd = s.source_strength, s.ground
    assert src.dtype == np.float32 and gnd.dtype == np.int8
    assert np.isclose(src.sum(), 1.0) and (src >= 0).all()
    assert gnd.sum() > 0 and not (src[gnd > 0] > 0).any() and not src[nd].any() and not gnd[nd].any()
    assert s.meta["connected"] and s.meta["ground"]["mode"] in CFG.advanced.ground.modes


def test_omniscape_contract():
    R, nd = _landscape(4)
    s = sample_omniscape(R, nd, CFG, np.random.default_rng(0))
    src = s.source_strength
    assert np.isclose(src.max(), 1.0) and (src >= 0).all() and not src[nd].any()
    assert s.meta["source_threshold"] == 0.0 and s.meta["n_source_pixels"] > 0.3 * (~nd).sum()


def test_generate_all_records_config_provenance():
    R, nd = _landscape(5)
    out = generate_all(R, nd, CFG, seed=99)
    assert set(out) >= {"points", "wall_to_wall_NS", "wall_to_wall_EW", "advanced", "omniscape"}
    for s in out.values():
        assert s.meta["seed"] == 99 and s.meta["source_config"]["config_id"] == "sources_default_v1"
        assert len(s.meta["source_config"]["sha256"]) == 64
        assert s.meta["connected"]


def test_tier_scaling():
    m = CFG.for_tier("M")
    assert m.points.min_separation_px == 2 * CFG.points.min_separation_px
    assert m.regions.min_patch_px == 4 * CFG.regions.min_patch_px
    assert CFG.for_tier("S") is CFG


def test_kirchhoff_residual_exact_solve():
    """Solve L v = b by hand on a small grid and check the QC residual is ~0 and detects a wrong sink."""
    import scipy.sparse.linalg as spla

    from ampscape.solve.qc import kirchhoff_residual

    R, nd = _landscape(9, (24, 24))
    G, idx = build_conductance_graph(R, nd)
    L = laplacian(G).tocsc()
    valid = idx >= 0
    rc = np.argwhere(valid)
    s_rc, g_rc = tuple(rc[5]), tuple(rc[-5])
    src = np.zeros(R.shape, bool)
    src[s_rc] = True
    gnd = np.zeros(R.shape, bool)
    gnd[g_rc] = True
    keep = np.ones(L.shape[0], bool)
    keep[idx[g_rc]] = False
    b = np.zeros(L.shape[0])
    b[idx[s_rc]] = 1.0
    v = np.zeros(L.shape[0])
    v[keep] = spla.spsolve(L[keep][:, keep].tocsc(), b[keep])
    vmap = np.zeros(R.shape)
    vmap[valid] = v[idx[valid]]
    inj = src.astype(float)
    assert kirchhoff_residual(R, nd, vmap, inj, grounded=gnd) < 1e-9
    assert kirchhoff_residual(R, nd, vmap, gnd.astype(float), grounded=src) > 1.0   # swapped roles are caught


def test_kirchhoff_residual_supernode():
    """A 2-pixel short-circuited source region: rows inside the region are replaced by the net-current equation."""

    from ampscape.solve.qc import kirchhoff_residual

    R = np.ones((16, 16), np.float32)
    nd = np.zeros((16, 16), bool)
    G, idx = build_conductance_graph(R, nd)
    L = laplacian(G).tolil()
    src = np.zeros(R.shape, bool)
    src[3, 3] = src[3, 4] = True
    gnd = np.zeros(R.shape, bool)
    gnd[12, 12] = True
    a, b_ = idx[3, 3], idx[3, 4]
    n = L.shape[0]
    # short-circuit: merge node b_ into a (sum rows/cols), then solve the reduced system
    Lm = L.tocsr()
    keep = np.ones(n, bool)
    keep[[b_, idx[12, 12]]] = False
    Ld = Lm.toarray()
    Ld[a, :] += Ld[b_, :]
    Ld[:, a] += Ld[:, b_]
    rhs = np.zeros(n)
    rhs[a] = 1.0
    v = np.zeros(n)
    v[keep] = np.linalg.solve(Ld[np.ix_(keep, keep)], rhs[keep])
    v[b_] = v[a]
    vmap = np.zeros(R.shape)
    vmap[idx >= 0] = v[idx[idx >= 0]]
    inj = src.astype(float) / 2
    assert kirchhoff_residual(R, nd, vmap, inj, grounded=gnd, supernode=src) < 1e-9
    assert kirchhoff_residual(R, nd, vmap, inj, grounded=gnd) > 0.1     # row-wise check would wrongly fail
