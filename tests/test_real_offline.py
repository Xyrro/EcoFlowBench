"""Offline tests for the real-tile pipeline (no network): grid geometry, tile naming, rasterisation."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from ecoflowbench.landscapes import real, sampling


@pytest.mark.parametrize("lat,lon,epsg", [
    (33.77, -84.39, 32616),   # Atlanta, zone 16N
    (-33.87, 151.21, 32756),  # Sydney, zone 56S
    (0.0, 0.0, 32631),        # zone 31N (lon 0 belongs to zone 31)
    (51.5, -0.1, 32630),      # London, zone 30N
    (64.1, -21.9, 32627),     # Reykjavik, 27N
    (-1.3, 36.8, 32737),      # Nairobi, 37S
])
def test_utm_epsg(lat, lon, epsg):
    assert real.utm_epsg(lat, lon) == epsg


def test_utm_epsg_polar_raises():
    with pytest.raises(ValueError):
        real.utm_epsg(85.0, 10.0)


def test_make_grid_geometry_and_reproducibility():
    g1 = real.make_grid(33.77, -84.39, 128, 100.0)
    g2 = real.make_grid(33.77, -84.39, 128, 100.0)
    assert g1 == g2 and g1.epsg == 32616 and g1.shape == (128, 128)
    left, bottom, right, top = g1.bounds
    assert np.isclose(right - left, 12800.0) and np.isclose(top - bottom, 12800.0)
    assert g1.transform.a == 100.0 and g1.transform.e == -100.0  # north-up
    xmin, ymin, xmax, ymax = g1.bounds_wgs84()
    assert xmin < -84.39 < xmax and ymin < 33.77 < ymax
    assert abs((xmax - xmin) * 111e3 * np.cos(np.radians(33.77)) - 12800) < 600  # ~12.8 km wide


def test_worldcover_tile_naming():
    urls = real.worldcover_tile_urls((-84.5, 33.7, -84.3, 33.9))
    assert urls == [f"{real.WORLDCOVER_BUCKET}/v200/2021/map/ESA_WorldCover_10m_2021_v200_N33W087_Map.tif"]
    urls = real.worldcover_tile_urls((2.9, -0.1, 3.1, 0.1))   # crosses lat 0 and lon 3
    names = sorted(u.rsplit("_", 2)[-2] for u in urls)
    assert names == ["N00E000", "N00E003", "S03E000", "S03E003"]


def test_copdem_tile_naming():
    urls = real.copdem_tile_urls((-84.5, 33.7, -84.3, 33.9))
    assert urls == [f"{real.COPDEM30_BUCKET}/Copernicus_DSM_COG_10_N33_00_W085_00_DEM/"
                    "Copernicus_DSM_COG_10_N33_00_W085_00_DEM.tif"]
    assert real.copdem_tile_urls((-84.5, -33.9, -84.3, -33.7), 90)[0].startswith(real.COPDEM90_BUCKET)
    assert "S34_00_W085_00" in real.copdem_tile_urls((-84.5, -33.9, -84.3, -33.7))[0]


def test_slope_degrees_plane():
    y, x = np.mgrid[0:64, 0:64].astype(float)
    elev = 10.0 * x  # 10 m rise per 100 m pixel -> atan(0.1) = 5.71 deg
    s = real.slope_degrees(elev, 100.0)
    assert np.allclose(s[1:-1, 1:-1], np.degrees(np.arctan(0.1)), atol=1e-4)
    elev[5, 5] = np.nan
    assert np.isnan(real.slope_degrees(elev, 100.0)[5, 5])


def test_distance_and_attribute_line():
    import geopandas as gpd
    from shapely.geometry import LineString

    g = real.make_grid(33.77, -84.39, 64, 100.0)
    xmin, ymin, xmax, ymax = g.bounds
    ymid = (ymin + ymax) / 2
    line = gpd.GeoDataFrame({"GP_RTP": [3]}, geometry=[LineString([(xmin - 10, ymid), (xmax + 10, ymid)])], crs=g.crs)
    dist, cls = real.distance_and_attribute(line, g, "GP_RTP")
    assert dist.shape == (64, 64) and cls.dtype == np.int16
    row = int((ymax - ymid) / 100.0)
    assert dist[row, :].max() <= 100.0 and (cls == 3).any() and cls.max() == 3
    # distance grows by ~100 m per row away from the line
    assert np.isclose(dist[row + 5, 10] - dist[row + 1, 10], 400.0, atol=100.0)
    # empty input -> sentinel far distance, class 0
    d0, c0 = real.distance_and_attribute(None, g, "GP_RTP")
    assert d0.min() > 60_000 and c0.max() == 0


def test_write_read_tile_roundtrip(tmp_path):
    g = real.make_grid(33.77, -84.39, 32, 100.0)
    rng = np.random.default_rng(0)
    ch = {
        "landcover": rng.choice([10, 30, 40, 80], (32, 32)).astype(np.int16),
        "elevation": rng.normal(300, 20, (32, 32)).astype(np.float32),
        "slope": rng.uniform(0, 10, (32, 32)).astype(np.float32),
        "road_distance": rng.uniform(0, 5000, (32, 32)).astype(np.float32),
        "road_class": rng.integers(0, 6, (32, 32)).astype(np.int16),
        "river_distance": rng.uniform(0, 5000, (32, 32)).astype(np.float32),
        "river_order": rng.integers(0, 5, (32, 32)).astype(np.int16),
        "ghm": rng.uniform(0, 1, (32, 32)).astype(np.float32),
    }
    ch["elevation"][0, 0] = np.nan
    spec = real.TileSpec("t0", 33.77, -84.39, "S", 32, 100.0, {"biome": 4})
    p = tmp_path / "t0.tif"
    sha = real.write_tile(str(p), ch, g, spec, {"accept": True}, {"worldcover": "2021 v200"})
    assert len(sha) == 64
    back, tags = real.read_tile(str(p))
    for k in real.CHANNELS:
        if k == "elevation":
            assert np.isnan(back[k][0, 0]) and np.allclose(back[k][1:], ch[k][1:])
        else:
            np.testing.assert_array_equal(back[k], ch[k])
    assert tags["tile_id"] == "t0" and tags["epsg"] == "32616"
    with rasterio.open(p) as src:
        assert src.descriptions == tuple(real.CHANNELS) and src.crs.to_epsg() == 32616


def test_sampling_helpers():
    t, edges = sampling.assign_terciles(np.array([0.0, 0.1, 0.2, 0.5, 0.8, 0.9, np.nan]))
    assert set(t.tolist()) == {0, 1, 2, -1} and edges[0] < edges[1]
    assert sampling.grip_region("Palearctic", 48.0, 2.0) == 4
    assert sampling.grip_region("Palearctic", 35.0, 51.0) == 5
    assert sampling.grip_region("Palearctic", 35.0, 120.0) == 6
    assert sampling.grip_region("Nearctic", 40.0, -100.0) == 1
    rng = np.random.default_rng(0)
    cands = [sampling.Candidate(0, 0, b, "b", "R", 1, "e", 0.1, t) for b in range(1, 8) for t in range(3) for _ in range(4)]
    chosen = sampling.balance_strata(cands, 50, rng, min_biomes=5)
    assert len(chosen) == 50
    from collections import Counter
    counts = Counter(c.stratum for c in chosen)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_read_into_grid_merges_partial_sources(tmp_path):
    """Two source rasters each covering half the tile must merge without NaN overwrite (regression)."""
    from rasterio.transform import from_origin

    g = real.make_grid(33.77, -84.39, 32, 100.0)
    xmin, ymin, xmax, ymax = g.bounds
    xmid = (xmin + xmax) / 2
    paths = []
    for i, (x0, x1) in enumerate([(xmin - 500, xmid), (xmid, xmax + 500)]):
        w = int(round((x1 - x0) / 50.0))
        h = int(round((ymax - ymin + 1000) / 50.0))
        data = np.full((h, w), 100.0 * (i + 1), dtype=np.float32)
        p = tmp_path / f"src{i}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=g.crs,
                           transform=from_origin(x0, ymax + 500, 50.0, 50.0)) as dst:
            dst.write(data, 1)
        paths.append(str(p))
    out = np.full(g.shape, np.nan, dtype=np.float32)
    for p in paths:  # order matters for the regression: the second source must not erase the first
        assert real._read_into_grid(p, g, real.Resampling.nearest, out, np.nan)
    assert np.isfinite(out).all()
    assert set(np.unique(out).tolist()) == {100.0, 200.0}


def test_fill_nearest():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    a[1, 1] = np.nan
    b, frac = real.fill_nearest(a)
    assert np.isfinite(b).all() and frac == 1 / 16 and b[1, 1] in {a[0, 1], a[1, 0], a[1, 2], a[2, 1]}
    a[:, :] = np.nan
    c, frac = real.fill_nearest(a)
    assert frac == 0.0 and np.isnan(c).all()
    a2 = np.zeros((4, 4), np.float32)
    a2[:2] = np.nan
    d, frac = real.fill_nearest(a2, max_fraction=0.1)
    assert frac == 0.0 and np.isnan(d[0, 0])  # above threshold -> untouched
