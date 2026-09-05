"""Stratified global sampling of tile centres (brief §4.2).

Strata = RESOLVE biome × realm (continent proxy) × gHM tercile. Candidate points are drawn
uniformly on land (RESOLVE ecoregion polygons), attributed, and then balanced per stratum.
Tiles that are mostly water / ice / NoData are rejected later by :func:`real.extract_tile`.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

REALM_TO_GRIP_REGION = {
    # RESOLVE realm -> GRIP4 region number(s); Palearctic is split by longitude/latitude below
    "Nearctic": [1], "Neotropic": [2], "Afrotropic": [3], "Australasia": [7], "Oceania": [7],
    "Indomalayan": [6], "Antarctica": [],
}
REALM_TO_HYDRO_REGION = {
    "Nearctic": ["na", "ar"], "Neotropic": ["sa", "na"], "Afrotropic": ["af"], "Australasia": ["au"],
    "Oceania": ["au"], "Indomalayan": ["as"], "Palearctic": ["eu", "as", "si", "af"], "Antarctica": [],
}


def grip_region(realm: str, lat: float, lon: float) -> int | None:
    """GRIP4 regional file that covers a point (approximate split of the Palearctic)."""
    if realm == "Palearctic":
        if lon < 60.0 and lat > 35.0:
            return 4          # Europe
        if lon < 100.0:
            return 5          # Middle East & Central Asia
        return 6              # East Asia
    r = REALM_TO_GRIP_REGION.get(realm, [])
    return r[0] if r else None


@dataclass
class Candidate:
    lat: float
    lon: float
    biome_num: int
    biome_name: str
    realm: str
    ecoregion_id: int
    ecoregion_name: str
    ghm: float
    ghm_tercile: int = -1
    grip_region: int | None = None

    @property
    def stratum(self) -> str:
        return f"B{self.biome_num:02d}|{self.realm}|H{self.ghm_tercile}"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["stratum"] = self.stratum
        return d


def sample_land_points(ecoregions, n: int, rng: np.random.Generator, min_lat: float = -60.0,
                       max_lat: float = 72.0):
    """Uniform-on-sphere candidate points that fall on a RESOLVE ecoregion polygon.

    Returns a GeoDataFrame with ecoregion attributes joined. Rock-and-ice, lakes and
    Antarctica are excluded.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    pts = []
    batch = max(1000, 4 * n)
    keep_cols = ["ECO_ID", "ECO_NAME", "BIOME_NUM", "BIOME_NAME", "REALM", "geometry"]
    eco = ecoregions[keep_cols]
    eco = eco[(eco["REALM"] != "Antarctica") & (eco["BIOME_NUM"] != 98) & (eco["BIOME_NUM"] != 99)]
    sindex = eco.sindex  # noqa: F841  (build once)
    while len(pts) < n:
        lon = rng.uniform(-180.0, 180.0, batch)
        # uniform on the sphere: sin(lat) uniform
        smin, smax = math.sin(math.radians(min_lat)), math.sin(math.radians(max_lat))
        lat = np.degrees(np.arcsin(rng.uniform(smin, smax, batch)))
        cand = gpd.GeoDataFrame({"lat": lat, "lon": lon},
                                geometry=[Point(x, y) for x, y in zip(lon, lat, strict=True)], crs="EPSG:4326")
        joined = gpd.sjoin(cand, eco, how="inner", predicate="within")
        joined = joined[~joined.index.duplicated()]
        pts.append(joined)
        if sum(len(p) for p in pts) >= n:
            break
    import pandas as pd

    out = gpd.GeoDataFrame(pd.concat(pts, ignore_index=True), crs="EPSG:4326")
    return out.iloc[:n].reset_index(drop=True)


def sample_ghm(ghm_path: str, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Point-sample the gHM raster (EPSG:4326 or Mollweide); NaN where NoData."""
    import rasterio
    from pyproj import Transformer

    with rasterio.open(ghm_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = tr.transform(lons, lats)
        else:
            xs, ys = lons, lats
        vals = np.array([v[0] for v in src.sample(zip(xs, ys, strict=True))], dtype=np.float64)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
        # gHM v1 GeoTIFF is scaled 0-1 float, but guard against integer-scaled variants
        if np.nanmax(vals) > 1.5:
            vals = vals / np.nanmax(vals)
    return vals


def assign_terciles(values: np.ndarray, edges: tuple[float, float] | None = None
                    ) -> tuple[np.ndarray, tuple[float, float]]:
    """Tercile index 0/1/2 of ``values``; edges computed from the data unless given (recorded)."""
    v = np.asarray(values, dtype=np.float64)
    if edges is None:
        e1, e2 = np.nanquantile(v, [1 / 3, 2 / 3])
        edges = (float(e1), float(e2))
    t = np.where(v <= edges[0], 0, np.where(v <= edges[1], 1, 2))
    t = np.where(np.isnan(v), -1, t)
    return t.astype(int), edges


def balance_strata(candidates: list[Candidate], n_target: int, rng: np.random.Generator,
                   per_stratum_cap: int | None = None, min_biomes: int = 5) -> list[Candidate]:
    """Round-robin selection across strata so counts are as balanced as possible.

    Strata are visited in random order; each round takes one candidate from every stratum that
    still has one, until ``n_target`` is reached. ``per_stratum_cap`` limits the per-stratum count.
    """
    by: dict[str, list[Candidate]] = {}
    for c in candidates:
        if c.ghm_tercile < 0:
            continue
        by.setdefault(c.stratum, []).append(c)
    for lst in by.values():
        rng.shuffle(lst)
    keys = list(by)
    rng.shuffle(keys)
    chosen: list[Candidate] = []
    counts = dict.fromkeys(keys, 0)
    while len(chosen) < n_target and keys:
        progressed = False
        for k in list(keys):
            if not by[k] or (per_stratum_cap and counts[k] >= per_stratum_cap):
                keys.remove(k)
                continue
            chosen.append(by[k].pop())
            counts[k] += 1
            progressed = True
            if len(chosen) >= n_target:
                break
        if not progressed:
            break
    if len({c.biome_num for c in chosen}) < min_biomes:
        raise RuntimeError(f"only {len({c.biome_num for c in chosen})} biomes covered; need {min_biomes}")
    return chosen
