"""Real-landscape tiling (Phase 2, brief §4.2).

A *tile* is a square raster grid in the WGS84/UTM zone of its centre (EPSG:326xx north /
327xx south, owner decision 2026-09-05) at a tier-specific pixel size. For each tile we build a
covariate stack from verified open sources (see ``docs/licenses.md``):

=====================  ===========================================  =========  ======================
channel                source                                       dtype      units / codes
=====================  ===========================================  =========  ======================
``landcover``          ESA WorldCover 2021 v200 (10 m COG, S3)      int16      WorldCover class codes
``elevation``          Copernicus DEM GLO-30 (COG, S3)              float32    m (EGM2008)
``slope``              derived from ``elevation`` on the tile grid  float32    degrees
``road_distance``      GRIP4 vector roads (regional shapefiles)     float32    m to nearest road
``road_class``         GRIP4 ``GP_RTP`` of the nearest road         int16      1 highway … 5 local, 0 none
``river_distance``     HydroRIVERS v1.0                             float32    m to nearest river
``river_order``        HydroRIVERS ``ORD_STRA`` of the nearest river int16     Strahler order, 0 none
``ghm``                gHM v1 (1 km, Kennedy et al. 2019)           float32    0–1
=====================  ===========================================  =========  ======================

Raster sources are read with windowed ``/vsicurl/`` requests directly from the public COGs, so
only the pixels covering the tile are transferred. Vector sources are read with a bounding-box
filter from the locally downloaded regional shapefiles. Everything is reprojected onto the tile
grid with ``rasterio.warp.reproject`` (nearest for categorical, bilinear for continuous).

No network access happens at import time; functions that hit the network are marked.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, transform_bounds
from scipy import ndimage

WORLDCOVER_BUCKET = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WORLDCOVER_VERSION = "v200"
WORLDCOVER_YEAR = 2021
COPDEM30_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
COPDEM90_BUCKET = "https://copernicus-dem-90m.s3.amazonaws.com"

WORLDCOVER_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare",
                      70: "snow_ice", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss_lichen"}
WORLDCOVER_NODATA = 0
GRIP_TYPE_FIELD = "GP_RTP"      # 1 highway, 2 primary, 3 secondary, 4 tertiary, 5 local
HYDRO_ORDER_FIELD = "ORD_STRA"  # Strahler order

CHANNELS = ["landcover", "elevation", "slope", "road_distance", "road_class",
            "river_distance", "river_order", "ghm"]

# GDAL/rasterio environment for anonymous, windowed COG access
GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "5",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
}


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------
def utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the WGS84/UTM zone containing (lat, lon). Polar regions raise."""
    if not (-80.0 <= lat <= 84.0):
        raise ValueError(f"latitude {lat} outside UTM coverage (-80..84)")
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass(frozen=True)
class TileGrid:
    """A north-up square grid in a projected CRS."""

    epsg: int
    transform: Affine
    size: int          # pixels per side
    pixel_m: float

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (self.size, self.size)
        return (left, bottom, right, top)

    def bounds_wgs84(self, pad_deg: float = 0.0) -> tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = transform_bounds(self.crs, CRS.from_epsg(4326), *self.bounds, densify_pts=21)
        return (xmin - pad_deg, ymin - pad_deg, xmax + pad_deg, ymax + pad_deg)

    def to_dict(self) -> dict:
        return {"epsg": self.epsg, "transform": list(self.transform)[:6], "size": self.size,
                "pixel_m": self.pixel_m}


def make_grid(lat: float, lon: float, size: int, pixel_m: float) -> TileGrid:
    """Square tile of ``size`` pixels at ``pixel_m`` resolution centred on (lat, lon) in local UTM."""
    from pyproj import Transformer

    epsg = utm_epsg(lat, lon)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tr.transform(lon, lat)
    half = size * pixel_m / 2.0
    # snap the origin to a whole pixel so grids are reproducible from (lat, lon, size, pixel_m)
    left = math.floor((x - half) / pixel_m) * pixel_m
    top = math.ceil((y + half) / pixel_m) * pixel_m
    return TileGrid(epsg, from_origin(left, top, pixel_m, pixel_m), int(size), float(pixel_m))


# ---------------------------------------------------------------------------
# Source tile naming (pure functions, tested offline)
# ---------------------------------------------------------------------------
def _ns(lat: int) -> str:
    return f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"


def _ew(lon: int) -> str:
    return f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"


def worldcover_tile_urls(bounds_wgs84: tuple[float, float, float, float]) -> list[str]:
    """URLs of the 3°×3° WorldCover map tiles intersecting a WGS84 bbox (l, b, r, t)."""
    xmin, ymin, xmax, ymax = bounds_wgs84
    urls = []
    for lat in range(int(math.floor(ymin / 3.0)) * 3, int(math.floor(ymax / 3.0)) * 3 + 1, 3):
        for lon in range(int(math.floor(xmin / 3.0)) * 3, int(math.floor(xmax / 3.0)) * 3 + 1, 3):
            if not (-90 <= lat < 90 and -180 <= lon < 180):
                continue
            name = f"ESA_WorldCover_10m_{WORLDCOVER_YEAR}_{WORLDCOVER_VERSION}_{_ns(lat)}{_ew(lon)}_Map.tif"
            urls.append(f"{WORLDCOVER_BUCKET}/{WORLDCOVER_VERSION}/{WORLDCOVER_YEAR}/map/{name}")
    return urls


def copdem_tile_urls(bounds_wgs84: tuple[float, float, float, float], resolution: int = 30) -> list[str]:
    """URLs of the 1°×1° Copernicus DEM tiles intersecting a WGS84 bbox."""
    xmin, ymin, xmax, ymax = bounds_wgs84
    bucket, code = (COPDEM30_BUCKET, "10") if resolution == 30 else (COPDEM90_BUCKET, "30")
    urls = []
    for lat in range(int(math.floor(ymin)), int(math.floor(ymax)) + 1):
        for lon in range(int(math.floor(xmin)), int(math.floor(xmax)) + 1):
            if not (-90 <= lat < 90 and -180 <= lon < 180):
                continue
            name = f"Copernicus_DSM_COG_{code}_{_ns(lat)}_00_{_ew(lon)}_00_DEM"
            urls.append(f"{bucket}/{name}/{name}.tif")
    return urls


# ---------------------------------------------------------------------------
# Raster readers (network)
# ---------------------------------------------------------------------------
def _read_into_grid(url_or_path: str, grid: TileGrid, resampling: Resampling, dst: np.ndarray,
                    nodata_fill) -> bool:
    """Reproject one source raster (COG URL or local file) onto ``grid`` in place; False if missing."""
    src_path = f"/vsicurl/{url_or_path}" if url_or_path.startswith("http") else url_or_path
    try:
        with rasterio.open(src_path) as src:
            if src.crs.to_epsg() == 4326:
                xmin, ymin, xmax, ymax = grid.bounds_wgs84(pad_deg=0.02)
            else:
                xmin, ymin, xmax, ymax = transform_bounds(grid.crs, src.crs, *grid.bounds, densify_pts=21)
            window = src.window(xmin, ymin, xmax, ymax).round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                return False
            data = src.read(1, window=window, boundless=True, fill_value=src.nodata if src.nodata is not None else nodata_fill)
            src_transform = src.window_transform(window)
            tmp = np.full(grid.shape, nodata_fill, dtype=dst.dtype)
            reproject(data, tmp, src_transform=src_transform, src_crs=src.crs,
                      src_nodata=src.nodata, dst_transform=grid.transform, dst_crs=grid.crs,
                      dst_nodata=nodata_fill, resampling=resampling)
            # pixels this source actually covered (NaN-safe: NaN != NaN is always True)
            if isinstance(nodata_fill, float) and math.isnan(nodata_fill):
                filled = np.isfinite(tmp)
            else:
                filled = tmp != nodata_fill
            dst[filled] = tmp[filled]
            return bool(filled.any())
    except rasterio.errors.RasterioIOError:
        return False


def read_worldcover(grid: TileGrid) -> np.ndarray:
    """[network] WorldCover class raster on the tile grid (int16, 0 = NoData)."""
    out = np.full(grid.shape, WORLDCOVER_NODATA, dtype=np.int16)
    with rasterio.Env(**GDAL_ENV):
        for url in worldcover_tile_urls(grid.bounds_wgs84(0.05)):
            _read_into_grid(url, grid, Resampling.nearest, out, WORLDCOVER_NODATA)
    return out


def read_copdem(grid: TileGrid) -> np.ndarray:
    """[network] Copernicus DEM elevation (float32 m) on the tile grid; GLO-90 fallback per tile."""
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    with rasterio.Env(**GDAL_ENV):
        bb = grid.bounds_wgs84(0.05)
        for url30, url90 in zip(copdem_tile_urls(bb, 30), copdem_tile_urls(bb, 90), strict=True):
            if not _read_into_grid(url30, grid, Resampling.bilinear, out, np.nan):
                _read_into_grid(url90, grid, Resampling.bilinear, out, np.nan)
    return out


def read_local_raster(path: str, grid: TileGrid, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Reproject a local global raster (e.g. gHM) onto the grid (float32, NaN = NoData)."""
    out = np.full(grid.shape, np.nan, dtype=np.float32)
    _read_into_grid(path, grid, resampling, out, np.nan)
    return out


def fill_nearest(a: np.ndarray, max_fraction: float = 0.10) -> tuple[np.ndarray, float]:
    """Fill NaN pixels with the nearest finite value; returns (filled, fraction_filled).

    Used for seam rows between adjacent source tiles (bilinear resampling leaves a one-pixel
    NaN row at each source boundary) and for sparse NoData in gHM. If more than ``max_fraction``
    of the pixels are NaN the array is returned unchanged so the QC step can reject the tile.
    """
    a = np.asarray(a, dtype=np.float32)
    nan = ~np.isfinite(a)
    frac = float(nan.mean())
    if frac == 0.0 or frac > max_fraction or nan.all():
        return a, 0.0
    _, (iy, ix) = ndimage.distance_transform_edt(nan, return_indices=True)
    out = a[iy, ix]
    return out, frac


def slope_degrees(elev: np.ndarray, pixel_m: float) -> np.ndarray:
    """Slope in degrees from an elevation grid (central differences; NaN propagates)."""
    e = np.asarray(elev, dtype=np.float64)
    filled = np.where(np.isfinite(e), e, np.nanmean(e) if np.isfinite(e).any() else 0.0)
    gy, gx = np.gradient(filled, pixel_m)
    s = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    s[~np.isfinite(e)] = np.nan
    return s


# ---------------------------------------------------------------------------
# Vector -> distance rasters
# ---------------------------------------------------------------------------
def distance_and_attribute(gdf, grid: TileGrid, attr: str, all_touched: bool = True
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise line features and return (distance_m to nearest feature, attribute of nearest).

    ``gdf`` may be empty or None: distance is then a large sentinel (10 × tile extent) and the
    attribute raster is 0. Attribute values must be positive integers.
    """
    extent_m = grid.size * grid.pixel_m
    far = np.float32(10.0 * extent_m)
    if gdf is None or len(gdf) == 0:
        return np.full(grid.shape, far, dtype=np.float32), np.zeros(grid.shape, dtype=np.int16)
    g = gdf.to_crs(grid.crs)
    shapes = [(geom, int(val)) for geom, val in zip(g.geometry, g[attr], strict=True)
              if geom is not None and not geom.is_empty]
    burned = rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=0,
                       all_touched=all_touched, dtype="int32")
    if not burned.any():
        return np.full(grid.shape, far, dtype=np.float32), np.zeros(grid.shape, dtype=np.int16)
    dist, (iy, ix) = ndimage.distance_transform_edt(burned == 0, sampling=grid.pixel_m,
                                                     return_indices=True)
    nearest_attr = burned[iy, ix].astype(np.int16)
    return dist.astype(np.float32), nearest_attr


def read_vectors_bbox(paths: list[str], bounds_wgs84: tuple[float, float, float, float], columns: list[str]):
    """Read features intersecting a WGS84 bbox from one or more shapefiles (pyogrio bbox filter)."""
    import geopandas as gpd
    import pandas as pd

    frames = []
    for p in paths:
        try:
            df = gpd.read_file(p, bbox=bounds_wgs84, columns=columns)
        except Exception as e:  # noqa: BLE001 - surface the file name
            raise RuntimeError(f"failed reading {p}: {e}") from e
        if len(df):
            frames.append(df)
    if not frames:
        return None
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


# ---------------------------------------------------------------------------
# Tile extraction
# ---------------------------------------------------------------------------
@dataclass
class SourcePaths:
    """Local paths of vector / global-raster sources (from scripts/download_sources.py + unzip)."""

    ghm_tif: str
    grip_shapefiles: list[str]
    hydrorivers_shapefiles: list[str]


@dataclass
class TileSpec:
    tile_id: str
    lat: float
    lon: float
    tier: str
    size: int
    pixel_m: float
    stratum: dict = dataclasses.field(default_factory=dict)


def extract_tile(spec: TileSpec, sources: SourcePaths) -> tuple[dict[str, np.ndarray], TileGrid, dict]:
    """[network] Build the covariate stack for one tile. Returns (channels, grid, qc)."""
    grid = make_grid(spec.lat, spec.lon, spec.size, spec.pixel_m)
    bb = grid.bounds_wgs84(0.02)

    landcover = read_worldcover(grid)
    elev_raw = read_copdem(grid)
    elev, frac_dem_filled = fill_nearest(elev_raw)
    slope = slope_degrees(elev, grid.pixel_m)
    ghm_raw = read_local_raster(sources.ghm_tif, grid)
    ghm, frac_ghm_filled = fill_nearest(ghm_raw, max_fraction=0.5)

    roads = read_vectors_bbox(sources.grip_shapefiles, bb, [GRIP_TYPE_FIELD])
    road_dist, road_cls = distance_and_attribute(roads, grid, GRIP_TYPE_FIELD)
    rivers = read_vectors_bbox(sources.hydrorivers_shapefiles, bb, [HYDRO_ORDER_FIELD])
    river_dist, river_ord = distance_and_attribute(rivers, grid, HYDRO_ORDER_FIELD)

    channels = {
        "landcover": landcover, "elevation": elev, "slope": slope,
        "road_distance": road_dist, "road_class": road_cls,
        "river_distance": river_dist, "river_order": river_ord, "ghm": ghm,
    }
    n = landcover.size
    qc = {
        "frac_lc_nodata": float((landcover == WORLDCOVER_NODATA).mean()),
        "frac_water": float((landcover == 80).mean()),
        "frac_snow_ice": float((landcover == 70).mean()),
        "frac_dem_nan": float(np.isnan(elev_raw).mean()),
        "frac_dem_filled": frac_dem_filled,
        "frac_ghm_nan": float(np.isnan(ghm_raw).mean()),
        "frac_ghm_filled": frac_ghm_filled,
        "n_road_features": 0 if roads is None else int(len(roads)),
        "n_river_features": 0 if rivers is None else int(len(rivers)),
        "n_pixels": int(n),
    }
    qc["frac_unusable"] = qc["frac_lc_nodata"] + qc["frac_water"] + qc["frac_snow_ice"]
    # DEM gaps above the infill threshold (10 %) are not filled and make the tile unusable
    qc["accept"] = bool(qc["frac_unusable"] <= 0.9 and qc["frac_dem_nan"] <= 0.10)
    return channels, grid, qc


def write_tile(path: str, channels: dict[str, np.ndarray], grid: TileGrid, spec: TileSpec, qc: dict,
               source_versions: dict) -> str:
    """Write the covariate stack as a tiled, deflate-compressed multi-band GeoTIFF; return sha256."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    bands = CHANNELS
    profile = {
        "driver": "GTiff", "height": grid.size, "width": grid.size, "count": len(bands),
        "dtype": "float32", "crs": grid.crs, "transform": grid.transform, "nodata": -9999.0,
        "tiled": True, "blockxsize": min(256, grid.size), "blockysize": min(256, grid.size),
        "compress": "deflate", "predictor": 2, "zlevel": 6,
    }
    with rasterio.open(path, "w", **profile) as dst:
        for i, name in enumerate(bands, start=1):
            a = channels[name].astype(np.float32)
            a = np.where(np.isfinite(a), a, -9999.0)
            dst.write(a, i)
            dst.set_band_description(i, name)
        dst.update_tags(tile_id=spec.tile_id, lat=spec.lat, lon=spec.lon, tier=spec.tier,
                        pixel_m=spec.pixel_m, epsg=grid.epsg, stratum=json.dumps(spec.stratum),
                        qc=json.dumps(qc), source_versions=json.dumps(source_versions),
                        channels=",".join(bands))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tile(path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Read a tile GeoTIFF back into a channel dict (NaN for NoData) plus its tags."""
    with rasterio.open(path) as src:
        names = src.tags().get("channels", ",".join(CHANNELS)).split(",")
        out = {}
        for i, name in enumerate(names, start=1):
            a = src.read(i).astype(np.float32)
            a[a == -9999.0] = np.nan
            if name in ("landcover", "road_class", "river_order"):
                a = np.where(np.isnan(a), 0, a).astype(np.int16)
            out[name] = a
        return out, dict(src.tags())


def local_sources_from_dir(sources_dir: str) -> SourcePaths:
    """Locate unzipped sources under ``sources_dir`` (see scripts/download_sources.py)."""
    root = pathlib.Path(sources_dir)
    ghm = sorted(root.glob("gHM/**/gHM.tif")) or sorted(root.glob("**/gHM*.tif"))
    grip = sorted(str(p) for p in root.glob("GRIP4_Region*/**/*.shp"))
    hydro = sorted(str(p) for p in root.glob("HydroRIVERS_v10_*/**/*.shp"))
    if not ghm:
        raise FileNotFoundError(f"gHM.tif not found under {root}")
    return SourcePaths(str(ghm[0]), grip, hydro)


def env_gdal() -> dict:
    """os.environ additions for anonymous COG access (used by scripts before importing rasterio)."""
    return {k: v for k, v in GDAL_ENV.items() if k not in os.environ}
