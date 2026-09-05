"""Resistance-table schema (YAML → pydantic) and the covariates → resistance mapping (brief §5).

A resistance table maps the real-tile covariate stack (see ``landscapes/real.py``) to a
resistance raster in ``[1, r_max]`` plus a NoData mask. Every term is optional so that the same
schema expresses a pure human-modification transform (``generic_hm``), class-based expert tables
(``large_mammal``, ``amphibian``, ``forest_bird``) and randomly perturbed tables (``random_table``).

Combination order (all in resistance units, applied to every valid pixel)::

    R  = base                              # landcover class table, or 1 + a·gHM^b, or constant
    R *= slope_factor(slope)               # 1 + per_degree·min(slope, cap)   (multiplicative)
    R *= elevation_factor(elevation)       # per-band multipliers (optional)
    R += road_penalty(road_class) if road_distance <= road_buffer_m   (additive, by GRIP4 class)
    R += ghm_weight · gHM                  # optional additive human-modification term
    R  = water_rule(R, landcover)          # water: barrier value, or NoData
    R  = clip(R, 1, r_max)

NoData mask = WorldCover NoData (class 0) ∪ classes listed in ``nodata_classes``.
Tables are versioned; ``table_id`` + ``version`` + the YAML sha256 are recorded per sample.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from ecoflowbench.landscapes.real import WORLDCOVER_CLASSES, WORLDCOVER_NODATA

WATER_CLASS = 80


class SlopeTerm(BaseModel):
    per_degree: float = Field(ge=0, description="multiplicative increase per degree of slope")
    cap_degrees: float = Field(default=45.0, gt=0, description="slope is clipped to this before use")


class ElevationBand(BaseModel):
    max_m: float
    factor: float = Field(gt=0)


class RoadTerm(BaseModel):
    by_class: dict[int, float] = Field(description="additive resistance per GRIP4 class 1..5")
    buffer_m: float = Field(default=50.0, ge=0, description="pixels within this distance of a road get the penalty")

    @model_validator(mode="after")
    def _classes(self):
        if not set(self.by_class) <= {1, 2, 3, 4, 5}:
            raise ValueError("road classes must be GRIP4 types 1..5")
        return self


class WaterRule(BaseModel):
    mode: Literal["barrier", "nodata", "table"] = "table"
    value: float | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _value(self):
        if self.mode == "barrier" and self.value is None:
            raise ValueError("barrier mode needs a value")
        return self


class GhmBase(BaseModel):
    """base = 1 + a · gHM^b (continuous human-modification transform)."""

    a: float = Field(gt=0)
    b: float = Field(gt=0)


class ResistanceTable(BaseModel):
    table_id: str
    version: int = 1
    description: str = ""
    citations: list[str] = Field(default_factory=list)
    r_max: float = Field(gt=1)
    base: Literal["landcover", "ghm", "constant"] = "landcover"
    landcover: dict[int, float] | None = None
    ghm_base: GhmBase | None = None
    constant_base: float | None = Field(default=None, ge=1)
    slope: SlopeTerm | None = None
    elevation_bands: list[ElevationBand] | None = None
    roads: RoadTerm | None = None
    ghm_additive: float | None = Field(default=None, ge=0)
    water: WaterRule = Field(default_factory=WaterRule)
    nodata_classes: list[int] = Field(default_factory=list)
    random: dict | None = None    # filled in for random_table (seed, log-sd, base_table_id)

    @model_validator(mode="after")
    def _consistent(self):
        if self.base == "landcover":
            if not self.landcover:
                raise ValueError("landcover base needs a landcover table")
            missing = set(WORLDCOVER_CLASSES) - set(self.landcover)
            if missing:
                raise ValueError(f"landcover table missing WorldCover classes {sorted(missing)}")
            bad = [c for c, v in self.landcover.items() if v < 1 or v > self.r_max]
            if bad:
                raise ValueError(f"landcover values outside [1, r_max] for classes {bad}")
        if self.base == "ghm" and self.ghm_base is None:
            raise ValueError("ghm base needs ghm_base")
        if self.base == "constant" and self.constant_base is None:
            raise ValueError("constant base needs constant_base")
        if self.elevation_bands:
            maxes = [b.max_m for b in self.elevation_bands]
            if maxes != sorted(maxes):
                raise ValueError("elevation_bands must be sorted by max_m")
        return self

    # -- helpers -----------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> ResistanceTable:
        p = pathlib.Path(path)
        with open(p) as f:
            d = yaml.safe_load(f)
        t = cls.model_validate(d)
        t._sha256 = hashlib.sha256(p.read_bytes()).hexdigest()  # type: ignore[attr-defined]
        t._path = str(p)  # type: ignore[attr-defined]
        return t

    @property
    def sha256(self) -> str | None:
        return getattr(self, "_sha256", None)

    def provenance(self) -> dict:
        return {"table_id": self.table_id, "version": self.version, "sha256": self.sha256, "r_max": self.r_max}


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
def _lookup(class_map: np.ndarray, table: dict[int, float], default: float) -> np.ndarray:
    lut = np.full(max(max(table), int(class_map.max()) if class_map.size else 0) + 1, default, dtype=np.float64)
    for c, v in table.items():
        lut[c] = v
    return lut[np.clip(class_map, 0, len(lut) - 1)]


def apply_table(table: ResistanceTable, cov: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map a covariate stack to (resistance float32 in [1, r_max], nodata bool, stats).

    ``cov`` needs the channels written by ``landscapes/real.py``. Resistance is 1.0 at NoData.
    """
    lc = np.asarray(cov["landcover"]).astype(np.int32)
    shape = lc.shape
    nodata = lc == WORLDCOVER_NODATA
    for c in table.nodata_classes:
        nodata |= lc == c
    if table.water.mode == "nodata":
        nodata |= lc == WATER_CLASS

    if table.base == "landcover":
        r = _lookup(lc, table.landcover, default=float(table.r_max))
    elif table.base == "ghm":
        g = np.nan_to_num(np.asarray(cov["ghm"], dtype=np.float64), nan=0.0).clip(0, 1)
        r = 1.0 + table.ghm_base.a * np.power(g, table.ghm_base.b)
    else:
        r = np.full(shape, float(table.constant_base), dtype=np.float64)

    if table.slope is not None:
        s = np.nan_to_num(np.asarray(cov["slope"], dtype=np.float64), nan=0.0)
        r = r * (1.0 + table.slope.per_degree * np.clip(s, 0.0, table.slope.cap_degrees))

    if table.elevation_bands:
        e = np.nan_to_num(np.asarray(cov["elevation"], dtype=np.float64), nan=0.0)
        factor = np.ones(shape, dtype=np.float64)
        lower = -np.inf
        for band in table.elevation_bands:
            sel = (e > lower) & (e <= band.max_m)
            factor[sel] = band.factor
            lower = band.max_m
        factor[e > lower] = table.elevation_bands[-1].factor
        r = r * factor

    if table.roads is not None:
        dist = np.asarray(cov["road_distance"], dtype=np.float64)
        cls = np.asarray(cov["road_class"]).astype(np.int32)
        on_road = (dist <= table.roads.buffer_m) & (cls > 0)
        penalty = _lookup(cls, table.roads.by_class, default=0.0)
        r = r + np.where(on_road, penalty, 0.0)

    if table.ghm_additive:
        g = np.nan_to_num(np.asarray(cov["ghm"], dtype=np.float64), nan=0.0).clip(0, 1)
        r = r + table.ghm_additive * g

    if table.water.mode == "barrier":
        r = np.where(lc == WATER_CLASS, float(table.water.value), r)

    r = np.clip(r, 1.0, float(table.r_max))
    r[nodata] = 1.0
    r = r.astype(np.float32)
    valid = r[~nodata]
    stats = {
        "table_id": table.table_id, "r_min": float(valid.min()) if valid.size else np.nan,
        "r_max_obs": float(valid.max()) if valid.size else np.nan,
        "r_median": float(np.median(valid)) if valid.size else np.nan,
        "frac_nodata": float(nodata.mean()), "frac_at_rmax": float((valid >= table.r_max).mean()) if valid.size else np.nan,
        "log10_contrast": float(np.log10(valid.max() / valid.min())) if valid.size else np.nan,
    }
    return r, nodata, stats


def load_tables(directory: str | pathlib.Path) -> dict[str, ResistanceTable]:
    """Load every ``*.yaml`` in a directory keyed by ``table_id``."""
    out = {}
    for p in sorted(pathlib.Path(directory).glob("*.yaml")):
        t = ResistanceTable.from_yaml(p)
        if t.table_id in out:
            raise ValueError(f"duplicate table_id {t.table_id}")
        out[t.table_id] = t
    return out
