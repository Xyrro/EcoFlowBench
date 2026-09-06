"""Source-configuration schema (YAML -> pydantic) with per-tier scaling. See configs/tasks/sources_default.yaml."""

from __future__ import annotations

import hashlib
import pathlib
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class PointsCfg(BaseModel):
    k_range: tuple[int, int] = (2, 8)
    min_separation_px: int = Field(default=16, ge=1)
    frac_anywhere: float = Field(default=0.3, ge=0, le=1)
    low_resistance_quantile: float = Field(default=0.25, gt=0, lt=1)
    max_attempts: int = Field(default=200, ge=1)


class WallToWallCfg(BaseModel):
    strip_width_px: int = Field(default=2, ge=1)


class RegionsCfg(BaseModel):
    habitat_classes: list[int] = Field(default_factory=lambda: [10, 90, 95])
    min_patch_px: int = Field(default=50, ge=1)
    k_range: tuple[int, int] = (2, 6)
    min_separation_px: int = Field(default=16, ge=1)
    max_region_px: int = Field(default=2000, ge=1)


class SourceFieldCfg(BaseModel):
    mode: Literal["inverse_resistance", "random_field"] = "inverse_resistance"
    power: float = Field(default=1.0, gt=0)
    quantile: float = Field(default=0.7, ge=0, lt=1)
    normalize_total: float | None = 1.0
    scale_max: float | None = None
    random_field_length_scale: float = Field(default=16.0, gt=0)


class GroundCfg(BaseModel):
    modes: list[Literal["edge", "all_edges", "patches"]] = Field(default_factory=lambda: ["edge", "all_edges", "patches"])
    edge_width_px: int = Field(default=1, ge=1)
    n_patches_range: tuple[int, int] = (1, 3)
    patch_radius_px: int = Field(default=3, ge=1)


class AdvancedCfg(BaseModel):
    source: SourceFieldCfg = Field(default_factory=SourceFieldCfg)
    ground: GroundCfg = Field(default_factory=GroundCfg)


class OmniscapeCfg(BaseModel):
    source: SourceFieldCfg = Field(default_factory=lambda: SourceFieldCfg(quantile=0.5, normalize_total=None, scale_max=1.0))
    source_threshold: float = Field(default=0.0, ge=0)


class SourceConfig(BaseModel):
    config_id: str
    points: PointsCfg = Field(default_factory=PointsCfg)
    wall_to_wall: WallToWallCfg = Field(default_factory=WallToWallCfg)
    regions: RegionsCfg = Field(default_factory=RegionsCfg)
    advanced: AdvancedCfg = Field(default_factory=AdvancedCfg)
    omniscape: OmniscapeCfg = Field(default_factory=OmniscapeCfg)
    tiers: dict[str, dict] = Field(default_factory=lambda: {"S": {"scale": 1}})
    sha256: str | None = None

    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> SourceConfig:
        p = pathlib.Path(path)
        d = yaml.safe_load(p.read_text())
        d["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        return cls.model_validate(d)

    def for_tier(self, tier: str) -> SourceConfig:
        """Scale every pixel quantity by the tier's `scale` (raster size / 128)."""
        scale = int(self.tiers.get(tier, {"scale": 1}).get("scale", 1))
        if scale == 1:
            return self
        d = self.model_dump()
        d["points"]["min_separation_px"] *= scale
        d["wall_to_wall"]["strip_width_px"] *= scale
        d["regions"]["min_patch_px"] *= scale * scale
        d["regions"]["min_separation_px"] *= scale
        d["regions"]["max_region_px"] *= scale * scale
        d["advanced"]["source"]["random_field_length_scale"] *= scale
        d["advanced"]["ground"]["edge_width_px"] *= scale
        d["advanced"]["ground"]["patch_radius_px"] *= scale
        return SourceConfig.model_validate(d)

    def provenance(self) -> dict:
        return {"config_id": self.config_id, "sha256": self.sha256}
