"""Focal-node and source-configuration generators (brief §6).

All generators take the resistance raster ``R`` (float32, 1.0 at NoData), the NoData mask and a
``numpy.random.Generator``; they return :class:`SourceSample` objects whose ``focal_mask`` is an
int32 label raster (0 = none), ``focal_table`` a list of node records, and whose metadata
records every parameter and the connectivity check result.

Connectivity is checked on the **exact solver graph** (``sources.graph``: 8-neighbour,
NoData removed, average conductance), never on an approximation: focal candidates are drawn
from the largest connected component and, after placement, every focal pixel is verified to
lie in one component (``meta["connected"]``).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ecoflowbench.sources.config import SourceConfig, SourceFieldCfg
from ecoflowbench.sources.graph import all_in_one_component, component_labels


@dataclass
class SourceSample:
    kind: str                                  # points | wall_to_wall | regions | advanced | omniscape
    focal_mask: np.ndarray | None = None       # int32 (H, W)
    focal_table: list[dict] = field(default_factory=list)
    source_strength: np.ndarray | None = None  # float32 (H, W)  (advanced / omniscape)
    ground: np.ndarray | None = None           # int8 (H, W)     (advanced)
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.focal_table)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "k": self.k, "focal_table": self.focal_table, "meta": self.meta}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _largest_component_mask(R: np.ndarray, nodata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = component_labels(R, nodata)
    return labels == 0, labels


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _connectivity_meta(labels: np.ndarray, focal_mask: np.ndarray) -> dict:
    ok, n_comp = all_in_one_component(labels, focal_mask > 0)
    return {"connected": ok, "n_components_touched": n_comp, "n_graph_components": int(labels.max() + 1)}


def _region_wkt(mask: np.ndarray) -> str:
    """Polygon WKT (pixel coordinates: x = col, y = row) of a boolean region."""
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    geoms = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask) if v == 1]
    return unary_union(geoms).wkt if geoms else "POLYGON EMPTY"


# ---------------------------------------------------------------------------
# T1 / T2: point focal nodes
# ---------------------------------------------------------------------------
def sample_points(R: np.ndarray, nodata: np.ndarray, cfg: SourceConfig, rng: np.random.Generator,
                  k: int | None = None) -> SourceSample:
    """K point focal nodes with minimum Chebyshev separation, mixed placement.

    Each node is independently placed "anywhere" (any pixel of the largest component) with
    probability ``frac_anywhere``, otherwise restricted to low-resistance pixels
    (R <= ``low_resistance_quantile`` quantile of valid pixels) of the largest component. The
    placement of each node is recorded in the table and summarised in ``meta``.
    """
    pc = cfg.points
    H, W = R.shape
    comp, labels = _largest_component_mask(R, nodata)
    k = int(rng.integers(pc.k_range[0], pc.k_range[1] + 1)) if k is None else int(k)
    valid_r = R[~nodata]
    thr = float(np.quantile(valid_r, pc.low_resistance_quantile))
    low = comp & (R <= thr)
    any_rc = np.argwhere(comp)
    low_rc = np.argwhere(low)
    if len(any_rc) < k:
        raise RuntimeError("largest component has fewer pixels than K")
    sep = pc.min_separation_px
    chosen: list[tuple[int, int]] = []
    placements: list[str] = []
    attempts = 0
    relaxations = 0
    while len(chosen) < k:
        want_any = rng.uniform() < pc.frac_anywhere or len(low_rc) == 0
        pool = any_rc if want_any else low_rc
        cand = tuple(int(x) for x in pool[rng.integers(len(pool))])
        attempts += 1
        if all(_cheb(cand, c) >= sep for c in chosen):
            chosen.append(cand)
            placements.append("anywhere" if want_any else "low_resistance")
        elif attempts >= pc.max_attempts:
            sep = max(1, int(math.floor(sep * 0.75)))
            relaxations += 1
            attempts = 0
    mask = np.zeros((H, W), dtype=np.int32)
    table = []
    for i, ((r, c), p) in enumerate(zip(chosen, placements, strict=True), start=1):
        mask[r, c] = i
        table.append({"label": i, "row": r, "col": c, "kind": "point", "placement": p, "n_pixels": 1,
                      "resistance": float(R[r, c])})
    meta = {
        "k": k, "min_separation_px_requested": pc.min_separation_px, "min_separation_px_used": sep,
        "separation_relaxations": relaxations, "low_resistance_threshold": thr,
        "n_anywhere": placements.count("anywhere"), "n_low_resistance": placements.count("low_resistance"),
        "placement": "anywhere" if placements.count("anywhere") == k else
                     "low_resistance" if placements.count("low_resistance") == k else "mixed",
        **_connectivity_meta(labels, mask),
    }
    return SourceSample("points", mask, table, meta=meta)


# ---------------------------------------------------------------------------
# T1W: wall-to-wall strips
# ---------------------------------------------------------------------------
def sample_wall_to_wall(R: np.ndarray, nodata: np.ndarray, cfg: SourceConfig, orientation: str) -> SourceSample:
    """Two edge strips (label 1 = north or west, 2 = south or east) restricted to the largest component."""
    if orientation not in ("NS", "EW"):
        raise ValueError("orientation must be NS or EW")
    H, W = R.shape
    w = cfg.wall_to_wall.strip_width_px
    comp, labels = _largest_component_mask(R, nodata)
    mask = np.zeros((H, W), dtype=np.int32)
    if orientation == "NS":
        mask[:w, :] = 1
        mask[H - w:, :] = 2
    else:
        mask[:, :w] = 1
        mask[:, W - w:] = 2
    mask[~comp] = 0
    table = []
    for lab in (1, 2):
        rc = np.argwhere(mask == lab)
        if len(rc) == 0:
            raise RuntimeError(f"strip {lab} has no valid pixels in the largest component")
        table.append({"label": lab, "row": int(round(rc[:, 0].mean())), "col": int(round(rc[:, 1].mean())),
                      "kind": "strip", "placement": "strip", "n_pixels": int(len(rc)), "wkt": _region_wkt(mask == lab)})
    meta = {"orientation": orientation, "strip_width_px": w, "k": 2, "placement": "strip",
            "strip_pixels": [t["n_pixels"] for t in table], **_connectivity_meta(labels, mask)}
    return SourceSample("wall_to_wall", mask, table, meta=meta)


# ---------------------------------------------------------------------------
# T1 regions from habitat patches
# ---------------------------------------------------------------------------
def sample_regions(R: np.ndarray, nodata: np.ndarray, landcover: np.ndarray, cfg: SourceConfig,
                   rng: np.random.Generator, k: int | None = None) -> SourceSample | None:
    """Focal regions = habitat patches (WorldCover classes in ``habitat_classes``), >= min size.

    Returns None when fewer than ``k_range[0]`` eligible, well-separated patches exist.
    """
    rc_cfg = cfg.regions
    H, W = R.shape
    comp, labels = _largest_component_mask(R, nodata)
    habitat = np.isin(landcover, rc_cfg.habitat_classes) & comp
    lab, n = ndimage.label(habitat, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return None
    sizes = ndimage.sum(habitat, lab, index=np.arange(1, n + 1))
    eligible = [i + 1 for i, s in enumerate(sizes) if s >= rc_cfg.min_patch_px]
    if len(eligible) < rc_cfg.k_range[0]:
        return None
    centroids = {i: tuple(int(round(v)) for v in ndimage.center_of_mass(habitat, lab, i)) for i in eligible}
    k = int(rng.integers(rc_cfg.k_range[0], rc_cfg.k_range[1] + 1)) if k is None else int(k)
    order = list(rng.permutation(eligible))
    chosen: list[int] = []
    for i in order:
        if all(_cheb(centroids[i], centroids[j]) >= rc_cfg.min_separation_px for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    if len(chosen) < rc_cfg.k_range[0]:
        return None
    mask = np.zeros((H, W), dtype=np.int32)
    table = []
    for new_lab, i in enumerate(chosen, start=1):
        m = lab == i
        if m.sum() > rc_cfg.max_region_px:
            # keep exactly the max_region_px patch pixels nearest to the centroid (deterministic)
            cy, cx = centroids[i]
            rc = np.argwhere(m)
            d2 = (rc[:, 0] - cy) ** 2 + (rc[:, 1] - cx) ** 2
            keep = rc[np.argsort(d2, kind="stable")[: rc_cfg.max_region_px]]
            m = np.zeros_like(m)
            m[keep[:, 0], keep[:, 1]] = True
        mask[m] = new_lab
        table.append({"label": new_lab, "row": centroids[i][0], "col": centroids[i][1], "kind": "region",
                      "placement": "habitat_patch", "n_pixels": int(m.sum()), "wkt": _region_wkt(m)})
    meta = {"k": len(chosen), "habitat_classes": rc_cfg.habitat_classes, "min_patch_px": rc_cfg.min_patch_px,
            "n_eligible_patches": len(eligible), "placement": "habitat_patch", **_connectivity_meta(labels, mask)}
    return SourceSample("regions", mask, table, meta=meta)


# ---------------------------------------------------------------------------
# T3 / T4: source strength and grounds
# ---------------------------------------------------------------------------
def suitability_field(R: np.ndarray, nodata: np.ndarray, comp: np.ndarray, sc: SourceFieldCfg,
                      rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Suitability proxy in [0, 1] on the largest component (0 elsewhere)."""
    if sc.mode == "inverse_resistance":
        s = np.power(1.0 / np.asarray(R, dtype=np.float64), sc.power)
        meta = {"mode": sc.mode, "power": sc.power}
    else:
        from ecoflowbench.landscapes.synthetic import gaussian_random_field

        s = gaussian_random_field(R.shape, sc.random_field_length_scale, rng).astype(np.float64)
        meta = {"mode": sc.mode, "length_scale": sc.random_field_length_scale}
    s = np.where(comp, s, 0.0)
    vals = s[comp]
    thr = float(np.quantile(vals, sc.quantile))
    src = np.where(comp & (s >= thr), s, 0.0)
    if src.sum() <= 0:  # degenerate (uniform R): take everything in the component
        src = np.where(comp, 1.0, 0.0)
    meta.update({"quantile": sc.quantile, "threshold": thr})
    return src, meta


def sample_advanced(R: np.ndarray, nodata: np.ndarray, cfg: SourceConfig, rng: np.random.Generator) -> SourceSample:
    """T3: source-strength raster (sum = normalize_total) and ground raster; grounds never overlap sources."""
    H, W = R.shape
    comp, labels = _largest_component_mask(R, nodata)
    src, smeta = suitability_field(R, nodata, comp, cfg.advanced.source, rng)
    gc = cfg.advanced.ground
    mode = str(rng.choice(gc.modes))
    ground = np.zeros((H, W), dtype=np.int8)
    ew = gc.edge_width_px
    if mode == "edge":
        side = str(rng.choice(["N", "S", "E", "W"]))
        sl = {"N": (slice(0, ew), slice(None)), "S": (slice(H - ew, H), slice(None)),
              "W": (slice(None), slice(0, ew)), "E": (slice(None), slice(W - ew, W))}[side]
        ground[sl] = 1
        gmeta = {"mode": mode, "side": side}
    elif mode == "all_edges":
        ground[:ew, :] = 1
        ground[H - ew:, :] = 1
        ground[:, :ew] = 1
        ground[:, W - ew:] = 1
        gmeta = {"mode": mode}
    else:
        n_p = int(rng.integers(gc.n_patches_range[0], gc.n_patches_range[1] + 1))
        rc = np.argwhere(comp)
        yy, xx = np.mgrid[0:H, 0:W]
        centres = []
        for _ in range(n_p):
            cy, cx = (int(v) for v in rc[rng.integers(len(rc))])
            ground[(yy - cy) ** 2 + (xx - cx) ** 2 <= gc.patch_radius_px**2] = 1
            centres.append((cy, cx))
        gmeta = {"mode": mode, "n_patches": n_p, "centres": centres, "radius_px": gc.patch_radius_px}
    ground[~comp] = 0
    src[ground > 0] = 0.0
    if ground.sum() == 0 or src.sum() <= 0:
        raise RuntimeError("degenerate advanced configuration (no ground or no source in the largest component)")
    nt = cfg.advanced.source.normalize_total
    if nt:
        src = src * (nt / src.sum())
    meta = {"source": smeta, "ground": gmeta, "n_source_pixels": int((src > 0).sum()),
            "n_ground_pixels": int(ground.sum()), "source_total": float(src.sum()),
            **_connectivity_meta(labels, (src > 0) | (ground > 0))}
    return SourceSample("advanced", None, [], src.astype(np.float32), ground, meta)


def sample_omniscape(R: np.ndarray, nodata: np.ndarray, cfg: SourceConfig, rng: np.random.Generator) -> SourceSample:
    """T4: Omniscape source-strength raster scaled to max = scale_max; threshold recorded."""
    comp, labels = _largest_component_mask(R, nodata)
    src, smeta = suitability_field(R, nodata, comp, cfg.omniscape.source, rng)
    sm = cfg.omniscape.source.scale_max or 1.0
    src = src * (sm / src.max())
    meta = {"source": smeta, "scale_max": sm, "source_threshold": cfg.omniscape.source_threshold,
            "n_source_pixels": int((src > cfg.omniscape.source_threshold).sum()),
            **_connectivity_meta(labels, src > 0)}
    return SourceSample("omniscape", None, [], src.astype(np.float32), None, meta)


def generate_all(R: np.ndarray, nodata: np.ndarray, cfg: SourceConfig, seed: int,
                 landcover: np.ndarray | None = None) -> dict[str, SourceSample]:
    """Every configuration for one landscape from one seed (sub-seeded per kind, so kinds are independent)."""
    ss = np.random.SeedSequence(int(seed))
    kids = ss.spawn(5)
    out = {
        "points": sample_points(R, nodata, cfg, np.random.default_rng(kids[0])),
        "wall_to_wall_NS": sample_wall_to_wall(R, nodata, cfg, "NS"),
        "wall_to_wall_EW": sample_wall_to_wall(R, nodata, cfg, "EW"),
        "advanced": sample_advanced(R, nodata, cfg, np.random.default_rng(kids[1])),
        "omniscape": sample_omniscape(R, nodata, cfg, np.random.default_rng(kids[2])),
    }
    if landcover is not None:
        reg = sample_regions(R, nodata, landcover, cfg, np.random.default_rng(kids[3]))
        if reg is not None:
            out["regions"] = reg
    for s in out.values():
        s.meta["seed"] = int(seed)
        s.meta["source_config"] = cfg.provenance()
    return out


def as_records(samples: dict[str, SourceSample]) -> list[dict]:
    return [{"kind": k, **dataclasses.asdict(s)["meta"]} for k, s in samples.items()]
