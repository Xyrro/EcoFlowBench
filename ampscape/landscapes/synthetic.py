"""Synthetic landscape generators (Phase 2, brief §4.1).

Every generator returns a *cost field* ``f`` with shape ``(H, W)``, dtype float32, values in
``[0, 1]`` (0 = most permeable, 1 = most resistant). Fields are mapped to resistance rasters in
``[1, contrast]`` by :func:`field_to_resistance`. All randomness comes from a
``numpy.random.Generator`` passed explicitly, so ``(generator, params, seed)`` reproduces a
landscape bitwise on the same platform. NLMpy (Etherington et al. 2015) uses NumPy's *global*
RNG internally, so its calls are wrapped in :func:`seeded_global_rng`.

Families implemented
--------------------
* Gaussian random fields (correlation length, anisotropy, orientation)
* Midpoint-displacement fractals (NLMpy ``mpd``, roughness H)
* NLMpy models: random cluster, planar gradient, edge gradient, distance gradient, mosaic
  (random element nearest-neighbour)
* Overlays: linear barriers (roads/rivers) with density, width, orientation and gaps;
  categorical patch mosaics
* NoData blobs (masked regions) with a single-connected-component guarantee
* Contrast control: dynamic range in {10, 100, 1000, 10000}

The documented sampling priors live in :data:`DEFAULT_PRIOR` and are consumed by
:func:`sample_landscape`, which records every drawn parameter in ``params``.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

Shape = tuple[int, int]

CONTRAST_LEVELS: tuple[int, ...] = (10, 100, 1000, 10_000)
GRF_LENGTH_SCALES: tuple[int, ...] = (2, 8, 32, 128)
FRACTAL_ROUGHNESS: tuple[float, ...] = (0.2, 0.5, 0.8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def seeded_global_rng(seed: int) -> Iterator[None]:
    """Temporarily seed NumPy's legacy global RNG (used by NLMpy) and restore it afterwards."""
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


def rescale01(a: np.ndarray) -> np.ndarray:
    """Min-max rescale to [0, 1] (constant arrays map to 0)."""
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if not np.isfinite(lo) or hi - lo <= 0:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _check_shape(shape: Shape) -> Shape:
    h, w = int(shape[0]), int(shape[1])
    if h < 4 or w < 4:
        raise ValueError(f"shape must be at least 4x4, got {shape}")
    return h, w


def _sub_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**32 - 1))


# ---------------------------------------------------------------------------
# Continuous fields
# ---------------------------------------------------------------------------
def gaussian_random_field(
    shape: Shape,
    length_scale: float,
    rng: np.random.Generator,
    anisotropy: float = 1.0,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Gaussian random field with a Gaussian covariance kernel, via FFT spectral filtering.

    Parameters
    ----------
    length_scale : correlation length in pixels along the principal axis.
    anisotropy   : ratio of correlation lengths (principal / secondary), >= 1.
    angle_deg    : orientation of the principal axis, degrees counter-clockwise from +x (columns).
    """
    h, w = _check_shape(shape)
    if length_scale <= 0 or anisotropy < 1:
        raise ValueError("length_scale must be > 0 and anisotropy >= 1")
    noise = rng.standard_normal((h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    th = math.radians(angle_deg)
    # rotate frequency coordinates into the principal frame
    fu = fx * math.cos(th) + fy * math.sin(th)
    fv = -fx * math.sin(th) + fy * math.cos(th)
    lu, lv = float(length_scale), float(length_scale) / float(anisotropy)
    # Gaussian kernel in Fourier space: exp(-2 pi^2 (l_u^2 f_u^2 + l_v^2 f_v^2))
    kernel = np.exp(-2.0 * math.pi**2 * (lu**2 * fu**2 + lv**2 * fv**2))
    spec = np.fft.fft2(noise) * np.sqrt(kernel)
    out = np.real(np.fft.ifft2(spec))
    return rescale01(out)


def midpoint_displacement(shape: Shape, roughness: float, rng: np.random.Generator) -> np.ndarray:
    """Midpoint-displacement fractal (NLMpy ``mpd``); ``roughness`` in (0, 1), higher = smoother."""
    h, w = _check_shape(shape)
    if not 0 < roughness < 1:
        raise ValueError("roughness must be in (0, 1)")
    from nlmpy import nlmpy as nlm

    with seeded_global_rng(_sub_seed(rng)):
        return rescale01(nlm.mpd(h, w, float(roughness)))


def nlm_random_cluster(
    shape: Shape, p: float, rng: np.random.Generator, neighbourhood: str = "8-neighbourhood"
) -> np.ndarray:
    """NLMpy random-cluster nearest-neighbour model; ``p`` = cluster proportion in (0, 0.6]."""
    h, w = _check_shape(shape)
    if not 0 < p <= 0.6:
        raise ValueError("p must be in (0, 0.6]")
    from nlmpy import nlmpy as nlm

    with seeded_global_rng(_sub_seed(rng)):
        return rescale01(nlm.randomClusterNN(h, w, float(p), neighbourhood))


def nlm_planar_gradient(shape: Shape, direction_deg: float, rng: np.random.Generator) -> np.ndarray:
    """NLMpy planar gradient in a given direction (degrees)."""
    h, w = _check_shape(shape)
    from nlmpy import nlmpy as nlm

    with seeded_global_rng(_sub_seed(rng)):
        return rescale01(nlm.planarGradient(h, w, float(direction_deg)))


def nlm_edge_gradient(shape: Shape, direction_deg: float, rng: np.random.Generator) -> np.ndarray:
    """NLMpy edge gradient (high at one edge, decaying towards the opposite one)."""
    h, w = _check_shape(shape)
    from nlmpy import nlmpy as nlm

    with seeded_global_rng(_sub_seed(rng)):
        return rescale01(nlm.edgeGradient(h, w, float(direction_deg)))


def nlm_distance_gradient(shape: Shape, n_sources: int, rng: np.random.Generator) -> np.ndarray:
    """NLMpy distance gradient from ``n_sources`` random source pixels (0 at sources)."""
    h, w = _check_shape(shape)
    if n_sources < 1:
        raise ValueError("n_sources must be >= 1")
    from nlmpy import nlmpy as nlm

    src = np.zeros((h, w), dtype=np.int64)
    rows = rng.integers(0, h, n_sources)
    cols = rng.integers(0, w, n_sources)
    src[rows, cols] = 1
    with seeded_global_rng(_sub_seed(rng)):
        return rescale01(nlm.distanceGradient(src))


def nlm_mosaic(shape: Shape, n_elements: int, rng: np.random.Generator) -> np.ndarray:
    """NLMpy random-element nearest-neighbour mosaic (Voronoi-like patches with random values)."""
    h, w = _check_shape(shape)
    if n_elements < 2:
        raise ValueError("n_elements must be >= 2")
    from nlmpy import nlmpy as nlm

    with seeded_global_rng(_sub_seed(rng)):
        return rescale01(nlm.randomElementNN(h, w, int(n_elements)))


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------
def linear_barriers(
    shape: Shape,
    n_lines: int,
    width_px: float,
    rng: np.random.Generator,
    orientation_deg: float | None = None,
    orientation_jitter_deg: float = 20.0,
    gap_fraction: float = 0.0,
    gap_length_px: float = 4.0,
) -> np.ndarray:
    """Boolean mask of linear barriers (roads / rivers) crossing the raster.

    Each line has a random anchor point, an orientation (uniform if ``orientation_deg`` is
    None, otherwise jittered around it), width ``width_px`` and, along its length, gaps that
    cover a fraction ``gap_fraction`` of the line in segments of ``gap_length_px``.
    """
    h, w = _check_shape(shape)
    if n_lines < 0 or width_px <= 0 or not 0 <= gap_fraction < 1:
        raise ValueError("invalid barrier parameters")
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(int(n_lines)):
        ay, ax = rng.uniform(0, h), rng.uniform(0, w)
        if orientation_deg is None:
            th = rng.uniform(0, math.pi)
        else:
            th = math.radians(orientation_deg + rng.normal(0, orientation_jitter_deg))
        dx, dy = math.cos(th), math.sin(th)
        # perpendicular distance and along-line coordinate
        perp = np.abs(-(xx - ax) * dy + (yy - ay) * dx)
        along = (xx - ax) * dx + (yy - ay) * dy
        line = perp <= width_px / 2.0
        if gap_fraction > 0:
            # periodic gaps: within each period of length L, the last gap_fraction*L is open
            period = gap_length_px / gap_fraction
            phase = rng.uniform(0, period)
            open_ = np.mod(along + phase, period) >= period * (1.0 - gap_fraction)
            line &= ~open_
        mask |= line
    return mask


def patch_mosaic(
    fld: np.ndarray, n_classes: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Quantise a continuous field into ``n_classes`` patches with random class resistances.

    Returns ``(field, class_map, class_values)`` where ``class_values[k]`` is the cost value in
    [0, 1] assigned to class ``k`` (sorted so classes are ordinal in the source field but not in
    resistance, mimicking land-cover tables).
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    qs = np.quantile(fld, np.linspace(0, 1, n_classes + 1)[1:-1])
    class_map = np.digitize(fld, qs).astype(np.int16)
    values = rng.uniform(0, 1, n_classes)
    values[rng.integers(0, n_classes)] = 0.0  # ensure a fully permeable class exists
    values[rng.integers(0, n_classes)] = 1.0  # and a maximally resistant one
    out = values[class_map].astype(np.float32)
    return out, class_map, [float(v) for v in values]


def random_nodata(
    shape: Shape, fraction: float, length_scale: float, rng: np.random.Generator
) -> np.ndarray:
    """NoData blobs covering about ``fraction`` of the raster (thresholded GRF).

    The returned mask is True at NoData pixels. Valid pixels are reduced to their largest
    8-connected component so the resistance graph is connected (isolated islands become NoData).
    """
    h, w = _check_shape(shape)
    if not 0 <= fraction < 0.5:
        raise ValueError("fraction must be in [0, 0.5)")
    if fraction == 0:
        return np.zeros((h, w), dtype=bool)
    g = gaussian_random_field((h, w), length_scale, rng)
    thr = np.quantile(g, fraction)
    nodata = g < thr
    return ensure_single_component(nodata)


def ensure_single_component(nodata: np.ndarray) -> np.ndarray:
    """Set every valid pixel outside the largest 8-connected valid component to NoData."""
    valid = ~nodata
    labels, n = ndimage.label(valid, structure=np.ones((3, 3), dtype=int))
    if n <= 1:
        return nodata.copy()
    sizes = ndimage.sum(valid, labels, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    return labels != keep


# ---------------------------------------------------------------------------
# Field -> resistance
# ---------------------------------------------------------------------------
def field_to_resistance(fld: np.ndarray, contrast: float, mapping: str = "log") -> np.ndarray:
    """Map a [0, 1] cost field to resistance in [1, contrast].

    ``mapping="log"``: R = contrast ** f   (log-uniform spread, the default)
    ``mapping="linear"``: R = 1 + (contrast - 1) * f
    """
    if contrast < 1:
        raise ValueError("contrast must be >= 1")
    f = np.clip(np.asarray(fld, dtype=np.float64), 0.0, 1.0)
    if mapping == "log":
        r = np.power(float(contrast), f)
    elif mapping == "linear":
        r = 1.0 + (float(contrast) - 1.0) * f
    else:
        raise ValueError(f"unknown mapping {mapping!r}")
    return np.clip(r, 1.0, float(contrast)).astype(np.float32)


# ---------------------------------------------------------------------------
# Sampling from documented priors
# ---------------------------------------------------------------------------
DEFAULT_PRIOR: dict = {
    # generator name -> sampling weight
    "generator_weights": {
        "grf": 0.30,
        "fractal": 0.20,
        "random_cluster": 0.15,
        "planar_gradient": 0.05,
        "edge_gradient": 0.05,
        "distance_gradient": 0.10,
        "mosaic": 0.15,
    },
    "grf": {"length_scale": GRF_LENGTH_SCALES, "anisotropy": (1.0, 4.0), "angle_deg": (0.0, 180.0)},
    "fractal": {"roughness": FRACTAL_ROUGHNESS},
    "random_cluster": {"p": (0.3, 0.58)},
    "planar_gradient": {"direction_deg": (0.0, 360.0)},
    "edge_gradient": {"direction_deg": (0.0, 360.0)},
    "distance_gradient": {"n_sources": (1, 6)},
    "mosaic": {"n_elements": (5, 60)},
    # overlays
    "p_patch_mosaic": 0.3,
    "patch_classes": (3, 8),
    "p_barriers": 0.5,
    "barriers": {
        "n_lines_per_128px": (1, 8),
        "width_px": (1.0, 3.0),
        "p_oriented": 0.5,
        "gap_fraction": (0.0, 0.3),
        "gap_length_px": (2.0, 8.0),
        "barrier_cost": (0.7, 1.0),   # cost value written on barrier pixels (before contrast)
    },
    "p_nodata": 0.3,
    "nodata": {"fraction": (0.02, 0.25), "length_scale": (4.0, 32.0)},
    "contrast": CONTRAST_LEVELS,
    "mapping": "log",
}


@dataclass
class SyntheticLandscape:
    """A generated landscape plus everything needed to regenerate it."""

    resistance: np.ndarray            # float32 (H, W) in [1, contrast]; 1.0 at NoData
    nodata_mask: np.ndarray           # bool (H, W)
    cost_field: np.ndarray            # float32 (H, W) in [0, 1]
    generator: str
    params: dict = field(default_factory=dict)
    seed: int = 0

    @property
    def contrast(self) -> float:
        return float(self.params["contrast"])

    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _ui(rng: np.random.Generator, lo_hi: tuple[int, int]) -> int:
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def generate_field(name: str, shape: Shape, params: dict, rng: np.random.Generator) -> np.ndarray:
    """Dispatch a base generator by name with explicit parameters."""
    if name == "grf":
        return gaussian_random_field(shape, params["length_scale"], rng,
                                     params.get("anisotropy", 1.0), params.get("angle_deg", 0.0))
    if name == "fractal":
        return midpoint_displacement(shape, params["roughness"], rng)
    if name == "random_cluster":
        return nlm_random_cluster(shape, params["p"], rng, params.get("neighbourhood", "8-neighbourhood"))
    if name == "planar_gradient":
        return nlm_planar_gradient(shape, params["direction_deg"], rng)
    if name == "edge_gradient":
        return nlm_edge_gradient(shape, params["direction_deg"], rng)
    if name == "distance_gradient":
        return nlm_distance_gradient(shape, params["n_sources"], rng)
    if name == "mosaic":
        return nlm_mosaic(shape, params["n_elements"], rng)
    raise ValueError(f"unknown generator {name!r}")


def sample_landscape(seed: int, shape: Shape, prior: dict | None = None) -> SyntheticLandscape:
    """Draw one synthetic landscape from the documented prior. Deterministic in ``seed``."""
    prior = DEFAULT_PRIOR if prior is None else prior
    h, w = _check_shape(shape)
    rng = np.random.default_rng(int(seed))

    names = list(prior["generator_weights"])
    weights = np.array([prior["generator_weights"][n] for n in names], dtype=float)
    name = str(rng.choice(names, p=weights / weights.sum()))
    gp = prior[name]
    params: dict = {"shape": [h, w], "generator": name}
    if name == "grf":
        base = {
            "length_scale": int(rng.choice(gp["length_scale"])),
            "anisotropy": _u(rng, gp["anisotropy"]),
            "angle_deg": _u(rng, gp["angle_deg"]),
        }
    elif name == "fractal":
        base = {"roughness": float(rng.choice(gp["roughness"]))}
    elif name == "random_cluster":
        base = {"p": _u(rng, gp["p"]), "neighbourhood": "8-neighbourhood"}
    elif name in ("planar_gradient", "edge_gradient"):
        base = {"direction_deg": _u(rng, gp["direction_deg"])}
    elif name == "distance_gradient":
        base = {"n_sources": _ui(rng, gp["n_sources"])}
    elif name == "mosaic":
        base = {"n_elements": _ui(rng, gp["n_elements"])}
    else:  # pragma: no cover
        raise ValueError(name)
    params["base"] = base
    fld = generate_field(name, (h, w), base, rng)

    # categorical patch mosaic overlay
    if rng.uniform() < prior["p_patch_mosaic"]:
        k = _ui(rng, prior["patch_classes"])
        fld, _, values = patch_mosaic(fld, k, rng)
        params["patch_mosaic"] = {"n_classes": k, "class_values": values}

    # linear barriers overlay
    if rng.uniform() < prior["p_barriers"]:
        bp = prior["barriers"]
        scale = (h * w) / (128 * 128)
        n_lines = max(1, int(round(_ui(rng, bp["n_lines_per_128px"]) * math.sqrt(scale))))
        oriented = rng.uniform() < bp["p_oriented"]
        bparams = {
            "n_lines": n_lines,
            "width_px": _u(rng, bp["width_px"]),
            "orientation_deg": _u(rng, (0.0, 180.0)) if oriented else None,
            "gap_fraction": _u(rng, bp["gap_fraction"]),
            "gap_length_px": _u(rng, bp["gap_length_px"]),
            "barrier_cost": _u(rng, bp["barrier_cost"]),
        }
        bmask = linear_barriers((h, w), bparams["n_lines"], bparams["width_px"], rng,
                                bparams["orientation_deg"], gap_fraction=bparams["gap_fraction"],
                                gap_length_px=bparams["gap_length_px"])
        fld = fld.copy()
        fld[bmask] = np.maximum(fld[bmask], bparams["barrier_cost"])
        params["barriers"] = bparams

    # NoData
    if rng.uniform() < prior["p_nodata"]:
        npz = prior["nodata"]
        nparams = {"fraction": _u(rng, npz["fraction"]), "length_scale": _u(rng, npz["length_scale"])}
        nodata = random_nodata((h, w), nparams["fraction"], nparams["length_scale"], rng)
        params["nodata"] = nparams
    else:
        nodata = np.zeros((h, w), dtype=bool)

    contrast = int(rng.choice(prior["contrast"]))
    params["contrast"] = contrast
    params["mapping"] = prior["mapping"]
    resistance = field_to_resistance(fld, contrast, prior["mapping"])
    resistance[nodata] = 1.0
    return SyntheticLandscape(resistance, nodata, fld.astype(np.float32), name, params, int(seed))


def regenerate(params: dict, seed: int) -> SyntheticLandscape:
    """Regenerate a landscape from stored ``params`` + ``seed`` (round-trip of :func:`sample_landscape`).

    The prior is not needed: ``sample_landscape`` is deterministic in ``seed`` given the prior,
    and ``params`` is only used to check the result.
    """
    ls = sample_landscape(seed, tuple(params["shape"]))
    if ls.params != params:
        raise RuntimeError("regenerated params differ from stored params; prior or code changed")
    return ls
