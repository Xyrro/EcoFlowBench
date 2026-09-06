"""Target transforms shared by the loader, the metrics and the baselines.

Current-density maps span many orders of magnitude and contain exact zeros (NoData, unreachable
pixels). Following the external review (2026-09-06) the log transform is

    log10_eps(C) = log10(C + EPS * max(C))        with EPS = 1e-6 (per map)

instead of log1p(C): log1p is scale-dependent (it behaves like the identity below 1 A and like a log
above, so the same map at a different injection scale gets a different error weighting), whereas the
ε-shifted log is scale-invariant and bounds the dynamic range to 6 decades below the map maximum.
EPS is documented here and recorded in ``stats/norm_stats.json``. For maps that are identically
zero the transform returns zeros.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def log10_eps(c: np.ndarray, eps: float = EPS, axis: tuple | None = None) -> np.ndarray:
    """log10(C + eps * max(C)); ``axis`` selects the max per map when C is batched (default: whole array)."""
    c = np.maximum(np.asarray(c, dtype=np.float64), 0.0)
    m = c.max() if axis is None else c.max(axis=axis, keepdims=True)
    if np.ndim(m) == 0 and m <= 0:
        return np.zeros_like(c, dtype=np.float32)
    return np.log10(c + eps * np.maximum(m, 1e-300)).astype(np.float32)


def inverse_log10_eps(y: np.ndarray, cmax: float, eps: float = EPS) -> np.ndarray:
    return (np.power(10.0, np.asarray(y, dtype=np.float64)) - eps * cmax).astype(np.float32)
