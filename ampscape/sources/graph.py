"""Exact reconstruction of the Circuitscape raster graph (for connectivity checks and residuals).

Mirrors ``Circuitscape.jl 5.17.1/src/raster/pairwise.jl::construct_graph`` with the AmpScape
solver conventions (``docs/task_specification.md`` §2):

* nodes = pixels that are not NoData (NoData = infinite resistance = no node)
* 8-neighbourhood (``connect_four_neighbors_only = False``)
* average-conductance rule (``connect_using_avg_resistances = False``):
  cardinal g_ij = (g_i + g_j) / 2, diagonal g_ij = (g_i + g_j) / (2·√2), with g = 1/R.

Because connectivity does not depend on the weights, the component labelling is exact for any
weighting; the weights are kept exact anyway so the same graph can be reused for the physics
metrics in Phase 9 and for cross-checking Circuitscape in Phase 5.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def node_index(nodata: np.ndarray) -> np.ndarray:
    """int64 (H, W) map pixel -> node id (0-based), -1 at NoData. Column-major like Circuitscape is
    irrelevant for connectivity; we use row-major ids."""
    valid = ~np.asarray(nodata, dtype=bool)
    idx = np.full(valid.shape, -1, dtype=np.int64)
    idx[valid] = np.arange(int(valid.sum()))
    return idx


def build_conductance_graph(resistance: np.ndarray, nodata: np.ndarray, four_neighbors: bool = False,
                            avg_resistances: bool = False) -> tuple[sp.csr_matrix, np.ndarray]:
    """Symmetric sparse conductance matrix G (n×n) and the node index map.

    ``avg_resistances=True`` reproduces Circuitscape's non-default rule (1/mean(R), diagonal /√2).
    """
    r = np.asarray(resistance, dtype=np.float64)
    nd = np.asarray(nodata, dtype=bool)
    if r.shape != nd.shape:
        raise ValueError("resistance and nodata shapes differ")
    if np.any(r[~nd] <= 0) or not np.all(np.isfinite(r[~nd])):
        raise ValueError("resistance must be finite and > 0 on valid pixels")
    idx = node_index(nd)
    g = np.where(nd, 0.0, 1.0 / np.where(nd, 1.0, r))
    rows, cols, vals = [], [], []

    H, W = idx.shape

    def add(di: int, dj: int, diagonal: bool) -> None:
        # pair pixel (i, j) with (i + di, j + dj); di >= 0, dj in {-1, 0, 1}
        j0, j1 = max(0, -dj), W - max(0, dj)
        a = idx[0:H - di, j0:j1]
        b = idx[di:H, j0 + dj:j1 + dj]
        ga = g[0:H - di, j0:j1]
        gb = g[di:H, j0 + dj:j1 + dj]
        m = (a >= 0) & (b >= 0)
        if avg_resistances:
            w = 1.0 / ((1.0 / ga[m] + 1.0 / gb[m]) / 2.0)
        else:
            w = (ga[m] + gb[m]) / 2.0
        if diagonal:
            w = w / math.sqrt(2.0)
        rows.append(a[m])
        cols.append(b[m])
        vals.append(w)

    add(0, 1, False)   # east
    add(1, 0, False)   # south
    if not four_neighbors:
        add(1, 1, True)    # south-east
        add(1, -1, True)   # south-west
    n = int((idx >= 0).sum())
    if rows:
        i = np.concatenate(rows)
        j = np.concatenate(cols)
        v = np.concatenate(vals)
        G = sp.coo_matrix((v, (i, j)), shape=(n, n)).tocsr()
        G = G + G.T
    else:
        G = sp.csr_matrix((n, n))
    return G, idx


def laplacian(G: sp.csr_matrix) -> sp.csr_matrix:
    d = np.asarray(G.sum(axis=1)).ravel()
    return (sp.diags(d) - G).tocsr()


def component_labels(resistance: np.ndarray, nodata: np.ndarray, four_neighbors: bool = False) -> np.ndarray:
    """int32 (H, W) connected-component label per pixel on the exact solver graph; -1 at NoData.

    Labels are ordered by component size (0 = largest).
    """
    G, idx = build_conductance_graph(resistance, nodata, four_neighbors=four_neighbors)
    n = G.shape[0]
    if n == 0:
        return np.full(np.asarray(nodata).shape, -1, dtype=np.int32)
    _, lab = connected_components(G, directed=False)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    out = np.full(idx.shape, -1, dtype=np.int32)
    out[idx >= 0] = remap[lab][idx[idx >= 0]]
    return out


def all_in_one_component(labels: np.ndarray, mask: np.ndarray) -> tuple[bool, int]:
    """True if every pixel where ``mask`` is set lies in the same (non-NoData) component."""
    vals = np.unique(labels[np.asarray(mask, dtype=bool)])
    ok = len(vals) == 1 and vals[0] >= 0
    return bool(ok), int(len(vals))
