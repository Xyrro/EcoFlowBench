"""Quality control for solved samples (brief §7.2 + owner rules).

Checks (each adds a flag string to ``qc_flags`` when it fails):
* ``not_converged``           solver reported an error / did not converge
* ``nonfinite_output``        NaN or Inf in any output array
* ``all_zero_output``         an output map is identically zero
* ``residual_high``           Kirchhoff residual ‖L v − b‖/‖b‖ computed in Julia on the exact graph from the
                              full-precision voltages exceeds ``residual_tol`` = 1e-6 (pairwise with K ≤ 4,
                              advanced). CHOLMOD gives 1e-12 typically; the double-precision floor for
                              contrast-10⁴ systems is ~1e-8 (observed 2.3e-8), CG+AMG gives 1e-6 to 3e-5.
                              (``residual_rel_f32``, the same quantity from the stored float32 maps, is
                              recorded for information only: float32 rounding dominates it at high contrast.)
* ``conservation_high``       for pairwise maps: node current at focal pixels ≠ injected current (1 per pair)
* ``isolated_focal``          any Reff entry is −1 (Circuitscape's code for disconnected focal nodes)
* ``omniscape_edge_artifact`` mean normalized current in the outer ring is > 3× the interior mean
* ``rmax_saturated``          > 50 % of valid resistance pixels at r_max (owner rule; excluded from train/val,
                              kept in the index)
* ``fallback_solver``         reference solver failed and the fallback (CG+AMG) produced the result (informational)

Samples with any flag other than ``fallback_solver`` / ``rmax_saturated`` are marked ``qc_pass = False``
and excluded from all splits; ``rmax_saturated`` samples are excluded from train/val only.
"""

from __future__ import annotations

import json

import numpy as np

from ecoflowbench.sources.graph import build_conductance_graph, laplacian

INFO_FLAGS = {"fallback_solver"}


def _f(x) -> float:
    """None / missing (JSON null from Julia's NaN sanitiser) -> NaN."""
    return float("nan") if x is None else float(x)
TRAINVAL_ONLY_FLAGS = {"rmax_saturated"}


def kirchhoff_residual(R: np.ndarray, nodata: np.ndarray, voltage: np.ndarray, injection: np.ndarray,
                       grounded: np.ndarray | None = None, supernode: np.ndarray | None = None,
                       graph: tuple | None = None) -> float:
    """‖L v − b‖₂ / ‖b‖₂ on the exact solver graph (mirror of the Julia implementation).

    ``injection`` (H,W) is the current injected per pixel, ``grounded`` marks pixels held at 0 V
    (rows dropped), ``supernode`` marks pixels of one short-circuited focal region whose rows are
    replaced by the single net-current equation Σ_region (L v) = Σ_region b.
    """
    if graph is None:
        G, idx = build_conductance_graph(R, nodata)
        L = laplacian(G)
    else:
        L, idx = graph
    valid = idx >= 0
    v = voltage[valid].astype(np.float64)
    b = injection[valid].astype(np.float64)
    r = L @ v - b
    gnd = grounded[valid] if grounded is not None else np.zeros_like(v, dtype=bool)
    sn = supernode[valid] if supernode is not None else np.zeros_like(v, dtype=bool)
    free = ~gnd & ~sn
    rr = float(np.sum(r[free] ** 2))
    if sn.any():
        rr += float(np.sum(r[sn])) ** 2
    nb = np.linalg.norm(b[~gnd])
    return float(np.sqrt(rr) / nb) if nb > 0 else float("nan")


def qc_pairwise(R, nodata, focal, out: dict, r_max: float | None, residual_tol=1e-6, rmax_frac=0.5) -> dict:
    flags = []
    stats = json.loads(out["stats"])
    if not stats["converged"]:
        flags.append("not_converged")
    if stats.get("fallback_used"):
        flags.append("fallback_solver")
    cum = out["cum_current"]
    reff = out["reff"]
    if not np.isfinite(cum).all() or not np.isfinite(reff).all():
        flags.append("nonfinite_output")
    if np.all(cum == 0):
        flags.append("all_zero_output")
    if np.any(reff == -1):
        flags.append("isolated_focal")
    # authoritative residual: computed in Julia from the full-precision voltages (stats.solver_params)
    resid = _f(stats.get("solver_params", {}).get("residual_rel"))
    resid_f32 = float("nan")
    cons = float("nan")
    if stats["converged"] and np.isfinite(resid) and resid > residual_tol:
        flags.append("residual_high")
    if "voltage" in out and stats["converged"] and np.isfinite(out["voltage"]).all():
        pair_index = out["pair_index"]
        G, idx = build_conductance_graph(R, nodata)      # build the exact graph once per sample
        graph = (laplacian(G), idx)
        worst = 0.0
        worst_c = 0.0
        for p in range(pair_index.shape[0]):
            i, j = int(pair_index[p, 0]), int(pair_index[p, 1])
            # Circuitscape convention: node i grounded, unit current injected at node j
            gnd, src = focal == i, focal == j
            inj = np.zeros(R.shape, np.float64)
            inj[src] = 1.0 / src.sum()
            worst = max(worst, kirchhoff_residual(R, nodata, out["voltage"][p], inj, grounded=gnd,
                                                  supernode=src if src.sum() > 1 else None, graph=graph))
            c = out["pairwise_current"][p]
            worst_c = max(worst_c, abs(c[src].max() - 1.0), abs(c[gnd].max() - 1.0))
        resid_f32, cons = worst, worst_c   # resid_f32 is informational only (float32 rounding dominates it)
        if cons > 1e-6:
            flags.append("conservation_high")
    if r_max is not None:
        frac = float((R[~nodata] >= r_max - 1e-6).mean())
        if frac > rmax_frac:
            flags.append("rmax_saturated")
    return {"qc_flags": flags, "residual_rel": resid, "residual_rel_f32": resid_f32, "conservation_err": cons,
            "solve_time_s": _f(stats["wall_s"]), "maxrss_mb": _f(stats["maxrss_mb"]), "solver": stats["solver"],
            "converged": stats["converged"]}


def qc_advanced(R, nodata, source, ground, out: dict, residual_tol=1e-6) -> dict:
    flags = []
    stats = json.loads(out["stats"])
    if not stats["converged"]:
        flags.append("not_converged")
    if stats.get("fallback_used"):
        flags.append("fallback_solver")
    cur, volt = out["current"], out["voltage"]
    if not (np.isfinite(cur).all() and np.isfinite(volt).all()):
        flags.append("nonfinite_output")
    if np.all(cur == 0):
        flags.append("all_zero_output")
    resid = float(stats.get("solver_params", {}).get("residual_rel", float("nan")))
    resid_f32 = float("nan")
    if stats["converged"] and np.isfinite(resid) and resid > residual_tol:
        flags.append("residual_high")
    if stats["converged"] and np.isfinite(volt).all():
        resid_f32 = kirchhoff_residual(R, nodata, volt, source.astype(np.float64), grounded=ground > 0)  # informational
    return {"qc_flags": flags, "residual_rel": resid, "residual_rel_f32": resid_f32, "conservation_err": float("nan"),
            "solve_time_s": _f(stats["wall_s"]), "maxrss_mb": _f(stats["maxrss_mb"]), "solver": stats["solver"],
            "converged": stats["converged"]}


def qc_omniscape(R, nodata, out: dict, ring: int = 2) -> dict:
    flags = []
    stats = json.loads(out["stats"])
    if not stats["converged"]:
        flags.append("not_converged")
    arrays = [out["cum_current"], out["flow_potential"], out["normalized"]]
    if not all(np.isfinite(a).all() for a in arrays):
        flags.append("nonfinite_output")
    if np.all(arrays[0] == 0):
        flags.append("all_zero_output")
    n = out["normalized"]
    valid = ~nodata
    edge = np.zeros_like(valid)
    edge[:ring, :] = edge[-ring:, :] = edge[:, :ring] = edge[:, -ring:] = True
    e_mean = n[valid & edge].mean() if (valid & edge).any() else 0.0
    i_mean = n[valid & ~edge].mean() if (valid & ~edge).any() else 0.0
    edge_ratio = float(e_mean / i_mean) if i_mean > 0 else float("nan")
    if np.isfinite(edge_ratio) and edge_ratio > 3.0:
        flags.append("omniscape_edge_artifact")
    return {"qc_flags": flags, "residual_rel": float("nan"), "conservation_err": float("nan"), "edge_ratio": edge_ratio,
            "solve_time_s": _f(stats["wall_s"]), "maxrss_mb": _f(stats["maxrss_mb"]), "solver": stats["solver"],
            "converged": stats["converged"]}


def qc_pass(flags: list[str]) -> tuple[bool, bool]:
    """(pass for test/index, pass for train/val)."""
    hard = [f for f in flags if f not in INFO_FLAGS and f not in TRAINVAL_ONLY_FLAGS]
    return (len(hard) == 0, len(hard) == 0 and not any(f in TRAINVAL_ONLY_FLAGS for f in flags))
