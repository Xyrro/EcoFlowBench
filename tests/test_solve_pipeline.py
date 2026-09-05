"""Offline tests for the Phase 5 Python layer: manifest, prepare (HDF5 layout), QC logic, finalize on a fake shard."""

from __future__ import annotations

import json
import pathlib

import h5py
import numpy as np
import pytest

from ecoflowbench.solve.manifest import (
    DEFAULT_CONFIGS,
    from_frame,
    plan_synthetic,
    sample_uuid,
    to_frame,
)
from ecoflowbench.solve.prepare import prepare_shard
from ecoflowbench.solve.qc import qc_advanced, qc_omniscape, qc_pairwise, qc_pass
from ecoflowbench.sources import SourceConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = SourceConfig.from_yaml(ROOT / "configs" / "tasks" / "sources_default.yaml")


def test_manifest_deterministic_ids_and_shards():
    a = plan_synthetic("ds", 7, "S", 100, shard_size=3)
    b = plan_synthetic("ds", 7, "S", 100, shard_size=3)
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    assert [s.shard for s in a] == [0, 0, 0, 1, 1, 1, 2]
    assert sample_uuid("ds", "synthetic", "S:100") == a[0].sample_id
    assert sample_uuid("other", "synthetic", "S:100") != a[0].sample_id
    df = to_frame(a)
    back = from_frame(df)
    assert back[0].config_list() == list(DEFAULT_CONFIGS) and back[0].generator == a[0].generator


def test_prepare_layout(tmp_path):
    specs = plan_synthetic("ds", 2, "S", 300, shard_size=2)
    specs[0].extra = json.dumps({"k_override": 4})
    n = prepare_shard(specs, str(tmp_path / "in.h5"), CFG)
    assert n == 2
    with h5py.File(tmp_path / "in.h5") as f:
        for s in specs:
            g = f["samples"][s.sample_id]
            assert g["inputs"]["resistance"].shape == (128, 128) and g["inputs"]["resistance"].dtype == np.float32
            assert g["inputs"]["nodata_mask"].dtype == np.uint8
            assert set(g["configs"]) == set(DEFAULT_CONFIGS)
            assert g["configs"]["points"].attrs["kind"] == "points"
            assert g["configs"]["advanced"]["ground"].dtype == np.int8
            assert g.attrs["omni_radius"] == 16 and g.attrs["omni_block_size"] == 3
            meta = json.loads(g.attrs["meta"])
            assert meta["source_config"]["config_id"] == "sources_default_v1" and meta["generator"]
        k = len(json.loads(f["samples"][specs[0].sample_id].attrs["meta"])["configs"]["points"]["focal_table"])
        assert k == 4
    assert prepare_shard(specs, str(tmp_path / "in.h5"), CFG) == 0     # idempotent


def _stats(**kw):
    d = {"solver": "cholmod", "converged": True, "wall_s": 0.1, "maxrss_mb": 100.0, "fallback_used": False,
         "solver_params": {"residual_rel": 1e-12}}
    d.update(kw)
    return json.dumps(d)


def test_qc_pairwise_flags():
    H = W = 16
    R = np.ones((H, W), np.float32)
    nd = np.zeros((H, W), bool)
    focal = np.zeros((H, W), np.int32)
    focal[2, 2] = 1
    focal[12, 12] = 2
    out = {"stats": _stats(), "cum_current": np.ones((H, W), np.float32), "reff": np.array([[0, 1.0], [1.0, 0]]),
           "labels": np.array([1, 2]), "pair_index": np.array([[1, 2]])}
    q = qc_pairwise(R, nd, focal, out, r_max=1000.0)
    assert q["qc_flags"] == [] and qc_pass(q["qc_flags"]) == (True, True)
    q = qc_pairwise(R, nd, focal, dict(out, stats=_stats(converged=False, solver_params={"residual_rel": None})), 1000.0)
    assert "not_converged" in q["qc_flags"] and qc_pass(q["qc_flags"]) == (False, False)
    q = qc_pairwise(R, nd, focal, dict(out, reff=np.array([[0, -1.0], [-1.0, 0]])), 1000.0)
    assert "isolated_focal" in q["qc_flags"]
    q = qc_pairwise(R, nd, focal, dict(out, stats=_stats(solver_params={"residual_rel": 1e-5})), 1000.0)
    assert "residual_high" in q["qc_flags"]
    q = qc_pairwise(R, nd, focal, dict(out, stats=_stats(fallback_used=True)), 1000.0)
    assert q["qc_flags"] == ["fallback_solver"] and qc_pass(q["qc_flags"]) == (True, True)
    R2 = np.full((H, W), 1000.0, np.float32)
    R2[:4] = 1.0
    q = qc_pairwise(R2, nd, focal, out, r_max=1000.0)
    assert q["qc_flags"] == ["rmax_saturated"] and qc_pass(q["qc_flags"]) == (True, False)
    q = qc_pairwise(R, nd, focal, dict(out, cum_current=np.zeros((H, W), np.float32)), 1000.0)
    assert "all_zero_output" in q["qc_flags"]


def test_qc_advanced_and_omniscape():
    H = W = 8
    R = np.ones((H, W), np.float32)
    nd = np.zeros((H, W), bool)
    src = np.zeros((H, W), np.float32)
    src[1, 1] = 1.0
    gnd = np.zeros((H, W), np.int8)
    gnd[6, 6] = 1
    out = {"stats": _stats(), "current": np.ones((H, W), np.float32), "voltage": np.full((H, W), np.nan, np.float32)}
    q = qc_advanced(R, nd, src, gnd, out)
    assert "nonfinite_output" in q["qc_flags"]
    n = np.ones((H, W), np.float32)
    n[:2, :] = 10.0          # ring of width 2: (16*10 + 32*1)/48 = 4 > 3x the interior mean
    o = {"stats": _stats(), "cum_current": n, "flow_potential": n, "normalized": n}
    q = qc_omniscape(R, nd, o)
    assert "omniscape_edge_artifact" in q["qc_flags"] and q["edge_ratio"] > 3


@pytest.mark.skipif(not (ROOT / "data" / "devtest" / "shard-00000.outputs.h5").exists(), reason="dev outputs not present")
def test_finalize_dev_shard(tmp_path):
    from ecoflowbench.solve.finalize import finalize_shard

    idx = finalize_shard(str(ROOT / "data/devtest/shard-00000.inputs.h5"), str(ROOT / "data/devtest/shard-00000.outputs.h5"),
                         str(tmp_path / "final.h5"), "test", {"solver": "cholmod"})
    assert idx.qc_pass.all() and set(idx.kind) == {"points", "wall_to_wall", "advanced", "omniscape"}
    with h5py.File(tmp_path / "final.h5") as f:
        sid = list(f)[0]
        assert "inputs" in f[sid] and "configs" in f[sid]
        meta = json.loads(f[sid].attrs["meta"])
        assert meta["solver_versions"]["circuitscape"] == "5.17.1" and meta["solver_preset"]["solver"] == "cholmod"
