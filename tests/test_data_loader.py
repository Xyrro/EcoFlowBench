"""Loader, HF layout export, subsets — on the mini build (skipped if absent)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MINI = ROOT / "data" / "builds" / "mini"
pytestmark = pytest.mark.skipif(not (MINI / "index.parquet").exists(), reason="mini build not present")


def test_dataset_contracts():
    from ampscape.data import AmpScapeDataset, log1p_targets

    for task, keys in {"T1": ["cum_current", "reff", "focal_onehot"], "T2": ["reff", "reff_mask", "focal_table"],
                       "T1W": ["pairwise_current", "voltage", "pair_index"], "T3": ["current", "voltage", "ground"],
                       "T4": ["cum_current", "flow_potential", "normalized"]}.items():
        ds = AmpScapeDataset(task, split="train", tier="S", root=MINI)
        assert len(ds) > 0
        d = ds[0]
        for k in keys:
            assert k in d, (task, k)
        assert d["resistance"].shape == (1, 128, 128) and d["resistance"].dtype == np.float32
        assert d["log_resistance"].min() >= 0 and set(np.unique(d["nodata"])) <= {0.0, 1.0}
        assert np.allclose(np.exp(d["log_resistance"]), d["resistance"], rtol=1e-5)
        assert log1p_targets(np.array([-1.0, 0.0, 1.0])).tolist() == [0.0, 0.0, np.log1p(1.0)]
    t2 = AmpScapeDataset("T2", split="train", tier="S", root=MINI)[0]
    K = int(t2["reff_mask"][:, 0].sum())
    assert K == t2["focal_table"].shape[0] and np.allclose(t2["reff"][:K, :K], t2["reff"][:K, :K].T)
    assert np.all(np.diag(t2["reff"][:K, :K]) == 0)


def test_split_and_qc_filters_and_torch():
    from ampscape.data import AmpScapeDataset

    tr = AmpScapeDataset("T1", split="train", tier="S", root=MINI)
    te = AmpScapeDataset("T1", split=["test_id", "test_ood", "ood_region"], tier="S", root=MINI)
    assert set(tr.index.sample_id).isdisjoint(set(te.index.sample_id))
    assert tr.index.qc_trainval.all()
    ood = AmpScapeDataset("T1", split=None, tier="S", root=MINI, ood="test_ood_contrast")
    assert len(ood) > 0 and (ood.index.contrast >= 10000).all()
    torch_ds = tr.torch()
    b = torch_ds[0]
    assert tuple(b["cum_current"].shape) == (1, 128, 128) and str(b["cum_current"].dtype) == "torch.float32"


def test_normalization_train_only(tmp_path):
    from ampscape.data import AmpScapeDataset, compute_norm_stats

    st = compute_norm_stats(MINI, "S", out=tmp_path / "norm.json")
    assert st["n_train_items"] == len(AmpScapeDataset("T1", split="train", tier="S", root=MINI))
    assert 0 < st["log_resistance"]["std"] < 5


def test_hf_layout_export_and_subsets(tmp_path):
    from ampscape.data import AmpScapeDataset
    from ampscape.data.subsets import assign_subsets
    from ampscape.io.hf_layout import TASK_GROUPS, export_build, split_shard_by_task_group

    idx = pd.read_parquet(MINI / "index.parquet")
    subs = assign_subsets(idx, {"mini": 60, "core": 120, "full": None})
    assert subs["mini"] <= subs["core"] <= subs["full"] and len(subs["full"]) == idx.sample_id.nunique()
    assert 50 <= len(subs["mini"]) <= 80
    for split in idx.split.unique():
        assert idx[idx.sample_id.isin(subs["mini"]) & (idx.split == split)].shape[0] > 0
    sh = sorted((MINI / "shards").glob("shard-*.h5"))[0]
    paths = split_shard_by_task_group(sh, tmp_path, "S")
    assert set(paths) <= set(TASK_GROUPS) and "T1" in paths and "T4" in paths
    out = export_build(MINI, tmp_path / "hf", "S", subsets=subs)
    assert (tmp_path / "hf" / "index" / "S.parquet").exists() and (tmp_path / "hf" / "splits" / "mini" / "train.parquet").exists()
    ds = AmpScapeDataset("T4", split="train", tier="S", root=tmp_path / "hf", subset="mini")
    d = ds[0]
    assert len(ds) > 0 and d["cum_current"].shape == (1, 128, 128) and out.task_group.notna().all()
