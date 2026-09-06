"""Schema validator: accepts real shards, rejects corrupted copies; schema.md renders."""

from __future__ import annotations

import json
import pathlib
import shutil

import h5py
import numpy as np
import pytest

from ampscape.io.schema import MetaModel, schema_markdown, validate_shard

ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "builds" / "smoke5" / "shards" / "shard-00000.h5"
pytestmark = pytest.mark.skipif(not SMOKE.exists(), reason="smoke shard not present")


def test_valid_shard_passes():
    rep = validate_shard(str(SMOKE))
    assert rep.ok and rep.n_samples == 3 and rep.n_configs == 15


def _corrupt(tmp_path, fn):
    p = tmp_path / "bad.h5"
    shutil.copy(SMOKE, p)
    with h5py.File(p, "r+") as f:
        fn(f)
    return validate_shard(str(p))


def test_detects_bad_dtype_shape_values_and_meta(tmp_path):
    def bad_resistance(f):
        sid = list(f)[0]
        g = f[sid]["inputs"]
        r = g["resistance"][...]
        r[0, 0] = 0.5                     # < 1 on a valid pixel
        del g["resistance"]
        g.create_dataset("resistance", data=r)
    rep = _corrupt(tmp_path, bad_resistance)
    assert any("resistance < 1" in e for e in rep.errors)

    def bad_dtype(f):
        sid = list(f)[0]
        g = f[sid]["inputs"]
        m = g["nodata_mask"][...].astype(np.int32)
        del g["nodata_mask"]
        g.create_dataset("nodata_mask", data=m)
    rep = _corrupt(tmp_path, bad_dtype)
    assert any("nodata_mask: dtype" in e for e in rep.errors)

    def missing_output(f):
        sid = list(f)[0]
        del f[sid]["configs"]["points"]["outputs"]["reff"]
    rep = _corrupt(tmp_path, missing_output)
    assert any("missing dataset reff" in e for e in rep.errors)

    def nonfinite(f):
        sid = list(f)[0]
        g = f[sid]["configs"]["advanced"]["outputs"]
        c = g["current"][...]
        c[1, 1] = np.nan
        del g["current"]
        g.create_dataset("current", data=c)
    rep = _corrupt(tmp_path, nonfinite)
    assert any("non-finite" in e for e in rep.errors)

    def bad_meta(f):
        sid = list(f)[0]
        m = json.loads(f[sid].attrs["meta"])
        del m["solver_versions"]
        f[sid].attrs["meta"] = json.dumps(m)
    rep = _corrupt(tmp_path, bad_meta)
    assert any("meta invalid" in e for e in rep.errors)


def test_meta_model_and_schema_md():
    with h5py.File(SMOKE) as f:
        sid = list(f)[0]
        m = MetaModel.model_validate(json.loads(f[sid].attrs["meta"]))
        assert m.sample_id == sid and m.solver_versions["circuitscape"] == "5.17.1"
    md = schema_markdown()
    assert "## Sample group" in md and "omniscape" in md and "`resistance`" in md


def test_zarr_export_roundtrip(tmp_path):
    import zarr

    from ampscape.io.zarr_export import export_zarr

    n = export_zarr(SMOKE, tmp_path / "shard.zarr", max_samples=1)
    assert n == 1
    g = zarr.open_group(str(tmp_path / "shard.zarr"), mode="r")
    sid = list(g.group_keys())[0]
    with h5py.File(SMOKE) as f:
        np.testing.assert_array_equal(g[sid]["inputs"]["resistance"][...], f[sid]["inputs"]["resistance"][...])
        np.testing.assert_array_equal(g[sid]["configs"]["points"]["outputs"]["cum_current"][...],
                                      f[sid]["configs"]["points"]["outputs"]["cum_current"][...])
        assert json.loads(g[sid].attrs["meta"])["sample_id"] == sid
