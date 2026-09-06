"""Optional Zarr export of an HDF5 shard (brief §8.1): same group layout, same attrs, chunked per array."""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import zarr


def _json_safe(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_json_safe(x) for x in v.tolist()]
    return v


def export_zarr(shard_h5: str | pathlib.Path, out: str | pathlib.Path, max_samples: int | None = None) -> int:
    """Copy every sample group (datasets + attributes) of ``shard_h5`` into a Zarr store at ``out``."""
    root = zarr.open_group(str(out), mode="w")
    n = 0
    with h5py.File(shard_h5, "r") as f:
        for k, v in f.attrs.items():
            root.attrs[k] = _json_safe(v)

        def copy_group(src: h5py.Group, dst: zarr.Group) -> None:
            for k, v in src.attrs.items():
                dst.attrs[k] = _json_safe(v)
            for name, item in src.items():
                if isinstance(item, h5py.Group):
                    copy_group(item, dst.create_group(name))
                else:
                    arr = item[...]
                    z = dst.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=arr.shape if arr.ndim <= 1 else arr.shape[-2:] if arr.ndim == 2 else (1, *arr.shape[-2:]))
                    z[...] = arr
                    for k, v in item.attrs.items():
                        z.attrs[k] = _json_safe(v)

        for i, sid in enumerate(f):
            if max_samples and i >= max_samples:
                break
            copy_group(f[sid], root.create_group(sid))
            n += 1
    return n
