"""PNG quicklooks per sample (inputs, cumulative current, voltage where present) and a contact sheet."""

from __future__ import annotations

import json
import pathlib

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _log(a, mask=None):
    a = np.asarray(a, dtype=np.float64)
    img = np.log10(np.maximum(a, 1e-12))
    return np.ma.masked_array(img, mask) if mask is not None else img


def sample_quicklook(gs: h5py.Group, path: str) -> None:
    meta = json.loads(gs.attrs["meta"])
    R = gs["inputs"]["resistance"][...]
    nd = gs["inputs"]["nodata_mask"][...] > 0
    cfgs = list(gs["configs"])
    panels = [("log10 R", np.ma.masked_array(np.log10(R), nd), "magma")]
    for c in cfgs:
        g = gs["configs"][c]
        kind = g.attrs["kind"]
        o = g["outputs"]
        if kind in ("points", "wall_to_wall", "regions"):
            panels.append((f"{c}\ncum current", _log(o["cum_current"][...], nd), "viridis"))
            if "voltage" in o:
                panels.append((f"{c}\nvoltage pair 1", np.ma.masked_array(o["voltage"][0], nd), "coolwarm"))
        elif kind == "advanced":
            panels.append(("T3 current", _log(o["current"][...], nd), "viridis"))
            panels.append(("T3 voltage", np.ma.masked_array(o["voltage"][...], nd), "coolwarm"))
        else:
            panels.append(("T4 cum current", _log(o["cum_current"][...], nd), "viridis"))
            panels.append(("T4 normalized", np.ma.masked_array(o["normalized"][...], nd), "cividis"))
    n = len(panels)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, img, cmap) in zip(axes, panels, strict=False):
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=7)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{meta['sample_id'][:8]} {meta['family']} {meta.get('generator')} {meta.get('resistance_table_id') or ''} "
                 f"qc={','.join(meta.get('qc_flags', [])) or 'clean'}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def shard_quicklooks(final_h5: str, out_dir: str) -> list[str]:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with h5py.File(final_h5, "r") as f:
        for sid in f:
            p = out / f"{sid}.png"
            sample_quicklook(f[sid], str(p))
            paths.append(str(p))
    return paths


def contact_sheet(final_h5s: list[str], path: str, max_samples: int = 60, key: str = "points") -> None:
    """Grid of (log R, cum current) thumbnails for the first samples across shards."""
    thumbs = []
    for fh in final_h5s:
        with h5py.File(fh, "r") as f:
            for sid in f:
                if len(thumbs) >= max_samples:
                    break
                gs = f[sid]
                if key not in gs["configs"]:
                    continue
                nd = gs["inputs"]["nodata_mask"][...] > 0
                thumbs.append((np.ma.masked_array(np.log10(gs["inputs"]["resistance"][...]), nd),
                               _log(gs["configs"][key]["outputs"]["cum_current"][...], nd), json.loads(gs.attrs["meta"])))
    n = len(thumbs)
    cols = 10
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(1.6 * cols, 1.7 * rows * 2))
    axes = np.atleast_2d(axes)
    for i, (r_img, c_img, meta) in enumerate(thumbs):
        rr, cc = divmod(i, cols)
        axes[2 * rr, cc].imshow(r_img, cmap="magma", interpolation="nearest")
        axes[2 * rr + 1, cc].imshow(c_img, cmap="viridis", interpolation="nearest")
        axes[2 * rr, cc].set_title((meta.get("generator") or "")[:10] + " " + (meta.get("resistance_table_id") or "")[:8], fontsize=5)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
