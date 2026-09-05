#!/usr/bin/env python
"""Render a gallery of synthetic landscapes (log resistance) for visual QC.

Usage: python scripts/preview_synthetic.py --out docs/figures/synthetic_gallery.png --n 24 --size 128
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ecoflowbench.landscapes import synthetic as syn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()
    cols = 6
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= args.n:
            continue
        ls = syn.sample_landscape(args.seed0 + i, (args.size, args.size))
        img = np.log10(ls.resistance)
        img = np.ma.masked_array(img, ls.nodata_mask)
        ax.imshow(img, cmap="magma", vmin=0, vmax=np.log10(ls.contrast), interpolation="nearest")
        ax.set_facecolor("0.6")
        tags = [ls.generator]
        if "patch_mosaic" in ls.params:
            tags.append("mosaic")
        if "barriers" in ls.params:
            tags.append("barriers")
        if "nodata" in ls.params:
            tags.append("nodata")
        ax.set_title(f"{'+'.join(tags)}\ncontrast {ls.contrast}", fontsize=7)
    fig.suptitle("EcoFlowBench synthetic landscapes: log10 resistance (grey = NoData)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
