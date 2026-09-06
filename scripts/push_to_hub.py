#!/usr/bin/env python
"""Push an HF-layout staging directory to the (private) dataset repo. Dry-run by default; --push is owner-gated.

    python scripts/push_to_hub.py --layout data/hf/AmpScape [--repo Xirro/AmpScape] [--push] [--private]

Steps: (1) create the repo if missing (private), (2) upload README (dataset card), croissant.json, index/, splits/,
stats/, (3) upload every shard with the sync path (sha256 verified against the Hub's LFS object, .uploaded marker),
(4) print a manifest. Requires `hf auth login` on the login node.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--repo", default=f"{os.environ.get('HF_ORG', 'Xirro')}/AmpScape")
    ap.add_argument("--card", default=str(ROOT / "docs" / "dataset_card.md"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true", default=True)
    a = ap.parse_args()
    layout = pathlib.Path(a.layout)
    shutil.copy(a.card, layout / "README.md")
    shards = sorted(layout.glob("data/*/*/*.h5"))
    small = [p for p in layout.rglob("*") if p.is_file() and p.suffix != ".h5" and not p.name.endswith((".ok", ".uploaded", ".invalid"))]
    print(f"repo {a.repo}: {len(shards)} shards ({sum(p.stat().st_size for p in shards)/1e6:.0f} MB), {len(small)} metadata files")
    if not a.push:
        for p in shards[:5] + small[:8]:
            print("  would upload", p.relative_to(layout))
        print("dry run (pass --push after `hf auth login`)")
        return 0
    from huggingface_hub import HfApi

    from ampscape.io.sync import remote_sha256, sha256

    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(layout), allow_patterns=["README.md", "croissant.json", "index/*", "splits/**", "stats/*"],
                      commit_message="metadata: card, croissant, index, splits, stats")
    manifest = []
    for sh in shards:
        rel = str(sh.relative_to(layout))
        marker = sh.with_suffix(".uploaded")
        local = sha256(sh)
        if marker.exists() and json.loads(marker.read_text()).get("sha256") == local:
            manifest.append({"path": rel, "status": "already"})
            continue
        ok = False
        for attempt in range(1, 4):
            res = api.upload_file(path_or_fileobj=str(sh), path_in_repo=rel, repo_id=a.repo, repo_type="dataset", commit_message=f"add {rel}")
            remote = remote_sha256(api, a.repo, rel)
            if remote == local:
                marker.write_text(json.dumps({"repo": a.repo, "path": rel, "sha256": local, "commit": getattr(res, "oid", None), "attempt": attempt}))
                ok = True
                break
        manifest.append({"path": rel, "status": "verified" if ok else "MISMATCH", "sha256": local})
        print(("ok  " if ok else "FAIL"), rel)
    (layout / "upload_manifest.json").write_text(json.dumps(manifest, indent=1))
    bad = [m for m in manifest if m["status"] == "MISMATCH"]
    print(f"done: {len(manifest) - len(bad)} verified, {len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
