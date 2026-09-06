#!/usr/bin/env python
"""CLI wrapper: python scripts/sync_shards.py --build data/builds/mini --tier S [--push --delete]  (default: dry run)."""
from ampscape.io.sync import main

if __name__ == "__main__":
    raise SystemExit(main())
