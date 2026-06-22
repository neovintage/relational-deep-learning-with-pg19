"""Relational Deep Learning on PostgreSQL 19 SQL/PGQ property graphs."""

import os
from pathlib import Path

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://rdl:rdl@localhost:5439/rdl"
)

# Keep the RelBench download/cache inside the repo (./data) instead of the OS
# cache dir (~/Library/Caches on macOS), so the data is visible alongside the
# code for anyone working with it. This runs when the `pg_rdl` package is
# imported, which `python -m pg_rdl.load` does before load.py imports relbench,
# so it lands before pooch reads RELBENCH_CACHE_DIR. The contents are gitignored
# (see .gitignore); `data/.gitkeep` keeps the directory tracked. Override by
# exporting RELBENCH_CACHE_DIR yourself.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
os.environ.setdefault("RELBENCH_CACHE_DIR", str(DATA_DIR))
