"""Relational Deep Learning on PostgreSQL 19 SQL/PGQ property graphs."""

import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://rdl:rdl@localhost:5439/rdl"
)
