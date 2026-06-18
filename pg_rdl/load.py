"""Ingest a RelBench relational database into Postgres.

Torch-free: this is the "data path" half of the experiment. It writes each
RelBench table as a Postgres table, preserving the original (camelCase) column
names, then adds the PRIMARY KEY constraints that SQL/PGQ needs to infer vertex
and edge keys in ``sql/property_graph.sql``.

Usage:
    uv run python -m pg_rdl.load --dataset rel-f1
"""

from __future__ import annotations

import argparse

import sqlalchemy as sa
from relbench.datasets import get_dataset

from pg_rdl import DATABASE_URL


def load(dataset: str, database_url: str) -> None:
    db = get_dataset(dataset, download=True).get_db()
    engine = sa.create_engine(database_url)

    with engine.begin() as conn:
        for name, table in db.table_dict.items():
            print(f"  writing {name:24s} ({len(table.df):>7,} rows)")
            # to_sql preserves the exact (camelCase) column names; SQLAlchemy
            # quotes them on CREATE, so they stay case-sensitive in Postgres and
            # must be double-quoted in every query (see sql/property_graph.sql).
            table.df.to_sql(
                name, conn, if_exists="replace", index=False, method="multi",
                chunksize=2000,
            )

        # Primary keys — required for SQL/PGQ key inference on the vertex tables
        # and for the REFERENCES targets used by the reified edge tables.
        for name, table in db.table_dict.items():
            if table.pkey_col is None:
                print(f"  ! {name} has no pkey_col; skipping PK")
                continue
            pk = table.pkey_col
            conn.execute(
                sa.text(f'ALTER TABLE "{name}" ADD PRIMARY KEY ("{pk}")')
            )
            print(f"  PK   {name}.{pk}")

    print(f"loaded {dataset} into {engine.url.render_as_string(hide_password=True)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--database-url", default=DATABASE_URL)
    args = ap.parse_args()
    load(args.dataset, args.database_url)


if __name__ == "__main__":
    main()
