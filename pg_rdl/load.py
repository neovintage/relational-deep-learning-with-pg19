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
import re

import sqlalchemy as sa
from relbench.datasets import get_dataset

from pg_rdl import DATABASE_URL


def to_snake(name: str) -> str:
    """camelCase -> snake_case (driverId -> driver_id, positionOrder -> position_order).

    Snake-casing on ingest keeps every identifier lowercase, so the SQL files
    don't have to double-quote mixed-case columns. The first sub splits an
    acronym/word boundary (``...Results`` -> ``..._Results``); the second splits
    the trailing single-cap run (``...sId`` -> ``...s_Id``).
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def load(dataset: str, database_url: str) -> None:
    db = get_dataset(dataset, download=True).get_db()
    engine = sa.create_engine(database_url)

    with engine.begin() as conn:
        # Make the load idempotent. pandas' to_sql(if_exists="replace") issues a
        # bare DROP TABLE, which fails once sql/property_graph.sql has built the
        # reified edge tables and the `f1` property graph on top of these base
        # tables ("cannot drop table ... because other objects depend on it").
        # Resetting the schema with CASCADE clears those dependents first.
        print("  resetting schema public")
        conn.execute(sa.text("DROP SCHEMA public CASCADE"))
        conn.execute(sa.text("CREATE SCHEMA public"))

        for name, table in db.table_dict.items():
            print(f"  writing {name:24s} ({len(table.df):>7,} rows)")
            # Snake-case the camelCase RelBench columns on ingest so every
            # identifier folds cleanly to lowercase and the SQL files can stay
            # quote-free (see sql/property_graph.sql).
            df = table.df.rename(columns=to_snake)
            df.to_sql(
                name, conn, if_exists="replace", index=False, method="multi",
                chunksize=2000,
            )

        # Primary keys — required for SQL/PGQ key inference on the vertex tables
        # and for the REFERENCES targets used by the reified edge tables.
        for name, table in db.table_dict.items():
            if table.pkey_col is None:
                print(f"  ! {name} has no pkey_col; skipping PK")
                continue
            pk = to_snake(table.pkey_col)
            conn.execute(
                sa.text(f'ALTER TABLE {name} ADD PRIMARY KEY ({pk})')
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
