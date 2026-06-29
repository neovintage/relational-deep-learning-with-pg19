"""Generate the full FK-derived edge layer from RelBench metadata.

For every foreign key in the dataset we emit one reified edge table (the owning
row's primary key is the edge key), index its FK endpoint for traversal, and
union all of them into a single bidirectional ``graph_edges`` view that recursive
neighborhood walks traverse (see ``extract.fetch_neighborhood_recursive``).

Driven entirely by ``fkey_col_to_pkey_table``, so it scales to any RelBench
dataset with no hand-written SQL. The ``results_*`` edge tables it produces are
the same ones ``sql/property_graph.sql`` declares the ``f1`` property graph over,
so this module is the single source of truth for the edge layer.

Usage:
    uv run python -m pg_rdl.build_graph --dataset rel-f1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import sqlalchemy as sa
from relbench.datasets import get_dataset

from pg_rdl import DATABASE_URL
from pg_rdl.load import to_snake


@dataclass(frozen=True)
class Edge:
    table: str      # edge table name, e.g. "results_driver"
    src_type: str   # owning table (and node type), e.g. "results"
    src_key: str    # owning primary key, snake_case, e.g. "result_id"
    dst_type: str   # target table (and node type), e.g. "drivers"
    dst_key: str    # foreign key column, snake_case, e.g. "driver_id"


def edge_specs(db) -> list[Edge]:
    """One Edge per foreign key in the dataset (snake_cased to match Postgres).

    The edge table is named ``{owner}_{fk-without-_id}`` (results.driver_id ->
    results_driver), matching the names ``sql/property_graph.sql`` expects.
    """
    edges: list[Edge] = []
    for owner, table in db.table_dict.items():
        if table.pkey_col is None:
            continue  # no key to reify the edge on
        src_key = to_snake(table.pkey_col)
        for fk_col, target in table.fkey_col_to_pkey_table.items():
            dst_key = to_snake(fk_col)
            edges.append(
                Edge(
                    table=f"{owner}_{dst_key.removesuffix('_id')}",
                    src_type=owner,
                    src_key=src_key,
                    dst_type=target,
                    dst_key=dst_key,
                )
            )
    return edges


def graph_edges_view_sql(edges: list[Edge]) -> str:
    """Bidirectional union of every edge table into (src_type, src_id, dst_type, dst_id).

    Each edge contributes two arms so a recursive walk can flow in either
    direction. ``src_type``/``dst_type`` are the table names, since node ids
    collide across types (driver_id 8 and race_id 8 are different nodes).
    """
    arms = []
    for e in edges:
        # forward: owning row -> target
        arms.append(
            f"SELECT '{e.src_type}'::text AS src_type, {e.src_key} AS src_id, "
            f"'{e.dst_type}'::text AS dst_type, {e.dst_key} AS dst_id, "
            f"'{e.table}'::text AS edge_table FROM {e.table}"
        )
        # reverse: target -> owning row
        arms.append(
            f"SELECT '{e.dst_type}'::text, {e.dst_key}, "
            f"'{e.src_type}'::text, {e.src_key}, '{e.table}'::text FROM {e.table}"
        )
    return "CREATE VIEW graph_edges AS\n" + "\nUNION ALL\n".join(arms)


def build(dataset: str, database_url: str) -> None:
    db = get_dataset(dataset, download=True).get_db()
    edges = edge_specs(db)
    engine = sa.create_engine(database_url)

    with engine.begin() as conn:
        # f1 (and graph_edges) depend on the edge tables; tear them down first so a
        # failed rebuild can't leave a half-built layer (the whole block is one tx).
        conn.execute(sa.text("DROP PROPERTY GRAPH IF EXISTS f1"))
        conn.execute(sa.text("DROP VIEW IF EXISTS graph_edges"))
        for e in edges:
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {e.table} CASCADE"))

        for e in edges:
            # The FK may be nullable (optional relationship); a null-target edge is
            # meaningless, so drop those rows. The PK still holds on the subset.
            conn.execute(sa.text(
                f"CREATE TABLE {e.table} AS "
                f"SELECT {e.src_key}, {e.dst_key} FROM {e.src_type} "
                f"WHERE {e.dst_key} IS NOT NULL"
            ))
            conn.execute(sa.text(f"ALTER TABLE {e.table} ADD PRIMARY KEY ({e.src_key})"))
            conn.execute(sa.text(f"CREATE INDEX ON {e.table} ({e.dst_key})"))
            print(f"  edge {e.table:28s} {e.src_type}.{e.src_key} -> {e.dst_type}.{e.dst_key}")

        conn.execute(sa.text(graph_edges_view_sql(edges)))
        print(f"  view graph_edges  ({2 * len(edges)} arms over {len(edges)} edge tables)")

    print(f"built {len(edges)} edge tables + graph_edges into "
          f"{engine.url.render_as_string(hide_password=True)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--database-url", default=DATABASE_URL)
    args = ap.parse_args()
    build(args.dataset, args.database_url)


if __name__ == "__main__":
    main()
