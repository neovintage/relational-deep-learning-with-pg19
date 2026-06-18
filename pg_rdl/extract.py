"""SQL/PGQ extraction → PyG HeteroData.

Two layers:
- ``fetch_neighborhood`` is **torch-free** and is the part verified live against
  Postgres 19: it runs the time-bounded GRAPH_TABLE / MATCH query and returns rows.
- ``build_subgraph`` marshals those rows (plus FeatureStore lookups) into a
  ``(tf_dict, edge_index_dict)`` pair for the model — the learning-path half of the
  experiment, now wired up and runnable via ``pg_rdl.train``.

The MATCH below is fixed-depth on purpose: PG19 SQL/PGQ has no variable-length
paths, so every hop is written explicitly. The ``res."date" < :seed_ts`` predicate
is the leakage guard — it sits on the result node, which carries the event date.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import sqlalchemy as sa

# driver -> past results -> races (and the constructor on each result).
# Bound parameters: :driver_id, :seed_ts.
NEIGHBORHOOD_SQL = sa.text(
    """
    SELECT * FROM GRAPH_TABLE (f1
      MATCH (d IS driver WHERE d."driverId" = :driver_id)
            <-[IS of_driver]-(res IS result WHERE res."date" < :seed_ts)
            -[IS in_race]->(ra IS race)
      COLUMNS (
        d."driverId"   AS center_driver,
        res."resultId" AS result_node,
        res."grid"     AS grid,
        res."position" AS position,
        res."statusId" AS status_id,
        res."date"     AS result_date,
        ra."raceId"    AS race_node,
        ra."year"      AS race_year
      )
    )
    """
)


def fetch_neighborhood(
    engine: sa.Engine, driver_id: int, seed_ts: datetime
) -> pd.DataFrame:
    """Run the time-bounded SQL/PGQ extraction for one (driver, seed) pair.

    Torch-free. Verified live against postgres:19beta1.
    """
    with engine.connect() as conn:
        return pd.read_sql(
            NEIGHBORHOOD_SQL, conn,
            params={"driver_id": int(driver_id), "seed_ts": seed_ts},
        )


# Node and edge types of the per-seed subgraph. Edges are bidirectional so message
# passing can flow race -> result -> driver (2 hops) and back.
NODE_TYPES = ["driver", "result", "race"]
EDGE_TYPES = [
    ("result", "to_driver", "driver"),
    ("driver", "to_result", "result"),
    ("race", "to_result", "result"),
    ("result", "to_race", "race"),
]


def build_subgraph(rows: pd.DataFrame, driver_id: int, fs):
    """Turn extracted neighborhood rows into (tf_dict, edge_index_dict).

    ``tf_dict`` holds a per-node-type TensorFrame sliced from the FeatureStore by
    node id; ``edge_index_dict`` holds the topology in local (contiguous) indices.
    The single seed driver is local index 0 of the ``driver`` node type.

    RelBench guarantees ids are 0..N-1, so a SQL/PGQ id indexes the TensorFrame
    directly.
    """
    import torch

    result_ids = rows["result_node"].astype(int).unique().tolist()
    race_ids = rows["race_node"].astype(int).unique().tolist()
    r_local = {rid: i for i, rid in enumerate(result_ids)}
    ra_local = {rid: i for i, rid in enumerate(race_ids)}

    tf_dict = {
        "driver": fs.tf["driver"][torch.tensor([int(driver_id)])],
        "result": fs.tf["result"][torch.tensor(result_ids)],
        "race": fs.tf["race"][torch.tensor(race_ids)],
    }

    res_idx = torch.tensor([r_local[r] for r in result_ids], dtype=torch.long)
    drv_idx = torch.zeros(len(result_ids), dtype=torch.long)  # all -> driver 0
    er_src = torch.tensor([r_local[int(r)] for r in rows["result_node"]], dtype=torch.long)
    er_dst = torch.tensor([ra_local[int(r)] for r in rows["race_node"]], dtype=torch.long)

    edge_index_dict = {
        ("result", "to_driver", "driver"): torch.stack([res_idx, drv_idx]),
        ("driver", "to_result", "result"): torch.stack([drv_idx, res_idx]),
        ("race", "to_result", "result"): torch.stack([er_dst, er_src]),
        ("result", "to_race", "race"): torch.stack([er_src, er_dst]),
    }
    return tf_dict, edge_index_dict
