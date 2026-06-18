"""Per-node-type feature materialization via PyTorch Frame.

This is the "deep tabular" half of RDL and is reused verbatim from the RelBench
recipe (``torch_frame.data.Dataset(...).materialize()``): each node table is turned
into a ``TensorFrame`` plus column statistics, computed **once over the full table**.
Per-seed subgraphs (built from SQL/PGQ extraction in ``extract.py``) then just index
into these by node id.

This is deliberately the *only* place we lean on RelBench's own tooling — and it's
the feature encoder, not the graph builder. The graph topology comes from SQL/PGQ,
which is the whole point of the experiment.

RelBench guarantees each table's primary key is ``0..N-1`` (consecutive), and we
loaded those ids verbatim into Postgres, so an id returned by a SQL/PGQ MATCH is a
direct row index into the corresponding TensorFrame.
"""

from __future__ import annotations

import torch_frame
from torch_frame.data import Dataset

# Node label -> RelBench table name.
NODE_TABLE = {"driver": "drivers", "result": "results", "race": "races"}

# Curated column types per table. We intentionally stick to categorical/numerical
# (no text/timestamp) so no text-embedder is needed. statusId is the DNF status
# code; position/rank carry NaN for unclassified finishers (torch_frame imputes).
COL_TO_STYPE = {
    "drivers": {
        "nationality": torch_frame.categorical,
    },
    "results": {
        "grid": torch_frame.numerical,
        "position": torch_frame.numerical,
        "positionOrder": torch_frame.numerical,
        "points": torch_frame.numerical,
        "laps": torch_frame.numerical,
        "rank": torch_frame.numerical,
        "statusId": torch_frame.categorical,
    },
    "races": {
        "year": torch_frame.numerical,
        "round": torch_frame.numerical,
    },
}


class FeatureStore:
    """Materialized TensorFrames + stats for each node type, keyed by node label."""

    def __init__(self, db):
        self.tf = {}          # node label -> TensorFrame (row index == node id)
        self.col_stats = {}   # node label -> {col: {StatType: ...}}
        self.col_names = {}   # node label -> {stype: [col, ...]}
        for node, table in NODE_TABLE.items():
            df = db.table_dict[table].df
            ds = Dataset(df=df, col_to_stype=COL_TO_STYPE[table]).materialize()
            self.tf[node] = ds.tensor_frame
            self.col_stats[node] = ds.col_stats
            self.col_names[node] = ds.tensor_frame.col_names_dict
