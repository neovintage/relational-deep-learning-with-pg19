"""The RDL model: RelBench's tabular encoder + heterogeneous GraphSAGE + a head.

We reuse RelBench's reference modules (``HeteroEncoder`` over PyTorch Frame, and
``HeteroGraphSAGE``) so the learning recipe is the faithful RDL baseline. The
novelty of this project is upstream — the per-seed subgraphs are built from SQL/PGQ
extraction (``extract.build_subgraph``), not from RelBench's graph builder.
"""

from __future__ import annotations

import torch
from torch import nn

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE


class RDLModel(nn.Module):
    def __init__(self, fs, node_types, edge_types, channels: int = 128):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict=fs.col_names,
            node_to_col_stats=fs.col_stats,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=node_types,
            edge_types=edge_types,
            channels=channels,
        )
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),  # one logit for driver-dnf (binary)
        )

    def forward(self, tf_dict, edge_index_dict) -> torch.Tensor:
        x_dict = self.encoder(tf_dict)
        x_dict = self.gnn(x_dict, edge_index_dict)
        return self.head(x_dict["driver"]).squeeze(-1)
