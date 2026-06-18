"""Training loop for the driver-dnf task.

For each (driverId, t, label) seed, extract the driver's time-bounded neighborhood
via SQL/PGQ (``extract.fetch_neighborhood``), build a HeteroData subgraph
(``extract.build_subgraph``), encode + message-pass (``model.RDLModel``), and train
a binary classifier. Evaluate with AUROC on the temporal validation split.

NOTE ON SCALE: this does one Postgres round-trip per seed, so for a runnable demo we
cap the number of train/val seeds (--max-train / --max-val). The printed AUROC is a
SMOKE-TEST number on a subset with empty-neighborhood seeds skipped — it is NOT
comparable to RelBench's leaderboard (which scores every entity).

Usage:
    uv run python -m pg_rdl.train --task driver-dnf --max-train 400 --max-val 200
"""

from __future__ import annotations

import argparse
import warnings

# torch_frame converts read-only numpy arrays to tensors; the warning is benign.
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

import sqlalchemy as sa
import torch
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score

from pg_rdl import DATABASE_URL
from pg_rdl.extract import EDGE_TYPES, NODE_TYPES, build_subgraph, fetch_neighborhood
from pg_rdl.features import FeatureStore
from pg_rdl.model import RDLModel


def _seeds(task, split: str, max_n: int, seed: int):
    df = task.get_table(split).df
    if max_n and len(df) > max_n:
        df = df.sample(n=max_n, random_state=seed)
    return df


def _run_split(model, engine, fs, df, task, optimizer=None):
    """One pass over the seeds. Trains if optimizer is given, else evaluates."""
    train = optimizer is not None
    model.train(train)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    total_loss, n, skipped = 0.0, 0, 0
    ys, ps = [], []

    for _, row in df.iterrows():
        rows = fetch_neighborhood(engine, row[task.entity_col], row[task.time_col])
        if rows.empty:  # driver with no prior results — nothing to message-pass on
            skipped += 1
            continue
        tf_dict, edge_index_dict = build_subgraph(rows, row[task.entity_col], fs)
        label = torch.tensor([float(row[task.target_col])])  # shape [1] to match logit

        with torch.set_grad_enabled(train):
            logit = model(tf_dict, edge_index_dict)
            loss = loss_fn(logit, label)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n += 1
        ys.append(float(row[task.target_col]))
        ps.append(torch.sigmoid(logit).item())

    auroc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return total_loss / max(n, 1), auroc, n, skipped


def run(dataset, task_name, database_url, epochs, max_train, max_val, seed):
    torch.manual_seed(seed)
    engine = sa.create_engine(database_url)
    db = get_dataset(dataset, download=True).get_db()
    task = get_task(dataset, task_name, download=True)

    fs = FeatureStore(db)
    model = RDLModel(fs, NODE_TYPES, EDGE_TYPES, channels=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    train_df = _seeds(task, "train", max_train, seed)
    val_df = _seeds(task, "val", max_val, seed)
    print(f"train seeds {len(train_df)} | val seeds {len(val_df)}")

    for epoch in range(epochs):
        tr_loss, tr_auc, tr_n, tr_skip = _run_split(
            model, engine, fs, train_df, task, optimizer
        )
        _, va_auc, va_n, va_skip = _run_split(model, engine, fs, val_df, task)
        print(
            f"epoch {epoch:02d} | train loss {tr_loss:.4f} auroc {tr_auc:.4f} "
            f"(n={tr_n}, skip={tr_skip}) | val auroc {va_auc:.4f} "
            f"(n={va_n}, skip={va_skip})"
        )

    print("\n[smoke-test AUROC — capped subset, empty neighborhoods skipped; "
          "not comparable to the RelBench leaderboard]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--task", default="driver-dnf")
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=400)
    ap.add_argument("--max-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.dataset, args.task, args.database_url, args.epochs,
        args.max_train, args.max_val, args.seed)


if __name__ == "__main__":
    main()
