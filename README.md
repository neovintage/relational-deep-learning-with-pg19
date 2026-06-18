# pg-rdl-experiment

A proof of concept for doing **Relational Deep Learning (RDL)** on top of
**PostgreSQL 19's SQL/PGQ** property-graph feature.

> Status: **runs end to end.**
> `rel-f1` loads into `postgres:19beta1`, the reified property graph builds, and
> SQL/PGQ time-bounded extraction feeds a GNN that trains and reports AUROC
> (`make up load graph train`). The model reuses RelBench's own encoder + GraphSAGE
> (`HeteroEncoder`, `HeteroGraphSAGE`); only the per-seed subgraph topology comes
> from SQL/PGQ instead of RelBench's graph builder.
>
> The training run is a **smoke test, not a benchmark**: it caps the number of
> seeds (one Postgres round-trip per seed), uses batch size 1, and skips
> empty-neighborhood seeds — so the AUROC it prints is *not* comparable to the
> RelBench leaderboard. Making it leaderboard-comparable (mini-batching, all
> entities) is the next step. See [docs/gotchas.md](docs/gotchas.md).

## The question this PoC is trying to answer

Relational Deep Learning and SQL/PGQ independently start from the same insight:
a relational schema with primary-/foreign-key links **is** a graph.

- **RDL** (Stanford / RelBench) turns that graph into a learning problem: each
  row is a node, each PK/FK link is an edge, deep tabular encoders produce node
  features, and a Graph Neural Network does message passing to make predictions.
- **SQL/PGQ** (new in Postgres 19) turns that same graph into a *query* surface:
  you declare which tables are nodes and which are edges, then pattern-match over
  them with `MATCH`. It compiles to relational joins. No learning, no tensors.

So the experiment is narrow and specific:

> **Can Postgres 19's SQL/PGQ serve as the "schema → graph" and
> neighborhood-extraction layer of an RDL pipeline, feeding subgraphs to an
> external GNN trainer?**

We already expect friction, and documenting it is part of the point:

- The PG19 SQL/PGQ implementation has **no variable-length paths** — every hop is
  written explicitly, and multi-hop traversal still needs recursive CTEs. GNNs
  are inherently multi-hop, so this is the central tension.
- PGQ pattern matching isn't built for the fast, randomized mini-batch neighbor
  sampling that GNN training wants at scale.

A successful PoC is not "this is the best way to train a GNN." It's a clear,
reproducible answer to whether the database can *own the graph definition* for
both querying and learning.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        PostgreSQL 19 (beta)                        │
│                                                                    │
│   relational tables  ──►  CREATE PROPERTY GRAPH (SQL/PGQ)          │
│   (RelBench dataset)      entities + REIFIED fact tables = nodes   │
│                           FK helper views/tables  = edges          │
│                                                                    │
│   MATCH queries  ──►  time-bounded k-hop neighborhood per seed     │
│   (optional) pgvector  ──►  store learned embeddings for serving   │
└───────────────────────────────┬──────────────────────────────────┘
                                 │  SQLAlchemy / psycopg (GRAPH_TABLE rows)
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Python extraction layer                       │
│   GRAPH_TABLE rows  ──►  (tf_dict, edge_index_dict) per seed       │
│   (node features sliced from once-materialized TensorFrames)       │
└───────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  GNN training (PyTorch / PyG)                       │
│   RelBench HeteroEncoder + HeteroGraphSAGE + binary head           │
│   (RDL recipe; only the graph builder is replaced by SQL/PGQ)      │
└──────────────────────────────────────────────────────────────────┘
```

### Components

| Layer | Tech | Role |
|-------|------|------|
| Data store | PostgreSQL 19 beta | Holds the relational dataset; defines the property graph via SQL/PGQ |
| Graph definition | `CREATE PROPERTY GRAPH` (SQL/PGQ, core PG19) | Declares nodes/edges over existing tables — no data movement |
| Extraction | Python + SQLAlchemy | Runs time-bounded `MATCH` queries, builds `(tf_dict, edge_index_dict)` per seed |
| Learning | PyTorch + PyG + RelBench | `HeteroEncoder` + `HeteroGraphSAGE` + binary head (the RDL model) |
| Serving (stretch) | `pgvector` | Land learned embeddings back in Postgres for inference |

### Dataset

We start with **`rel-f1`** (Formula 1) from RelBench — small, well-understood,
fast to iterate on. The loader scripts will ingest its CSVs into Postgres tables,
after which we define the property graph over them. Larger RelBench datasets are
a later concern.

### Predictive task: `driver-dnf`

The PoC targets one concrete RelBench task end to end:

| | |
|---|---|
| Task | **`driver-dnf`** |
| Type | Binary **entity-level classification** |
| Question | *For a given driver at time `t`, will they DNF (Did Not Finish) a race in the next 1 month?* |
| Entity | a driver (`driverId`) |
| Label | `1` if the driver DNFs any race in `(t, t + 30 days]`, else `0` |
| Metric | **AUROC** |

RelBench ships this as **train / validation / test label tables**, split
**temporally** (val and test are later in time than train). Each row is an
`(driverId, timestamp, label)` triple — the timestamp `t` is the *seed time* at
which the prediction is made.

That temporal split is the part that constrains the architecture:

> **No future leakage.** When we build a driver's neighborhood for a seed time
> `t`, the `MATCH` queries may only traverse races that happened **before `t`**.
> A naive property-graph extraction that ignores `t` will pull future races and
> silently inflate AUROC.

The other two `rel-f1` tasks (`driver-top3`, binary/AUROC; `driver-position`,
regression/MAE) are stretch goals once the `driver-dnf` path works end to end.

## Why this is a bit of a build

SQL/PGQ ships **inside Postgres 19 core** (it's not an extension), but Postgres 19
is in beta as of June 2026, so it is **not** available through Homebrew
(`brew` only goes up to `postgresql@18`). We therefore run it via Docker, which is
also the most reproducible path.

## Prerequisites (macOS, Apple Silicon)

- macOS with Homebrew
- A container runtime: **Docker Desktop** or **Colima** + Docker CLI
- [`uv`](https://docs.astral.sh/uv/) for Python environment management
- ~5 GB free disk for the Postgres image and dataset

```bash
brew install uv
brew install colima docker   # or install Docker Desktop
colima start                 # if using Colima
```

## Setup

### 1. Clone and enter

```bash
git clone <this-repo>
cd pg-rdl-experiment
```

### 2. Start Postgres 19

```bash
docker compose up -d
```

This launches the official `postgres:19beta1` image (pinned by digest in
`docker-compose.yml` for reproducibility) with a database named `rdl` on port
`5439` (non-standard to avoid colliding with a local PG18).

Connection string:

```
postgresql://rdl:rdl@localhost:5439/rdl
```

Verify the version, then run the self-contained SQL/PGQ smoke test
(`sql/smoke_test.sql`) which creates a tiny throwaway graph, runs a `MATCH`, and
tears it down:

```bash
psql postgresql://rdl:rdl@localhost:5439/rdl -c "SELECT version();"
psql postgresql://rdl:rdl@localhost:5439/rdl -f sql/smoke_test.sql
```

`sql/smoke_test.sql` (the full file is idempotent; abbreviated here):

```sql
CREATE TABLE _v (id int PRIMARY KEY, name text);
-- edge tables need their own key too — a PK lets PGQ infer the edge key
CREATE TABLE _e (id serial PRIMARY KEY, src int REFERENCES _v(id), dst int REFERENCES _v(id));

INSERT INTO _v VALUES (1,'a'), (2,'b'), (3,'c');
INSERT INTO _e (src, dst) VALUES (1,2), (2,3);

CREATE PROPERTY GRAPH _smoke
  VERTEX TABLES (_v LABEL thing PROPERTIES (id, name))
  EDGE TABLES (
    _e SOURCE KEY (src) REFERENCES _v (id)
       DESTINATION KEY (dst) REFERENCES _v (id)
       LABEL link
  );

SELECT * FROM GRAPH_TABLE (_smoke
  MATCH (a IS thing)-[IS link]->(b IS thing)
  COLUMNS (a.id AS from_id, b.id AS to_id));

DROP PROPERTY GRAPH _smoke;
DROP TABLE _e, _v;
```

If that returns two edge rows, SQL/PGQ is live. **Verified against
`postgres:19beta1`** — two gotchas this shook out: edge tables need their own key
(not just the vertices), and the data mount moved (the compose file mounts
`/var/lib/postgresql`, not `…/data`, per the PG18+ image convention).

### 3. Create the Python environment

```bash
uv sync
```

`uv` resolves and pins everything from `pyproject.toml` into `uv.lock`. PyTorch
and PyTorch Geometric wheels are selected for macOS / Apple Silicon (CPU build;
MPS where supported).

### 4. Load the dataset

```bash
uv run python -m pg_rdl.load --dataset rel-f1
```

Downloads `rel-f1` via the `relbench` package and ingests its 9 tables into
Postgres (`drivers`, `constructors`, `circuits`, `races`, `results`, `qualifying`,
`standings`, `constructor_results`, `constructor_standings`).

> Run `load` **before** `graph`. Re-running `load` after the graph exists will
> error: the loader does `to_sql(if_exists="replace")` (a plain `DROP TABLE`), and
> the property graph + helper edge tables depend on those tables. To reload from
> scratch: `make down && make up && make load graph`.

> **Identifier-casing decision.** RelBench's columns are camelCase (`driverId`,
> `statusId`, `positionOrder`). Postgres folds *unquoted* identifiers to lowercase,
> so `driverId` in SQL silently becomes `driverid`. The loader therefore writes
> columns **verbatim** and the SQL files **double-quote** mixed-case identifiers
> (e.g. `"driverId"`). The examples in this README omit the quotes for readability —
> the generated `.sql` files add them. (If we'd rather keep SQL quote-free, the
> alternative is to snake_case on ingest; recorded in Open questions.)

### 5. Define the property graph

```bash
psql postgresql://rdl:rdl@localhost:5439/rdl -f sql/property_graph.sql
```

#### The modeling decision: reify fact tables as vertices

SQL/PGQ and RDL disagree about what a fact table is, and we have to resolve it
before writing any SQL:

- **SQL/PGQ** models edges as **binary FK relationships**. A `results` row links a
  driver, a race, *and* a constructor — three FKs on one row. A naive binary edge
  has to pick one pair and demote the rest, throwing away the row's own feature
  columns (`grid`, `position`, `points`, `laps`, …).
- **RDL** treats **every row as a node**, fact tables included. `results` becomes
  its own node type, carrying its features, wired to driver, race, and constructor
  nodes — so the GNN learns representations *of result rows*.

**Decision: we reify the fact tables (`results`, `qualifying`, `standings`, …) as
vertices** and expose each of their FKs as a narrow edge **view**. This is the
RDL-faithful choice and the one that actually stresses PGQ's binary-edge model.
(The simpler FK-as-edge collapse and a Python-side hybrid are recorded in
[Open questions](#open-questions--things-well-hit) as fallbacks if this proves too
slow.)

The edge views are one-liners — each FK column paired with the fact-row PK:

```sql
CREATE VIEW results_driver      AS SELECT resultId, driverId      FROM results;
CREATE VIEW results_race        AS SELECT resultId, raceId        FROM results;
CREATE VIEW results_constructor AS SELECT resultId, constructorId FROM results;
```

`sql/property_graph.sql` then declares entities and the reified `results` node.
**Column names below are verified by actually loading `rel-f1`** (camelCase, kept
from the Ergast source — see `pg_rdl/load.py`). The relevant DNF signal lives in
`results.statusId` (a status code where `1` = *Finished* and the rest are
not-classified / retirement reasons); `position` is also null when a driver isn't
classified. RelBench's `driver-dnf` **label is precomputed from these** — we consume
it, we don't recompute it — but `statusId`/`position` are exactly the features a
result node should carry:

The id/KEY columns are listed in `PROPERTIES` explicitly — a KEY column is **not**
automatically queryable, so a `MATCH ... WHERE d.driverId = …` fails unless
`driverId` is also a property. (Verified the hard way.) The real `.sql` file
double-quotes every identifier; omitted here for readability.

```sql
CREATE PROPERTY GRAPH f1
  VERTEX TABLES (
    drivers      KEY (driverId)      LABEL driver
      PROPERTIES (driverId, code, nationality, dob),
    constructors KEY (constructorId) LABEL constructor
      PROPERTIES (constructorId, name, nationality),
    circuits     KEY (circuitId)     LABEL circuit
      PROPERTIES (circuitId, country),
    races        KEY (raceId)        LABEL race
      PROPERTIES (raceId, year, round, date),
    results      KEY (resultId)      LABEL result
      PROPERTIES (resultId, grid, position, positionOrder, points, laps, rank,
                  statusId, date)
  )
  EDGE TABLES (
    results_driver
      SOURCE KEY (resultId) REFERENCES results (resultId)
      DESTINATION KEY (driverId) REFERENCES drivers (driverId)
      LABEL of_driver,
    results_race
      SOURCE KEY (resultId) REFERENCES results (resultId)
      DESTINATION KEY (raceId) REFERENCES races (raceId)
      LABEL in_race,
    results_constructor
      SOURCE KEY (resultId) REFERENCES results (resultId)
      DESTINATION KEY (constructorId) REFERENCES constructors (constructorId)
      LABEL for_constructor
  );
```

Then extraction (`sql/extract.sql` / driven from `pg_rdl/extract.py`) pulls the
neighborhood a GNN needs, **bounded by the seed time `t`** so we never traverse into
the future (see the `driver-dnf` task above). For a driver at seed time `t`, the
relevant subgraph is their *past* result nodes and the races/constructors those
attach to — a textbook 2-hop message-passing neighborhood. Each hop is spelled out
because PG19 has no variable-length paths:

```sql
-- :driver_id and :seed_ts are bound per row of the driver-dnf label table
SELECT *
FROM GRAPH_TABLE (f1
  MATCH (d IS driver WHERE d.driverId = :driver_id)
        <-[IS of_driver]-(res IS result WHERE res.date < :seed_ts)
        -[IS in_race]->(ra IS race)
  COLUMNS (
    d.driverId   AS center_driver,
    res.resultId AS result_node,
    res.position AS position,
    res.statusId AS status_id,
    res.date     AS result_date,
    ra.raceId    AS race_node
  )
);
```

The `WHERE res.date < :seed_ts` predicate is the leakage guard — it sits on the
**result node** (which carries the event date), so every result fed into the
neighborhood predates the prediction. Proving this filter actually blocks leakage at
every hop is one of the things the PoC must demonstrate, not assume.

### 6. Run the PoC

```bash
uv sync                              # full deps, incl. torch
uv run python -m pg_rdl.train --task driver-dnf --max-train 400 --max-val 200
```

For each `(driverId, t, label)` seed, this extracts the driver's time-bounded
neighborhood via SQL/PGQ, slices per-node-type `TensorFrame`s from the feature
store, encodes them (RelBench's `HeteroEncoder`), message-passes
(`HeteroGraphSAGE`), and trains a binary head with `BCEWithLogitsLoss`, printing
**AUROC** per epoch on the temporal val split. Example output:

```
train seeds 200 | val seeds 150
epoch 00 | train loss 0.4644 auroc 0.3832 (n=186, skip=14) | val auroc 0.5777 ...
```

> **It's a smoke test, not a benchmark.** `--max-train` / `--max-val` cap the seeds
> (one Postgres round-trip each), batch size is 1, and empty-neighborhood seeds are
> skipped. The AUROC is *not* comparable to the RelBench leaderboard — it confirms
> the SQL/PGQ → GNN path trains, nothing more.

### The model (RDL recipe, `pg_rdl/model.py`)

We **reuse RelBench's own reference modules** so the learning recipe is the faithful
RDL baseline — the novelty of the PoC is the *data path* (SQL/PGQ → graph), not the
architecture:

1. **Per-table tabular encoder — RelBench's `HeteroEncoder` (over `pytorch-frame`).**
   `pg_rdl/features.py` materializes each node table once into a `TensorFrame` +
   column stats (categorical `statusId`/`nationality`, numerical `grid`/`position`/
   `points`/…). Per-seed subgraphs slice these by node id. This is the "deep tabular"
   half of RDL — no manual feature engineering.
2. **`HeteroGraphSAGE`** (RelBench's, channels 128) over node types `driver`,
   `result`, `race` and the bidirectional `result↔driver` / `result↔race` relations,
   so a driver's embedding absorbs its past results and the races they happened in.
3. **Task head.** An MLP on the seed `driver` node embedding → 1 logit, trained with
   `BCEWithLogitsLoss`, scored by **AUROC**.

> The only thing we *don't* take from RelBench is its graph builder
> (`make_pkey_fkey_graph`) — that's exactly the job SQL/PGQ is doing here. The
> curated feature columns (`pg_rdl/features.py:COL_TO_STYPE`) are a lightweight
> subset (categorical + numerical, no text/timestamp encoders), so absolute numbers
> won't match the published baseline.

A subtlety worth calling out: PyG's `NeighborLoader` can already do **temporal
neighbor sampling** natively via `time_attr` (only sample neighbors older than the
seed). That overlaps with our SQL/PGQ time-bounded extraction — so a real sub-question
of this PoC is whether the SQL/PGQ path *competes with* or *complements* PyG's own
sampler. That tension feeds directly into the null hypothesis below.

## Reproducibility checklist

- **Postgres** pinned by image digest in `docker-compose.yml` (not just the
  `19beta1` tag, which can be re-pushed).
- **Python deps** pinned in `uv.lock`; recreate exactly with `uv sync --frozen`.
- **Dataset** version pinned via the `relbench` package version.
- **Random seeds** set for data splits and model init (documented per run).
- No host Postgres required — everything runs against the container, so a local
  PG18 won't interfere.

## Project layout

```
pg-rdl-experiment/
├── README.md
├── Makefile                    # up / smoke / load / graph / extract / train
├── docker-compose.yml          # Postgres 19 beta, pinned by digest
├── pyproject.toml              # Python deps (uv)
├── uv.lock                     # committed — reproducible env
├── sql/
│   ├── smoke_test.sql          # self-contained SQL/PGQ availability check  [VERIFIED]
│   └── property_graph.sql      # reified property graph over rel-f1         [VERIFIED]
└── pg_rdl/
    ├── load.py                 # RelBench → Postgres ingestion               [VERIFIED]
    ├── features.py             # per-node-type TensorFrame materialization   [VERIFIED]
    ├── extract.py              # SQL/PGQ MATCH → (tf_dict, edge_index_dict)   [VERIFIED]
    ├── model.py                # RelBench HeteroEncoder + HeteroGraphSAGE + head  [VERIFIED]
    └── train.py                # per-seed training loop, AUROC on temporal val [VERIFIED]
```

Everything above is **run, not inferred** — `make up load graph` then
`uv run python -m pg_rdl.train` reproduces the smoke-test AUROC.

## Open questions / things we'll hit

- How do we express k-hop neighborhoods without variable-length paths — explicit
  per-hop `MATCH`, recursive CTEs, or a hybrid?
- Does the time filter (`res.date < :seed_ts`) correctly prevent leakage at every
  hop, and can we prove it (e.g. a leakage test that shuffles future results and
  confirms AUROC collapses to chance)?
- Is PGQ extraction fast enough to keep a GNN trainer fed across thousands of
  `(driver, t)` seeds, or does it become the bottleneck vs. a one-time export?
- Does SQL/PGQ extraction **compete with or complement** PyG's `NeighborLoader`,
  which already does temporal neighbor sampling natively via `time_attr`?
- Identifier casing: double-quote RelBench's camelCase in SQL, or snake_case on
  ingest? (Currently: quote. Revisit if the SQL gets unreadable.)
- Does defining the graph in the DB actually buy us anything over building the
  PyG graph directly from the tables? (This is the honest null hypothesis.)

## References

- [Relational Deep Learning (arXiv 2312.04615)](https://arxiv.org/abs/2312.04615)
- [RelBench (arXiv 2407.20060)](https://arxiv.org/abs/2407.20060) ·
  [snap-stanford/relbench](https://github.com/snap-stanford/relbench) ·
  [rel-f1 tasks](https://relbench.stanford.edu/datasets/rel-f1/)
- [PostgreSQL 19 Beta 1 release](https://www.postgresql.org/about/news/postgresql-19-beta-1-released-3313/)
- [Representing graphs in PostgreSQL with SQL/PGQ (EDB)](https://www.enterprisedb.com/blog/representing-graphs-postgresql-sqlpgq)
- [PyTorch Frame (tabular deep learning)](https://pytorch-frame.readthedocs.io)
