# Gotchas

Things that bit us building this PoC against PostgreSQL 19 Beta 1 + SQL/PGQ +
RelBench. Each one was found by actually running the code, not by reading docs.

## Postgres / Docker

### 1. PG18+ images mount data at `/var/lib/postgresql`, not `/var/lib/postgresql/data`

The stock `docker-compose.yml` snippet everyone copies mounts the volume at
`/var/lib/postgresql/data`. On `postgres:18`+ images the container **exits on
startup** with:

```
Error: in 18+, these Docker images are configured to store database data in a
       subdirectory ... place a single mount at /var/lib/postgresql
```

Fix: mount the parent directory.

```yaml
volumes:
  - pgdata:/var/lib/postgresql      # not /var/lib/postgresql/data
```

### 2. Pin the image by digest, not just the beta tag

`postgres:19beta1` is a moving tag during the beta — it can be re-pushed. For a
reproducible PoC, pin the digest:

```bash
docker inspect --format='{{index .RepoDigests 0}}' postgres:19beta1
# postgres@sha256:dc371...  → use that in docker-compose.yml
```

## SQL/PGQ (`CREATE PROPERTY GRAPH`)

### 3. Edge tables need their own key — not just the vertex tables

Vertex keys can be inferred from a PRIMARY KEY, but an **edge table also needs a
key**. Without one:

```
ERROR: no key specified and no suitable primary key exists for definition of
       element "_e"
```

Fix: give every edge table (including the reified FK helper tables) its own PK, or
an explicit `KEY (...)` clause. In this project each `results_<entity>` helper
table gets `PRIMARY KEY ("resultId")` (one result row → exactly one edge).

### 4. A KEY column is **not** automatically a queryable property

If you write an explicit `PROPERTIES (...)` list, only those columns are
queryable. The KEY/id column is *not* added for free, so:

```sql
MATCH (d IS driver WHERE d."driverId" = 1)   -- ERROR if driverId not in PROPERTIES
```

fails with `property "driverId" for element variable "d" not found`. Fix: list the
id column in `PROPERTIES` alongside the features.

### 5. `PROPERTIES` takes column names/aliases only — no computed expressions

`PROPERTIES ((position IS NULL) AS dnf)` is **not** allowed. Properties are bare
column names with optional `col AS name` aliasing. Compute derived signals
(e.g. DNF = `position IS NULL`) in the query `COLUMNS (...)` or downstream in
Python, not in the graph definition.

### 6. camelCase identifiers must be double-quoted everywhere

RelBench keeps Ergast's camelCase (`driverId`, `statusId`, `positionOrder`).
Postgres folds **unquoted** identifiers to lowercase, so `driverId` silently
becomes `driverid` → `column "driverid" does not exist`. Every identifier in every
`.sql` file and query string is double-quoted. (Alternative: snake_case on ingest;
we chose to quote, to keep a 1:1 mapping with RelBench's column names.)

## RelBench / data

### 7. Don't trust secondhand schema summaries — load the data

A web summary claimed RelBench's cleaned `results` table drops its status column.
**False.** Loading `rel-f1` shows `results` keeps both `statusId` and
`positionOrder`:

```
results cols: resultId, raceId, driverId, constructorId, number, grid, position,
              positionOrder, points, laps, milliseconds, fastestLap, rank,
              statusId, date
```

`statusId = 1` means *Finished*; other values are retirement/not-classified
reasons, and `position` is null when a driver isn't classified. The DNF label for
the `driver-dnf` task is **precomputed by RelBench** from these — consume it, don't
recompute it.

### 8. `to_sql(if_exists="replace")` breaks the graph if you reload

`pandas.to_sql(..., if_exists="replace")` issues a plain `DROP TABLE`. Once the
property graph and `results_<entity>` helper tables exist, they depend on the base
tables, so re-running `load` errors. Always `load` **before** `graph`; to reload
from scratch: `make down && make up && make load graph`.

## Python / env

### 9. Python 3.14 is too new for the torch/relbench wheel matrix (2026-06)

The host had Python 3.14; the torch / torch-geometric / relbench wheels don't cover
it yet. `pyproject.toml` pins `requires-python = ">=3.10,<3.14"` and uv selects a
compatible interpreter (3.13) automatically.

## Modeling / training

### 10. Reuse RelBench's encoder, not its graph builder

RelBench ships `HeteroEncoder` and `HeteroGraphSAGE` (in `relbench.modeling.nn`).
The encoder consumes per-node-type `TensorFrame`s and is fully reusable — that's the
faithful RDL "deep tabular" recipe. The part you must **not** reuse is
`make_pkey_fkey_graph` (in `relbench.modeling.graph`): building the graph from PK/FK
links is exactly the job SQL/PGQ is doing in this project. We materialize features
with RelBench's tooling and build topology from SQL/PGQ.

### 11. RelBench reindexes primary keys to `0..N-1`

`make_pkey_fkey_graph` asserts `df[pkey] == arange(len)`. RelBench guarantees it, and
we load those ids verbatim into Postgres — so an id returned by a SQL/PGQ MATCH is a
**direct row index** into the materialized `TensorFrame`. No id→index map needed.

### 12. `TensorFrame` materialization is once-per-table, then indexed

Compute `Dataset(df, col_to_stype).materialize()` **once over the full table** (stats
must be global), then slice per-seed subgraphs with `tf[id_tensor]`. Don't
re-materialize per subgraph. Stick to categorical/numerical stypes unless you wire up
a text embedder — `statusId` is categorical, `position`/`rank` are numerical with
NaNs that torch_frame imputes from the global stats.

### 13. Per-seed extraction means batch size 1 → noisy AUROC

The thesis (SQL/PGQ extracts each neighborhood) makes the natural loop one Postgres
round-trip per seed, i.e. batch size 1. That's slow (hence `--max-train`/`--max-val`
caps) and gives noisy gradients, so the smoke-test AUROC bounces around chance. This
is a known limitation, **not** a bug: making it leaderboard-grade means batching
multiple seeds into one disjoint `HeteroData` (and scoring every entity, not a
capped, empty-skipped subset). Report the number as a smoke test, never a benchmark.

### 14. A single-logit head needs a shape-`[1]` target

The head emits `[1]` for the one seed driver node; `BCEWithLogitsLoss` rejects a
scalar target (`Target size ([]) must be the same as input size ([1])`). Wrap the
label as `torch.tensor([float(label)])`.
