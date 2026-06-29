# pg-rdl-experiment

> NOTE: I used claude to bootstrap this project and reviewed a lot of it myself to get a deeper understanding of what was going on.

This is me going through the process of seeing if PG19 property graphs (SQL/PGQ) would be useful for doing **Relational Deep Learning (RDL)**. This isn't a benchmark in any way and more of an exploration on DX. 

**The code runs and is reproducable**. This exploration caps the number of seeds (one Postgres round-trip per see), uses batch size 1, and skips empty-neighborhood seeds. AUROC will print at the end of the run but it's not comparable to the RelBench leaderboard. So don't do it!

Next Steps: writing Postgres adaptors for PyG Remote Backends. This will take care of all of the mini-batching, all entities, etc. For Postgres services that do branching or allow for read replicas, doing the training on that could be very worth while.

## Why did I do this in the first place?

Relational Deep Learning and SQL/PGQ have the same insight which is a relational schema with primary-/foreign-key links **is** a graph.

- **RDL** (Stanford / RelBench) turns that graph into a learning problem: each row is a node, each PK/FK link is an edge, deep tabular encoders produce node features, and a Graph Neural Network does message passing to make predictions.
- **SQL/PGQ** (new in Postgres 19) turns that same graph into a *query* surface. You can declare which tables are nodes and which are edges, then pattern-match over them with `MATCH`. It compiles to relational joins. 

So the experiment is narrow and specific:

> **Can Postgres 19's Property Graphs serve as the "schema → graph" and neighborhood-extraction layer of an RDL pipeline, feeding subgraphs to an external GNN trainer?**

We already expect friction, and documenting it is part of the point:

- The PG19 SQL/PGQ implementation has **no variable-length paths**. This means every hop is written explicitly, and multi-hop traversal still needs recursive CTEs. GNNs are inherently multi-hop, so we're already off to a bad start.
- PGQ pattern matching isn't built for the fast, randomized mini-batch neighbor sampling that GNN training wants at scale.

A successful PoC is not "this is the best way to train a GNN." It's a clear, reproducible answer to whether the Postgres can *own the graph definition* for both querying and learning.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        PostgreSQL 19 (beta)                        │
│                                                                    │
│   tables (RelBench)  ──►  build_graph.py create every FK as       │
│                           an edge table (row = node, FK = edge)    │
│                           + CREATE PROPERTY GRAPH f1 over them     │
│                           + graph_edges view (unified edge list)   │
│                                                                    │
│   extract per seed   ──►  time-bounded k-hop neighborhood:         │
│     MATCH (fixed 2)  or   recursive CTE over graph_edges (any k)   │
│   (optional) pgvector  ──►  store learned embeddings for serving   │
└───────────────────────────────┬──────────────────────────────────┘
                                 │  SQLAlchemy / psycopg (rows)
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
| Edge layer | `pg_rdl/build_graph.py` (generated from FK metadata) | Reifies every foreign key into an edge table (+ PK + endpoint index) and a unified `graph_edges` view |
| Graph definition | `CREATE PROPERTY GRAPH` (SQL/PGQ, core PG19) | Declares `f1` over the generated edge tables — no data movement |
| Extraction | Python + `psycopg` | Fixed-depth `MATCH` *or* recursive CTE over `graph_edges`; builds PyG `HeteroData` subgraphs |
| Learning | PyTorch + PyTorch Geometric | GNN message passing + tabular encoders (the RDL model) |
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

Verify the version, then run the self-contained SQL/PGQ smoke test (`sql/smoke_test.sql`) which creates a tiny throwaway graph, runs a `MATCH`, and tears it down. `smoke_test.sql` is idempotent, dont worry if you run it multiple times. The commands and the output should look like this:

```bash
$ docker compose exec db psql -U rdl -d rdl
rdl=# SELECT version();
                                      version
---------------------------------------------------------------------------------------------------------------------------------
PostgreSQL 19beta1 (Debian 19~beta1-1.pgdg13+1) on aarch64-unknown-linux-gnu, compiled by gcc (Debian 14.2.0-19) 14.2.0, 64-bit
(1 row)

$ docker exec -i pg-rdl psql -U rdl -d rdl < sql/smoke_test.sql
psql:sql/smoke_test.sql:5: NOTICE:  property graph "_smoke" does not exist, skipping
DROP PROPERTY GRAPH
psql:sql/smoke_test.sql:6: NOTICE:  table "_e" does not exist, skipping
psql:sql/smoke_test.sql:6: NOTICE:  table "_v" does not exist, skipping
DROP TABLE
CREATE TABLE
CREATE TABLE
INSERT 0 3
INSERT 0 2
CREATE PROPERTY GRAPH
 from_id | to_id
---------+-------
       1 |     2
       2 |     3
(2 rows)

DROP PROPERTY GRAPH
DROP TABLE
```


### 3. Create the Python environment

```bash
uv sync
```

### 4. Load the dataset

```bash
$ uv run python -m pg_rdl.load --dataset rel-f1
Loading Database object from ~/pg-rdl-experiment/data/rel-f1/db...
Done in 0.02 seconds.
  resetting schema public
  writing qualifying               (  4,082 rows)
  writing drivers                  (    857 rows)
  writing results                  ( 20,323 rows)
  writing standings                ( 28,115 rows)
  writing races                    (    820 rows)
  writing constructors             (    211 rows)
  writing constructor_results      (  9,408 rows)
  writing circuits                 (     77 rows)
  writing constructor_standings    ( 10,170 rows)
  PK   qualifying.qualifyId
  PK   drivers.driverId
  PK   results.resultId
  PK   standings.driverStandingsId
  PK   races.raceId
  PK   constructors.constructorId
  PK   constructor_results.constructorResultsId
  PK   circuits.circuitId
  PK   constructor_standings.constructorStandingsId
loaded rel-f1 into postgresql+psycopg://rdl:***@localhost:5439/rdl
```

Downloads `rel-f1` via the `relbench` package and ingests its 9 tables into Postgres (`drivers`, `constructors`, `circuits`, `races`, `results`, `qualifying`, `standings`, `constructor_results`, `constructor_standings`). The data will be cached on disk in the `./data` folder and all of the identifiers in the data will also be converted to snake case (e.g. `driverId` becomes `driver_id`)

> Run `load` **before** `graph`. Re-running `load` after the graph exists will error: the loader does `to_sql(if_exists="replace")` (a plain `DROP TABLE`), and the property graph + helper edge tables depend on those tables. To reload from scratch: `make down && make up && make load graph`.

### 5. Build the edge layer and define the property graph

First generate the edge tables (`build_graph.py` reifies every FK + builds the `graph_edges` view), then declare `f1` over them. `make graph` runs both in order:

```bash
$ make graph
# equivalently, by hand:
$ uv run python -m pg_rdl.build_graph --dataset rel-f1
$ docker exec -i pg-rdl psql -U rdl -d rdl < sql/property_graph.sql
```

### 6. Poke around the data model

If you're looking to see what property graphs exist within the database you'll need to make sure you're on a psql version that's at least 19. Since you're in the docker container, you'll be fine. For this purposes of this exercise you'll see that we only have one defined `f1`.

```bash
rdl=# \dG
List of property graphs
Schema | Name |      Type      | Owner
--------+------+----------------+-------
public | f1   | property graph | rdl
(1 row)
```

Just like any other table, view, or what have you, describe it within psql to get a more detailed definition:

```bash
rdl=# \d f1
Property Graph "public.f1"
Element Alias    |       Element Table        | Element Kind | Source Vertex Alias | Destination Vertex Alias
---------------------+----------------------------+--------------+---------------------+--------------------------
circuits            | public.circuits            | vertex       |                     |
constructors        | public.constructors        | vertex       |                     |
drivers             | public.drivers             | vertex       |                     |
races               | public.races               | vertex       |                     |
results             | public.results             | vertex       |                     |
results_constructor | public.results_constructor | edge         | results             | constructors
results_driver      | public.results_driver      | edge         | results             | drivers
results_race        | public.results_race        | edge         | results             | races
```

`f1` only declares the three `results_*` edges, but the generator built an edge table for *every* foreign key in the schema. They're ordinary tables, so list them like anything else:

```bash
rdl=# SELECT relname AS edge_table
      FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = 'public' AND c.relkind = 'r'
        AND c.relname ~ '_(driver|race|constructor|circuit)$'
      ORDER BY 1;
            edge_table
-----------------------------------
 constructor_results_constructor
 constructor_results_race
 constructor_standings_constructor
 constructor_standings_race
 qualifying_constructor
 qualifying_driver
 qualifying_race
 races_circuit
 results_constructor
 results_driver
 results_race
 standings_driver
 standings_race
(13 rows)
```

Those 13 tables get unioned (both directions) into the `graph_edges` view, which is the homogeneous edge list the recursive extractor walks. Each row is one directed hop between two namespaced nodes:

```bash
rdl=# SELECT count(*) FROM graph_edges;
 count
--------
 338842
(1 row)

rdl=# SELECT * FROM graph_edges ORDER BY src_type, src_id LIMIT 6;
 src_type | src_id | dst_type | dst_id |  edge_table
----------+--------+----------+--------+---------------
 circuits |      0 | races    |    614 | races_circuit
 circuits |      0 | races    |    646 | races_circuit
 circuits |      0 | races    |    581 | races_circuit
 circuits |      0 | races    |    597 | races_circuit
 circuits |      0 | races    |    630 | races_circuit
 circuits |      0 | races    |    663 | races_circuit
(6 rows)
```

The sample shows the *reverse* arm of `races_circuit` (circuit → its races): every edge table contributes both directions, so a recursive walk can flow either way. Row counts across the 13 tables range from 820 (`races_circuit`) to 28,115 (`standings_driver`); the 338,842 total is twice the sum, since each edge is stored once per direction.

#### The modeling decision: fact tables as vertices

A row always becomes a graph element. The catch in SQL/PGQ is that you have to assign each table a role: a table is *either* a vertex or an edge, and an edge is binary (one source vertex, one destination vertex). It applies one rule to the whole schema: every row is a node, every foreign key is an edge. That rule is trivial in tensor-land (the RelBench reference code builds the graph in memory as a PyG `HeteroData` object), but it becomes a challenge with the postgres binary-edge constraint the moment you hit a fact table. I haven't looked into the SQL/PGQ spec to know if I'm missing something here, so I will acknowledge that.

A `results` row points at a driver, a race, *and* a constructor. Three foreign keys on one row. It also carries its own features: `grid`, `position`, `position_order`, `points`, `laps`, `status_id`. You can't model that as a single edge. A binary edge picks one pair, drops the other two, and has nowhere to put the feature columns. So `results` has to be a vertex, and the edges have to live somewhere else.

We make `results` a vertex and reify each of its foreign keys as its own narrow edge table. `pg_rdl/build_graph.py` does this for *every* foreign key in the schema (13 of them in rel-f1), driven straight from RelBench's FK metadata so there's no hand-written SQL. Here are the three it emits for `results`:

```sql
CREATE TABLE results_driver      AS SELECT result_id, driver_id      FROM results;
CREATE TABLE results_race        AS SELECT result_id, race_id        FROM results;
CREATE TABLE results_constructor AS SELECT result_id, constructor_id FROM results;

ALTER TABLE results_driver      ADD PRIMARY KEY (result_id);
ALTER TABLE results_race        ADD PRIMARY KEY (result_id);
ALTER TABLE results_constructor ADD PRIMARY KEY (result_id);
```

Each edge table also gets an index on its FK endpoint, and they all get unioned (both directions) into a single `graph_edges(src_type, src_id, dst_type, dst_id)` view that the recursive extractor walks.

The `ADD PRIMARY KEY` is the part that matters. SQL/PGQ infers an edge's key from the table's primary key, so without it these tables can't be used as edges. The key is `result_id` because one result row references exactly one driver, one race, and one constructor, so `result_id` is unique per edge in all three tables. I probably could use `MATERIALIZED VIEWS` instead of actually creating tables but it's unlikely `CREATE VIEW` would work.

The edge layer covers the *whole* schema now: `qualifying`, `standings`, `constructor_results`, `constructor_standings`, and `races → circuits` are all reified into edge tables and the `graph_edges` view too. The `f1` property graph below is a deliberately curated subset, just the results-centered neighborhood the `driver-dnf` MATCH needs. The recursive extractor (further down) walks the full `graph_edges` layer instead, so it isn't limited to what `f1` declares.

```sql
CREATE PROPERTY GRAPH f1
  VERTEX TABLES (
    drivers      KEY (driver_id)      LABEL driver
      PROPERTIES (driver_id, code, nationality, dob),
    constructors KEY (constructor_id) LABEL constructor
      PROPERTIES (constructor_id, name, nationality),
    circuits     KEY (circuit_id)     LABEL circuit
      PROPERTIES (circuit_id, country),
    races        KEY (race_id)        LABEL race
      PROPERTIES (race_id, year, round, date),
    results      KEY (result_id)      LABEL result
      PROPERTIES (result_id, grid, position, position_order, points, laps, rank,
                  status_id, date)
  )
  EDGE TABLES (
    results_driver
      SOURCE KEY (result_id) REFERENCES results (result_id)
      DESTINATION KEY (driver_id) REFERENCES drivers (driver_id)
      LABEL of_driver,
    results_race
      SOURCE KEY (result_id) REFERENCES results (result_id)
      DESTINATION KEY (race_id) REFERENCES races (race_id)
      LABEL in_race,
    results_constructor
      SOURCE KEY (result_id) REFERENCES results (result_id)
      DESTINATION KEY (constructor_id) REFERENCES constructors (constructor_id)
      LABEL for_constructor
  );
```

Then extraction (`pg_rdl/extract.py`) pulls the neighborhood a GNN needs, **bounded by the seed time `t`** so we never traverse into the future (see the `driver-dnf` task above). For a driver at seed time `t`, the relevant subgraph is their *past* result nodes and the races/constructors those attach to. There are two extraction paths, with identical output columns so the downstream `build_subgraph` doesn't care which one ran.

The first is `fetch_neighborhood`, a fixed-depth `GRAPH_TABLE` / `MATCH` over `f1`. Each hop is spelled out because PG19 has no variable-length paths:

```sql
-- :driver_id and :seed_ts are bound per row of the driver-dnf label table
SELECT *
FROM GRAPH_TABLE (f1
  MATCH (d IS driver WHERE d.driver_id = :driver_id)
        <-[IS of_driver]-(res IS result WHERE res.date < :seed_ts)
        -[IS in_race]->(ra IS race)
  COLUMNS (
    d.driver_id   AS center_driver,
    res.result_id AS result_node,
    res.position  AS position,
    res.status_id AS status_id,
    res.date      AS result_date,
    ra.race_id    AS race_node
  )
);
```

The `WHERE res.date < :seed_ts` predicate is the leakage guard. It sits on the **result node** (which carries the event date), so every result fed into the neighborhood predates the prediction. Proving this filter actually blocks leakage at every hop is one of the things I'm trying to demonstrate.

The second path is `fetch_neighborhood_recursive`, a recursive CTE over the `graph_edges` view. Instead of writing each hop by hand it walks the edge layer out to a `max_hops` parameter, so depth is a knob rather than a rewrite. It returns the same columns as the MATCH version (restricted to the seed driver's own past results, which is what `build_subgraph` assumes), so it's a drop-in alternative. For `max_hops = 2` on the current model it produces the same neighborhood as the MATCH; the point is that it generalizes to any depth and traverses the *full* edge layer, not just the curated `f1` subset. It's the path you'd reach for once you want deeper or richer neighborhoods, the thing PG19's missing variable-length paths would otherwise make you hand-write.

### 6. Run the test!

```bash
uv sync                              # full deps, incl. torch
uv run python -m pg_rdl.train --task driver-dnf --max-train 400 --max-val 200
```

For each `(driverId, t, label)` seed, this extracts the driver's time-bounded neighborhood via SQL/PGQ, slices per-node-type `TensorFrame`s from the feature store, encodes them (RelBench's `HeteroEncoder`), message-passes (`HeteroGraphSAGE`), and trains a binary head with `BCEWithLogitsLoss`, printing
**AUROC** per epoch on the temporal val split. Example output:

```
train seeds 200 | val seeds 150
epoch 00 | train loss 0.4644 auroc 0.3832 (n=186, skip=14) | val auroc 0.5777 ...
```

> **It's a smoke test, not a benchmark.** `--max-train` / `--max-val` cap the seeds (one Postgres round-trip each), batch size is 1, and empty-neighborhood seeds are skipped. The AUROC is *not* comparable to the RelBench leaderboard. This test confirms the SQL/PGQ → GNN path trains, nothing more.

### The model (RDL recipe, `pg_rdl/model.py`)

We **reuse RelBench's own reference modules** so the learning recipe is the faithful RDL baseline — the novelty of this test is the *data path* (SQL/PGQ → graph), not the architecture:

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

NOTE: PyG's `NeighborLoader` can already do **temporal neighbor sampling** natively via `time_attr` (only sample neighbors older than the seed). That overlaps with our SQL/PGQ time-bounded extraction.

## Things on my mind

- How do we express k-hop neighborhoods without variable-length paths? Is it per-hop `MATCH`, recursive CTEs, or something else? I think it's going to be recursive CTEs if you want maximum configurablility without having to think about the actual schema itself. I'm hypothesizing that most schemas don't evolve that quickly and because of that having explicit per-hop match statements might just be more scalable. 
- Does the time filter (`res.date < :seed_ts`) correctly prevent leakage at every hop, and can we prove it (e.g. a leakage test that shuffles future results and confirms AUROC collapses to chance)? I won't know this until I do the next steps of the project.
- Is PGQ extraction fast enough to keep a GNN trainer fed across thousands of `(driver, t)` seeds, or does it become the bottleneck vs. a one-time export? Going back to my idea around having a read replica or a branch to work from, it'll be a function of how many queries you can execute without impacting replication (if you're using a read replica)
- Does SQL/PGQ extraction **compete with or complement** PyG's `NeighborLoader`, which already does temporal neighbor sampling natively via `time_attr`? I don't know yet. I think this is a function of the next steps that I'll take on which are to build out the PyG RemoteBackend for Postgres.
- Does defining the graph in the DB actually buy us anything over building the PyG graph directly from the tables? I think the answer is yes because we're not putting the data into another storage medium to make the training and precidtions happen.

## References

- [Relational Deep Learning (arXiv 2312.04615)](https://arxiv.org/abs/2312.04615)
- [RelBench (arXiv 2407.20060)](https://arxiv.org/abs/2407.20060) ·
  [snap-stanford/relbench](https://github.com/snap-stanford/relbench) ·
  [rel-f1 tasks](https://relbench.stanford.edu/datasets/rel-f1/)
- [PostgreSQL 19 Beta 1 release](https://www.postgresql.org/about/news/postgresql-19-beta-1-released-3313/)
- [Representing graphs in PostgreSQL with SQL/PGQ (EDB)](https://www.enterprisedb.com/blog/representing-graphs-postgresql-sqlpgq)
- [PyTorch Frame (tabular deep learning)](https://pytorch-frame.readthedocs.io)
