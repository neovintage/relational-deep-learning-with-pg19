-- Reified property graph over the loaded rel-f1 tables.
--
-- RDL treats every row as a node, fact tables included. SQL/PGQ edges are binary
-- FK relationships. We bridge the two by *reifying* the `results` fact table as a
-- vertex and exposing each of its foreign keys as a narrow edge table (each with
-- its own primary key, which is what SQL/PGQ needs to infer the edge key).
--
-- The edge tables (results_driver, results_race, results_constructor) are built by
-- the generator in pg_rdl/build_graph.py, which reifies *every* foreign key in the
-- schema. Run it first; `make graph` does (build_graph, then this file). Here we
-- only declare the property-graph surface over those tables.
--
-- Identifiers are unquoted: the loader snake_cases the camelCase RelBench columns
-- on ingest (driverId -> driver_id), so everything folds cleanly to lowercase.

DROP PROPERTY GRAPH IF EXISTS f1;

CREATE PROPERTY GRAPH f1
  -- The KEY column is not automatically a queryable property, so we list the id
  -- columns explicitly alongside the feature columns.
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
      PROPERTIES (result_id, grid, position, position_order, points,
                  laps, rank, status_id, date)
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
