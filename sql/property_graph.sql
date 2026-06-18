-- Reified property graph over the loaded rel-f1 tables.
--
-- RDL treats every row as a node, fact tables included. SQL/PGQ edges are binary
-- FK relationships. We bridge the two by *reifying* the `results` fact table as a
-- vertex and exposing each of its foreign keys as a narrow edge table. Each edge
-- table gets its own primary key (resultId is unique per edge, since one result
-- row references exactly one driver / race / constructor), which is what SQL/PGQ
-- needs to infer the edge key.
--
-- All identifiers are double-quoted: RelBench columns are camelCase and Postgres
-- folds unquoted identifiers to lowercase.

DROP PROPERTY GRAPH IF EXISTS f1;
DROP TABLE IF EXISTS results_driver, results_race, results_constructor;

CREATE TABLE results_driver      AS SELECT "resultId", "driverId"      FROM results;
CREATE TABLE results_race        AS SELECT "resultId", "raceId"        FROM results;
CREATE TABLE results_constructor AS SELECT "resultId", "constructorId" FROM results;

ALTER TABLE results_driver      ADD PRIMARY KEY ("resultId");
ALTER TABLE results_race        ADD PRIMARY KEY ("resultId");
ALTER TABLE results_constructor ADD PRIMARY KEY ("resultId");

CREATE PROPERTY GRAPH f1
  -- The KEY column is not automatically a queryable property, so we list the id
  -- columns explicitly alongside the feature columns.
  VERTEX TABLES (
    drivers      KEY ("driverId")      LABEL driver
      PROPERTIES ("driverId", "code", "nationality", "dob"),
    constructors KEY ("constructorId") LABEL constructor
      PROPERTIES ("constructorId", "name", "nationality"),
    circuits     KEY ("circuitId")     LABEL circuit
      PROPERTIES ("circuitId", "country"),
    races        KEY ("raceId")        LABEL race
      PROPERTIES ("raceId", "year", "round", "date"),
    results      KEY ("resultId")      LABEL result
      PROPERTIES ("resultId", "grid", "position", "positionOrder", "points",
                  "laps", "rank", "statusId", "date")
  )
  EDGE TABLES (
    results_driver
      SOURCE KEY ("resultId") REFERENCES results ("resultId")
      DESTINATION KEY ("driverId") REFERENCES drivers ("driverId")
      LABEL of_driver,
    results_race
      SOURCE KEY ("resultId") REFERENCES results ("resultId")
      DESTINATION KEY ("raceId") REFERENCES races ("raceId")
      LABEL in_race,
    results_constructor
      SOURCE KEY ("resultId") REFERENCES results ("resultId")
      DESTINATION KEY ("constructorId") REFERENCES constructors ("constructorId")
      LABEL for_constructor
  );
