-- Self-contained SQL/PGQ availability check.
-- Builds a tiny graph, runs a MATCH, tears it down. If this runs without a
-- syntax error, SQL/PGQ is live in your Postgres build.

DROP PROPERTY GRAPH IF EXISTS _smoke;
DROP TABLE IF EXISTS _e, _v;

CREATE TABLE _v (id int PRIMARY KEY, name text);
-- edge tables need their own key too (PK here lets PGQ infer the edge key)
CREATE TABLE _e (id serial PRIMARY KEY, src int REFERENCES _v(id), dst int REFERENCES _v(id));

INSERT INTO _v VALUES (1, 'a'), (2, 'b'), (3, 'c');
INSERT INTO _e (src, dst) VALUES (1, 2), (2, 3);

CREATE PROPERTY GRAPH _smoke
  VERTEX TABLES (_v LABEL thing PROPERTIES (id, name))
  EDGE TABLES (
    _e SOURCE KEY (src) REFERENCES _v (id)
       DESTINATION KEY (dst) REFERENCES _v (id)
       LABEL link
  );

SELECT *
FROM GRAPH_TABLE (_smoke
  MATCH (a IS thing)-[IS link]->(b IS thing)
  COLUMNS (a.id AS from_id, b.id AS to_id)
);

DROP PROPERTY GRAPH _smoke;
DROP TABLE _e, _v;
