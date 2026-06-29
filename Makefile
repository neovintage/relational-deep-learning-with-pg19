DB_URL ?= postgresql://rdl:rdl@localhost:5439/rdl

.PHONY: up down smoke load graph extract train data-deps

up:                ## start Postgres 19
	docker compose up -d

down:              ## stop and remove Postgres 19 (keeps the volume)
	docker compose down

smoke:             ## verify SQL/PGQ is available
	psql $(DB_URL) -v ON_ERROR_STOP=1 -f sql/smoke_test.sql

data-deps:         ## install only the torch-free data path
	uv sync --extra data

load:              ## ingest rel-f1 into Postgres (+ primary keys)
	uv run python -m pg_rdl.load --dataset rel-f1

graph:             ## build the full FK-derived edge layer + property graph
	uv run python -m pg_rdl.build_graph --dataset rel-f1
	psql $(DB_URL) -v ON_ERROR_STOP=1 -f sql/property_graph.sql

extract:           ## sample one driver's time-bounded neighborhood (sanity check)
	psql $(DB_URL) -v ON_ERROR_STOP=1 -c "SELECT * FROM GRAPH_TABLE (f1 \
	  MATCH (d IS driver WHERE d.driver_id = 1) \
	        <-[IS of_driver]-(res IS result WHERE res.date < DATE '2010-01-01') \
	        -[IS in_race]->(ra IS race) \
	  COLUMNS (d.driver_id AS center, res.result_id AS result_node, \
	           res.status_id AS status_id, res.date AS result_date)) LIMIT 10;"

train:             ## train driver-dnf (requires full deps incl. torch)
	uv run python -m pg_rdl.train --task driver-dnf
