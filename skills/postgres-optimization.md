---
name: postgres-optimization
description: How to read query plans, choose indexes, and interpret PostgreSQL results safely.
version: 1
---

# PostgreSQL Query & Optimization Playbook

When generating or evaluating SQL against PostgreSQL:

## Safety first
- Emit a **single** `SELECT` (or `WITH … SELECT`). Never write DDL/DML — the executor
  rejects it and you waste a turn.
- Always pass values as **named bind parameters** (`:name`) via the tool's `params` object.
  Never string-interpolate user input into the SQL.
- If you don't know the schema, call the tool with `introspect: true` first.

## Writing good queries
- Project only the columns you need; avoid `SELECT *` on wide tables.
- Add a `LIMIT` when exploring; the executor caps rows but an explicit limit is clearer.
- Prefer set-based predicates (`IN`, `= ANY(:ids)`) over many `OR`s.

## Reading results
- Report row counts and summarize patterns rather than echoing every row.
- If a result set is large it is returned as an artifact reference — describe what's in it
  and how to use it rather than dumping the data.

## Optimization heuristics
- Slow filters on high-cardinality columns → suggest a btree index.
- Frequent range scans on timestamps → suggest a btree index on the timestamp column.
- Repeated joins on a foreign key → ensure both sides are indexed.
- Use `EXPLAIN (ANALYZE, BUFFERS)` mentally: a Seq Scan on a large table behind a selective
  filter is the usual culprit.
