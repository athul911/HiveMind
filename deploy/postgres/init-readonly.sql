-- Provisions a least-privilege read-only role used by the SQL execution tool.
-- The tool connects as `hivemind_ro`, which can only SELECT — never write or DDL.
-- This is a defense-in-depth boundary around LLM-generated SQL.

CREATE ROLE hivemind_ro WITH LOGIN PASSWORD 'hivemind_ro';

GRANT CONNECT ON DATABASE hivemind TO hivemind_ro;
GRANT USAGE ON SCHEMA public TO hivemind_ro;

-- Read-only on all current and future tables in public.
GRANT SELECT ON ALL TABLES IN SCHEMA public TO hivemind_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO hivemind_ro;

-- Explicitly deny write privileges (revoke anything inherited).
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM hivemind_ro;

-- A demo table so the SQL specialist has something to introspect/query out of the box.
CREATE TABLE IF NOT EXISTS demo_sales (
    id          SERIAL PRIMARY KEY,
    product     TEXT NOT NULL,
    region      TEXT NOT NULL,
    amount      NUMERIC(12,2) NOT NULL,
    sold_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO demo_sales (product, region, amount) VALUES
    ('Widget', 'EMEA', 120.50),
    ('Widget', 'AMER', 340.00),
    ('Gadget', 'APAC', 99.99),
    ('Gadget', 'EMEA', 220.10)
ON CONFLICT DO NOTHING;
GRANT SELECT ON demo_sales TO hivemind_ro;
