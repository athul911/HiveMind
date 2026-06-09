-- Richer analytics dataset for exercising the SQL specialist agent.
-- Safe to re-run (drops + recreates). Run via: make seed-data
-- The SQL tool connects as the read-only role `hivemind_ro`; we grant SELECT explicitly
-- at the end so the agent can query these tables.

SET client_min_messages = warning;
SELECT setseed(0.42);  -- reproducible random data

DROP TABLE IF EXISTS orders, products, customers CASCADE;

CREATE TABLE customers (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    region      TEXT NOT NULL,          -- EMEA / AMER / APAC
    segment     TEXT NOT NULL,          -- Consumer / SMB / Enterprise
    signup_date DATE NOT NULL
);

CREATE TABLE products (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,          -- Electronics / Home / Sports / Books
    unit_price  NUMERIC(10,2) NOT NULL
);

CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INT NOT NULL REFERENCES customers(id),
    product_id  INT NOT NULL REFERENCES products(id),
    quantity    INT NOT NULL,
    amount      NUMERIC(12,2) NOT NULL,
    status      TEXT NOT NULL,          -- completed / pending / refunded
    ordered_at  TIMESTAMPTZ NOT NULL
);

-- 12 products
INSERT INTO products (name, category, unit_price) VALUES
    ('Aurora Headphones', 'Electronics', 129.99),
    ('Nimbus Speaker',    'Electronics', 89.50),
    ('Pixel Monitor 27"', 'Electronics', 349.00),
    ('Comfort Office Chair', 'Home',     219.00),
    ('Cedar Desk',        'Home',        399.00),
    ('Lumen Floor Lamp',  'Home',        74.25),
    ('Trail Running Shoes','Sports',     119.00),
    ('Carbon Tennis Racket','Sports',    159.99),
    ('Yoga Mat Pro',      'Sports',      45.00),
    ('The Pragmatic Engineer','Books',   39.99),
    ('Designing Data Systems','Books',   54.99),
    ('SQL Performance Guide','Books',    34.50);

-- 60 customers spread across regions and segments
INSERT INTO customers (name, region, segment, signup_date)
SELECT
    'Customer ' || g,
    (ARRAY['EMEA','AMER','APAC'])[1 + floor(random()*3)::int],
    (ARRAY['Consumer','SMB','Enterprise'])[1 + floor(random()*3)::int],
    (CURRENT_DATE - (floor(random()*900)::int))
FROM generate_series(1, 60) g;

-- 800 orders over the last ~12 months
INSERT INTO orders (customer_id, product_id, quantity, amount, status, ordered_at)
SELECT
    1 + floor(random()*60)::int,
    p.id,
    q.quantity,
    q.quantity * p.unit_price,
    (ARRAY['completed','completed','completed','pending','refunded'])[1 + floor(random()*5)::int],
    now() - (random()*365 || ' days')::interval
FROM generate_series(1, 800) s
CROSS JOIN LATERAL (SELECT 1 + floor(random()*5)::int AS quantity) q
CROSS JOIN LATERAL (SELECT id, unit_price FROM products ORDER BY random() LIMIT 1) p;

CREATE INDEX ix_orders_customer ON orders(customer_id);
CREATE INDEX ix_orders_product  ON orders(product_id);
CREATE INDEX ix_orders_time     ON orders(ordered_at);

-- Let the read-only SQL-tool role read everything (guarded in case the role isn't present
-- on an older volume — it's normally created by deploy/postgres/init-readonly.sql).
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hivemind_ro') THEN
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO hivemind_ro;
  END IF;
END $$;

-- Quick sanity summary (printed when run interactively).
SELECT
    (SELECT count(*) FROM customers) AS customers,
    (SELECT count(*) FROM products)  AS products,
    (SELECT count(*) FROM orders)    AS orders,
    (SELECT round(sum(amount))::int FROM orders WHERE status='completed') AS completed_revenue;
