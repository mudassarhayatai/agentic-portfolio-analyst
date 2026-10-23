"""
Shared pytest fixture for the whole test suite. Every test gets its OWN
fresh temp database (pytest's tmp_path is unique per test) -- no shared
state, no test-ordering dependencies, and nothing here ever touches your
real portfolio.db.
"""

import sqlite3

import pytest

SCHEMA_AND_SEED_DATA = """
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    source_document_id INTEGER NOT NULL,
    confirmation_id INTEGER,
    trade_date DATE NOT NULL,
    settlement_date DATE,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    gross_value REAL,
    commission REAL DEFAULT 0,
    taxes REAL DEFAULT 0,
    other_charges REAL DEFAULT 0,
    net_value REAL,
    transaction_ref TEXT,
    source_page INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dividends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    payment_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    security TEXT,
    num_securities REAL NOT NULL,
    net_dividend REAL NOT NULL,
    UNIQUE(payment_date, symbol, num_securities, net_dividend)
);

CREATE TABLE market_price (
    symbol TEXT,
    price REAL,
    fetched_date TEXT
);

-- HBL: simple current holding, two BUYs, never sold
INSERT INTO transactions (source_document_id, trade_date, symbol, action, quantity, price, net_value) VALUES
    (1, '2023-01-15', 'HBL', 'BUY', 100, 120.0, 12050.0),
    (1, '2023-06-10', 'HBL', 'BUY', 200, 135.0, 27050.0);

-- ENGRO: fully exited -- net zero, must not appear as a current holding
INSERT INTO transactions (source_document_id, trade_date, symbol, action, quantity, price, net_value) VALUES
    (1, '2023-02-01', 'ENGRO', 'BUY', 50, 300.0, 15050.0),
    (1, '2023-09-01', 'ENGRO', 'SELL', 50, 320.0, 15950.0);

-- LUCK: the real FIFO test -- SELL of 150 spans both BUY lots
INSERT INTO transactions (source_document_id, trade_date, symbol, action, quantity, price, net_value) VALUES
    (1, '2023-01-01', 'LUCK', 'BUY', 100, 50.0, 5050.0),
    (1, '2023-03-01', 'LUCK', 'BUY', 100, 60.0, 6050.0),
    (1, '2023-05-01', 'LUCK', 'SELL', 150, 70.0, 10450.0);

-- OVERSOLD: a SELL with no BUY lots at all -- must warn, not crash
INSERT INTO transactions (source_document_id, trade_date, symbol, action, quantity, price, net_value) VALUES
    (1, '2023-04-01', 'OVERSOLD', 'SELL', 100, 10.0, 950.0);

INSERT INTO dividends (payment_date, symbol, security, num_securities, net_dividend) VALUES
    ('2023-08-15', 'HBL', 'HBL', 100, 400.0),
    ('2024-02-15', 'HBL', 'HBL', 300, 1200.0);

INSERT INTO market_price (symbol, price, fetched_date) VALUES
    ('HBL', 145.5, '2025-01-10'),
    ('HBL', 148.0, '2025-01-15');
"""


@pytest.fixture()
def db_path(tmp_path) -> str:
    path = str(tmp_path / "test_portfolio.db")
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_AND_SEED_DATA)
    conn.commit()
    conn.close()
    return path