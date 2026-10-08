"""
Builds a SQLite database (data/zomato.db) from the generated CSVs, so the
.sql files in this folder can be run against a real database rather than
just pandas. SQLite is used for portability (no server setup needed) — the
same queries will run unmodified against PostgreSQL if the project is
deployed with a real database later.

Run this after src/generate_data.py (it must produce zomato_data.csv and
fact_activity.csv first).
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "zomato.db")

users = pd.read_csv(os.path.join(DATA_DIR, "zomato_data.csv"), parse_dates=["join_date"])
activity = pd.read_csv(os.path.join(DATA_DIR, "fact_activity.csv"), parse_dates=["month"])

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)

users.to_sql("dim_users", conn, index=False)
activity.to_sql("fact_activity", conn, index=False)

conn.execute("CREATE INDEX idx_activity_user ON fact_activity(user_id)")
conn.execute("CREATE INDEX idx_activity_month ON fact_activity(months_since_join)")

conn.commit()

# Quick verification the load actually worked
n_users = conn.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
n_activity = conn.execute("SELECT COUNT(*) FROM fact_activity").fetchone()[0]
print(f"Database built at {DB_PATH}")
print(f"  dim_users: {n_users} rows")
print(f"  fact_activity: {n_activity} rows")

conn.close()
