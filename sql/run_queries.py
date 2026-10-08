"""
Runs the .sql files in this folder against data/zomato.db and prints the
results — proof the queries actually execute against a real database, not
just syntactically plausible SQL sitting unused in a file.

Usage: python sql/run_queries.py
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "zomato.db")


def run(sql_filename, label):
    path = os.path.join(BASE_DIR, sql_filename)
    with open(path) as f:
        query = f.read()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"\n=== {label} ({sql_filename}) ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    cohort_df = run("cohort_retention.sql", "Cohort Retention")
    rfm_df = run("rfm_scoring.sql", "RFM Segmentation")

    print("\n=== Sanity checks ===")
    m0 = cohort_df[cohort_df["months_since_join"] == 0]["retention_pct"].mean()
    m6 = cohort_df[cohort_df["months_since_join"] == 6]["retention_pct"].mean()
    print(f"Avg retention % at month 0: {m0:.1f}  |  at month 6: {m6:.1f}  (should drop)")
    print(f"RFM segments found: {rfm_df['rfm_segment'].tolist()}")
