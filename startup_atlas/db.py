"""SQLite storage for startups and clusters."""
import json
import os
import sqlite3

DB_PATH = os.environ.get(
    "ATLAS_DB", os.path.join(os.path.dirname(__file__), "data", "atlas.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS startups (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  accelerator TEXT,
  batch TEXT,
  website TEXT,
  logo TEXT,
  one_liner TEXT,
  description TEXT,
  tags TEXT DEFAULT '[]',
  industry TEXT,
  country TEXT,
  city TEXT,
  founded_year INTEGER,
  team_size INTEGER,
  status TEXT,
  capital_raised_usd REAL,
  last_round TEXT,
  valuation_usd REAL,
  revenue_usd REAL,
  sources TEXT DEFAULT '[]',
  cluster_id INTEGER,
  is_sample INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS clusters (
  id INTEGER PRIMARY KEY,
  label TEXT,
  keywords TEXT,
  size INTEGER
);
CREATE INDEX IF NOT EXISTS idx_s_acc ON startups(accelerator);
CREATE INDEX IF NOT EXISTS idx_s_cluster ON startups(cluster_id);
"""

JSON_COLS = ("tags", "sources", "keywords")


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def row_to_dict(row):
    d = dict(row)
    for k in JSON_COLS:
        if k in d and isinstance(d[k], str):
            d[k] = json.loads(d[k] or "[]")
    return d


FIELDS = [
    "id", "name", "accelerator", "batch", "website", "logo", "one_liner",
    "description", "tags", "industry", "country", "city", "founded_year",
    "team_size", "status", "capital_raised_usd", "last_round", "valuation_usd",
    "revenue_usd", "sources", "is_sample",
]


def upsert(conn, rec):
    """Insert a record; on conflict only fill fields that are currently NULL
    (so enrichment never clobbers data from a more trusted source)."""
    rec = dict(rec)
    rec["tags"] = json.dumps(rec.get("tags") or [])
    rec["sources"] = json.dumps(rec.get("sources") or [])
    rec.setdefault("is_sample", 0)
    vals = [rec.get(f) for f in FIELDS]
    placeholders = ",".join("?" * len(FIELDS))
    updates = ",".join(
        f"{f}=COALESCE(startups.{f}, excluded.{f})" for f in FIELDS if f not in ("id", "sources", "tags")
    )
    conn.execute(
        f"INSERT INTO startups ({','.join(FIELDS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(id) DO UPDATE SET {updates}",
        vals,
    )
