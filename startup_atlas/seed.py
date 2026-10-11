"""Bundled SAMPLE data so the site runs offline (see sample.json). Figures are
rounded, approximate public-knowledge values for demo purposes only - run the
live ingest (--yc --enrich) for real data."""
import json
import os

with open(os.path.join(os.path.dirname(__file__), "sample.json")) as f:
    SAMPLE = json.load(f)
