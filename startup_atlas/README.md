# Startup Atlas

Aggregates startups from top accelerators, clusters them by domain / use-case
(TF-IDF + spherical k-means, no ML dependencies), and shows funding, founding
year, team size and more. Separate from the LLM Council app.

```bash
pip install fastapi "uvicorn[standard]" httpx
python -m startup_atlas.ingest --sample          # offline demo data (approximate figures)
python -m startup_atlas.ingest --yc --enrich     # live: all public YC companies + Wikidata
python -m startup_atlas.ingest --file techstars.csv --accelerator Techstars   # any other accelerator
python -m startup_atlas.app                      # http://localhost:8002
```

`--file` accepts CSV/JSON with flexible column names (name, website, tagline,
description, tags, industry, country, city, founded, batch, raised, stage,
valuation, revenue). Re-run ingest any time; clusters are recomputed.

## Data sources and honest limits
| Field | Source |
|---|---|
| Company, batch, tags, one-liner, location, team size, status | YC directory (via yc-oss mirror) |
| Founded year | Wikidata (notable companies only); otherwise estimated from batch year and flagged in Sources |
| Capital raised, last round, valuation | Crunchbase API (set `CRUNCHBASE_API_KEY`, paid) or your CSV import |
| Revenue | Only from your own imports; no free reliable source exists |

There is no free, complete funding dataset for early-stage companies, so many
cards will show "—" until you add a funding source. Each record lists where its
numbers came from. Check each site's terms before scraping anything beyond the
connectors provided.
