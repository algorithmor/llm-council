"""CLI:  python -m startup_atlas.ingest [--sample] [--yc] [--file F --accelerator NAME]
                                         [--enrich] [--limit N]"""
import argparse

from . import cluster, db, sources
from .seed import SAMPLE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="load bundled sample data")
    ap.add_argument("--yc", action="store_true", help="fetch all YC companies")
    ap.add_argument("--file", help="CSV/JSON of another accelerator's companies")
    ap.add_argument("--accelerator", help="name for --file")
    ap.add_argument("--enrich", action="store_true", help="Wikidata (+Crunchbase if key set)")
    ap.add_argument("--limit", type=int, help="cap records enriched")
    a = ap.parse_args()

    records = []
    if a.sample:
        records += [dict(r, is_sample=1) for r in SAMPLE]
    if a.yc:
        records += sources.fetch_yc()
    if a.file:
        if not a.accelerator:
            ap.error("--file needs --accelerator")
        records += sources.import_file(a.file, a.accelerator)
    if not records:
        ap.error("nothing to ingest; pass --sample, --yc or --file")

    if a.enrich:
        real = [r for r in records if not r.get("is_sample")]
        print("wikidata enriched:", sources.enrich_wikidata(real, a.limit))
        print("crunchbase enriched:", sources.enrich_crunchbase(real, a.limit))
    for r in records:  # estimated fallback so every row has a 'founded' value
        if not r.get("founded_year") and r.get("_batch_year"):
            r["founded_year"] = r["_batch_year"]
            r.setdefault("sources", []).append({"name": "estimate: founded≈batch year", "url": None})

    conn = db.connect()
    for r in records:
        db.upsert(conn, {k: v for k, v in r.items() if not k.startswith("_")})
    conn.commit()
    print(f"ingested {len(records)} records into {cluster.run(conn)} clusters")


if __name__ == "__main__":
    main()
