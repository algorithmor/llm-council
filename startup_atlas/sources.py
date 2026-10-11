"""Data connectors.

* ``fetch_yc``          – every public YC company, via the community-maintained
                          yc-oss mirror of YC's directory (no API key).
* ``import_file``       – CSV / JSON from any other accelerator (Techstars, 500
                          Global, Antler, EF, Seedcamp, ...). Column names are
                          matched flexibly, see ``ALIASES``.
* ``enrich_wikidata``   – founded year / HQ country for notable companies (free).
* ``enrich_crunchbase`` – funding totals, last round, valuation, if
                          CRUNCHBASE_API_KEY is set (paid API).
"""
import csv
import json
import os
import re
import time

import httpx

YC_URL = os.environ.get("YC_URL", "https://yc-oss.github.io/api/companies/all.json")
UA = {"User-Agent": "startup-atlas/0.1 (research aggregator)"}


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")


def batch_year(batch):
    m = re.search(r"(19|20)\d{2}", batch or "")
    if m:
        return int(m.group())
    m = re.search(r"\b([WSFX])(\d{2})\b", batch or "")  # e.g. W21
    return 2000 + int(m.group(2)) if m else None


def country_of(location):
    parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    return parts[-1] if parts else None


# ---------------------------------------------------------------- Y Combinator
def fetch_yc(client=None):
    client = client or httpx.Client(headers=UA, timeout=60, follow_redirects=True)
    data = client.get(YC_URL).raise_for_status().json()
    out = []
    for c in data:
        loc = c.get("all_locations") or ""
        first_loc = loc.split(";")[0]
        year = batch_year(c.get("batch"))
        tags = list(c.get("tags") or [])
        if c.get("subindustry"):
            tags.append(c["subindustry"].split("->")[-1].strip())
        out.append({
            "id": f"yc-{c.get('slug') or slugify(c['name'])}",
            "name": c["name"],
            "accelerator": "Y Combinator",
            "batch": c.get("batch"),
            "website": c.get("website"),
            "logo": c.get("small_logo_thumb_url"),
            "one_liner": c.get("one_liner"),
            "description": c.get("long_description"),
            "tags": tags,
            "industry": c.get("industry"),
            "country": country_of(first_loc),
            "city": first_loc.split(",")[0].strip() or None,
            "founded_year": None,
            "team_size": c.get("team_size"),
            "status": c.get("status"),
            "sources": [{"name": "Y Combinator directory", "url": c.get("url")}],
            "_batch_year": year,
        })
    return out


# ------------------------------------------------------------- generic importer
ALIASES = {
    "name": ["name", "company", "company_name", "startup"],
    "website": ["website", "url", "domain", "homepage"],
    "one_liner": ["one_liner", "tagline", "short_description", "pitch"],
    "description": ["description", "long_description", "about", "summary"],
    "tags": ["tags", "keywords", "categories", "sectors"],
    "industry": ["industry", "sector", "vertical", "category"],
    "country": ["country", "hq_country"],
    "city": ["city", "hq_city", "location"],
    "founded_year": ["founded_year", "founded", "year_founded"],
    "batch": ["batch", "cohort", "class", "year"],
    "team_size": ["team_size", "employees", "headcount"],
    "status": ["status", "stage_status"],
    "capital_raised_usd": ["capital_raised_usd", "total_funding", "raised", "funding_total_usd"],
    "last_round": ["last_round", "stage", "last_funding_type"],
    "valuation_usd": ["valuation_usd", "valuation"],
    "revenue_usd": ["revenue_usd", "revenue", "arr"],
}


def _num(v):
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("$", "").upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(s[-1:], 1)
    try:
        return float(s[:-1] if mult != 1 else s) * mult
    except ValueError:
        return None


def import_file(path, accelerator):
    if path.endswith(".json"):
        with open(path) as f:
            rows = json.load(f)
    else:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        low = {k.strip().lower().replace(" ", "_"): v for k, v in r.items()}
        rec = {}
        for field, names in ALIASES.items():
            for n in names:
                if low.get(n) not in (None, ""):
                    rec[field] = low[n]
                    break
        if "name" not in rec:
            continue
        tags = rec.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in re.split(r"[;,|]", tags) if t.strip()]
        for k in ("capital_raised_usd", "valuation_usd", "revenue_usd"):
            rec[k] = _num(rec.get(k))
        for k in ("founded_year", "team_size"):
            rec[k] = int(_num(rec[k])) if _num(rec.get(k)) else None
        rec.update(
            id=f"{slugify(accelerator)}-{slugify(rec['name'])}",
            accelerator=accelerator,
            tags=tags,
            sources=[{"name": f"{accelerator} import", "url": None}],
        )
        out.append(rec)
    return out


# ------------------------------------------------------------------ enrichment
WD_API = "https://www.wikidata.org/w/api.php"


def enrich_wikidata(records, limit=None, delay=0.2):
    """Fill founded_year / country for companies that have a Wikidata entry.
    Matching is by exact (case-insensitive) label to avoid false positives."""
    client = httpx.Client(headers=UA, timeout=30)
    n = 0
    for rec in records[:limit]:
        if rec.get("founded_year"):
            continue
        try:
            hits = client.get(WD_API, params={
                "action": "wbsearchentities", "search": rec["name"],
                "language": "en", "type": "item", "limit": 5, "format": "json",
            }).json().get("search", [])
            hit = next((h for h in hits if h.get("label", "").lower() == rec["name"].lower()
                        and re.search(r"compan|startup|software|service|platform|corporation|provider",
                                      h.get("description", ""), re.I)), None)
            if not hit:
                continue
            ent = client.get(WD_API, params={
                "action": "wbgetentities", "ids": hit["id"], "props": "claims",
                "format": "json"}).json()["entities"][hit["id"]]["claims"]
            inc = ent.get("P571")
            if inc:
                t = inc[0]["mainsnak"]["datavalue"]["value"]["time"]  # +2010-01-01T..
                rec["founded_year"] = int(t[1:5])
                rec.setdefault("sources", []).append(
                    {"name": "Wikidata", "url": f"https://www.wikidata.org/wiki/{hit['id']}"})
                n += 1
        except Exception:
            continue
        time.sleep(delay)
    return n


def enrich_crunchbase(records, limit=None):
    key = os.environ.get("CRUNCHBASE_API_KEY")
    if not key:
        return 0
    client = httpx.Client(headers=UA, timeout=30)
    n = 0
    for rec in records[:limit]:
        if rec.get("capital_raised_usd"):
            continue
        permalink = slugify(rec["name"])
        try:
            r = client.get(
                f"https://api.crunchbase.com/api/v4/entities/organizations/{permalink}",
                params={"user_key": key, "field_ids":
                        "funding_total,last_funding_type,founded_on,num_employees_enum,valuation"})
            if r.status_code != 200:
                continue
            p = r.json().get("properties", {})
            if p.get("funding_total"):
                rec["capital_raised_usd"] = p["funding_total"].get("value_usd")
            rec["last_round"] = p.get("last_funding_type") or rec.get("last_round")
            if p.get("founded_on"):
                rec["founded_year"] = int(p["founded_on"]["value"][:4])
            if p.get("valuation"):
                rec["valuation_usd"] = p["valuation"].get("value_usd")
            rec.setdefault("sources", []).append(
                {"name": "Crunchbase", "url": f"https://www.crunchbase.com/organization/{permalink}"})
            n += 1
        except Exception:
            continue
    return n
