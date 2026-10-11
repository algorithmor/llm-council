"""FastAPI app: JSON API + static single-page frontend.
Run from the repo root:  python -m startup_atlas.app   (port 8002)"""
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from . import cluster, db

app = FastAPI(title="Startup Atlas")
STATIC = os.path.join(os.path.dirname(__file__), "static")

SORTS = {
    "raised": "capital_raised_usd DESC", "founded": "founded_year DESC",
    "name": "name COLLATE NOCASE ASC", "valuation": "valuation_usd DESC",
    "team": "team_size DESC",
}


@app.get("/api/startups")
def list_startups(
    q: str = "", accelerator: str = "", cluster_id: int | None = None,
    country: str = "", industry: str = "", min_raised: float | None = None,
    founded_from: int | None = None, founded_to: int | None = None,
    sort: str = "raised", limit: int = Query(60, le=500), offset: int = 0,
):
    where, args = [], []
    if q:
        where.append("(name LIKE ? OR one_liner LIKE ? OR description LIKE ? OR tags LIKE ?)")
        args += [f"%{q}%"] * 4
    for col, val in (("accelerator", accelerator), ("country", country), ("industry", industry)):
        if val:
            where.append(f"{col}=?")
            args.append(val)
    if cluster_id is not None:
        where.append("cluster_id=?"); args.append(cluster_id)
    if min_raised is not None:
        where.append("capital_raised_usd>=?"); args.append(min_raised)
    if founded_from:
        where.append("founded_year>=?"); args.append(founded_from)
    if founded_to:
        where.append("founded_year<=?"); args.append(founded_to)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    order = SORTS.get(sort, SORTS["raised"])
    # NULLs last regardless of direction
    order = f"{order.split()[0]} IS NULL, {order}"
    conn = db.connect()
    total = conn.execute(f"SELECT COUNT(*) FROM startups {w}", args).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM startups {w} ORDER BY {order} LIMIT ? OFFSET ?", args + [limit, offset]
    ).fetchall()
    return {"total": total, "items": [db.row_to_dict(r) for r in rows]}


@app.get("/api/startups/{sid}")
def get_startup(sid: str):
    conn = db.connect()
    row = conn.execute("SELECT * FROM startups WHERE id=?", (sid,)).fetchone()
    if not row:
        raise HTTPException(404, "not found")
    out = db.row_to_dict(row)
    sims = cluster.similar(conn, sid)
    by_id = {r["id"]: db.row_to_dict(r) for r in conn.execute(
        f"SELECT * FROM startups WHERE id IN ({','.join('?' * len(sims))})", [i for i, _ in sims]
    ).fetchall()} if sims else {}
    out["similar"] = [{**by_id[i], "score": s} for i, s in sims if i in by_id]
    return out


@app.get("/api/clusters")
def clusters():
    conn = db.connect()
    rows = conn.execute(
        "SELECT c.*, SUM(s.capital_raised_usd) AS total_raised FROM clusters c "
        "LEFT JOIN startups s ON s.cluster_id=c.id GROUP BY c.id ORDER BY c.size DESC"
    ).fetchall()
    return [db.row_to_dict(r) for r in rows]


@app.get("/api/facets")
def facets():
    conn = db.connect()
    col = lambda c: [r[0] for r in conn.execute(
        f"SELECT {c} FROM startups WHERE {c} IS NOT NULL AND {c}!='' "
        f"GROUP BY {c} ORDER BY COUNT(*) DESC LIMIT 100")]
    return {"accelerators": col("accelerator"), "countries": col("country"),
            "industries": col("industry")}


@app.get("/api/stats")
def stats():
    conn = db.connect()
    one = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "companies": one("SELECT COUNT(*) FROM startups"),
        "accelerators": one("SELECT COUNT(DISTINCT accelerator) FROM startups"),
        "countries": one("SELECT COUNT(DISTINCT country) FROM startups"),
        "clusters": one("SELECT COUNT(*) FROM clusters"),
        "total_raised": one("SELECT COALESCE(SUM(capital_raised_usd),0) FROM startups"),
        "with_funding": one("SELECT COUNT(*) FROM startups WHERE capital_raised_usd IS NOT NULL"),
        "has_sample": bool(one("SELECT COUNT(*) FROM startups WHERE is_sample=1")),
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    import uvicorn
    conn = db.connect()
    if conn.execute("SELECT COUNT(*) FROM startups").fetchone()[0] == 0:
        print("Empty database - loading bundled sample data (run ingest --yc for real data)")
        from .seed import SAMPLE
        for r in SAMPLE:
            db.upsert(conn, dict(r, is_sample=1))
        conn.commit()
        cluster.run(conn)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8002)))
