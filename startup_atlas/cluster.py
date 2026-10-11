"""Pure-Python TF-IDF + spherical k-means clustering of startups by
domain / use-case text. No third-party dependencies."""
import json
import math
import random
import re
from collections import Counter

STOP = set("""a an the and or of for to in on with by from as at is are be that this it its our
we you your their they them his her has have had can will more most new use using used via
make makes help helps helping build builds built platform company companies startup startups
solution solutions software service services product products provide provides providing
based all any one two than into over out up how what who which also not no per every each
business businesses customers customer users user teams team people world modern next
generation first leading simple easy better faster smarter enable enables""".split())

TOKEN = re.compile(r"[a-z][a-z0-9\+\-]{1,}")


def tokenize(text):
    toks = [t for t in TOKEN.findall((text or "").lower()) if t not in STOP]
    # light stemming so "payments"/"payment" and "robots"/"robot" collapse
    out = []
    for t in toks:
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def doc_text(s):
    # Tags and industry are strong, curated signals: weight them x3.
    tags = " ".join(s.get("tags") or [])
    return " ".join([
        s.get("one_liner") or "", s.get("description") or "",
        (tags + " ") * 3, ((s.get("industry") or "") + " ") * 3,
    ])


def build_vectors(startups):
    docs = [Counter(tokenize(doc_text(s))) for s in startups]
    df = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs)
    idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items() if c >= 2 or n < 50}
    vecs = []
    for d in docs:
        v = {t: (1 + math.log(c)) * idf[t] for t, c in d.items() if t in idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs


def dot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def normalize(v):
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {t: x / norm for t, x in v.items()}


def kmeans(vecs, k, iters=15, seed=7):
    rng = random.Random(seed)
    # k-means++ style seeding
    centers = [vecs[rng.randrange(len(vecs))]]
    while len(centers) < k:
        dists = [max(1e-9, 1 - max(dot(v, c) for c in centers)) ** 2 for v in vecs]
        r, acc = rng.random() * sum(dists), 0
        for v, d in zip(vecs, dists):
            acc += d
            if acc >= r:
                centers.append(v)
                break
        else:
            centers.append(vecs[-1])
    assign = [0] * len(vecs)
    for _ in range(iters):
        changed = False
        for i, v in enumerate(vecs):
            best = max(range(k), key=lambda c: dot(v, centers[c]))
            if best != assign[i]:
                assign[i], changed = best, True
        sums = [Counter() for _ in range(k)]
        for a, v in zip(assign, vecs):
            sums[a].update(v)
        centers = [normalize(dict(s)) if s else centers[i] for i, s in enumerate(sums)]
        if not changed:
            break
    return assign, centers


def label_for(center, global_df_terms):
    top = sorted(center.items(), key=lambda kv: -kv[1])[:8]
    terms = [t for t, _ in top]
    return " · ".join(t.title() for t in terms[:3]), terms


def run(conn):
    rows = conn.execute(
        "SELECT id, one_liner, description, tags, industry FROM startups"
    ).fetchall()
    if not rows:
        return 0
    startups = [
        {**dict(r), "tags": json.loads(r["tags"] or "[]")} for r in rows
    ]
    vecs = build_vectors(startups)
    n = len(startups)
    k = max(2, min(40, round(math.sqrt(n / 2)))) if n > 3 else 1
    k = min(k, n)
    assign, centers = kmeans(vecs, k)

    # Drop empty clusters and renumber.
    used = sorted(set(assign))
    remap = {old: new for new, old in enumerate(used)}
    conn.execute("DELETE FROM clusters")
    for old in used:
        members = sum(1 for a in assign if a == old)
        label, terms = label_for(centers[old], None)
        conn.execute(
            "INSERT INTO clusters (id, label, keywords, size) VALUES (?,?,?,?)",
            (remap[old], label, json.dumps(terms), members),
        )
    for s, a in zip(startups, assign):
        conn.execute("UPDATE startups SET cluster_id=? WHERE id=?", (remap[a], s["id"]))
    conn.commit()
    return len(used)


def similar(conn, startup_id, limit=6):
    """Nearest neighbours by cosine similarity (computed on demand)."""
    rows = conn.execute(
        "SELECT id, one_liner, description, tags, industry FROM startups"
    ).fetchall()
    startups = [{**dict(r), "tags": json.loads(r["tags"] or "[]")} for r in rows]
    idx = next((i for i, s in enumerate(startups) if s["id"] == startup_id), None)
    if idx is None:
        return []
    vecs = build_vectors(startups)
    scored = sorted(
        ((dot(vecs[idx], v), s["id"]) for i, (v, s) in enumerate(zip(vecs, startups)) if i != idx),
        reverse=True,
    )
    return [(i, round(sc, 3)) for sc, i in scored[:limit] if sc > 0]
