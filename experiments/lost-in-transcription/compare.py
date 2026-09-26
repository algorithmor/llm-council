#!/usr/bin/env python3
"""Compare ASR systems on a labelled set with the official Lost in Transcription metric.

Every prediction CSV (columns ``audio_filename,transcript``, i.e. submission format) is scored
against a reference CSV with the organisers' own normaliser, imported from ``score.py`` in
https://github.com/drivendataorg/lost-in-transcription-runtime, and the same word alignment
jiwer uses, so the WER printed here is the WER the platform would report on that set.

On top of the raw WER it reports what you need to choose between approaches on a small set:

* a bootstrap 95% confidence interval for each system,
* the *paired* difference to the best system (or ``--baseline``), with its own interval and the
  share of bootstrap resamples in which the system beats the baseline,
* substitution / deletion / insertion rates and how much WER is caused by casing alone,
* error rates on reference fillers, Spanish words, English words, and the first word after a
  language switch (a PIER-style code-switching diagnostic),
* the oracle WER (best system picked per clip) and word-level ROVER / MBR combinations,
* WER broken down by any metadata column (speaker, duration bucket, ...).

Example::

    python compare.py --ref dev_gold.csv --scorer ../lost-in-transcription-runtime/score.py \\
        preds/*.csv --combine all --out-dir results/dev
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

FILLERS = {
    "uh", "um", "uhm", "umm", "er", "erm", "eh", "ah", "ahh", "oh", "mm", "mmm", "hm", "hmm",
    "mhm", "uh-huh", "huh", "em", "ehm",
}
SCORER_CANDIDATES = (  # a runtime checkout here, or next to this repo
    "lost-in-transcription-runtime/score.py",
    "../lost-in-transcription-runtime/score.py",
    "../../lost-in-transcription-runtime/score.py",
    "../../../lost-in-transcription-runtime/score.py",
)
_MULTISPACE = re.compile(r"\s\s+")
_SPANISH_CHARS = set("áéíóúüñ")


# --------------------------------------------------------------------------- scoring primitives

def find_scorer(explicit: Optional[str]) -> Path:
    """Locate the official score.py (argument, $LIT_SCORE_PY, or a sibling runtime checkout)."""
    here = Path(__file__).resolve().parent
    candidates = [explicit, os.environ.get("LIT_SCORE_PY")]
    candidates += [c for c in SCORER_CANDIDATES] + [str(here / c) for c in SCORER_CANDIDATES]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    sys.exit(
        "Cannot find the official score.py. Clone "
        "https://github.com/drivendataorg/lost-in-transcription-runtime and pass "
        "--scorer path/to/score.py (or set LIT_SCORE_PY)."
    )


def load_normalizer(path: Path) -> Callable[[str], str]:
    """Import ``normalize_text`` from the official scorer so scoring can never drift from it."""
    spec = importlib.util.spec_from_file_location("lit_official_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_text


def tokenize(text: str) -> list[str]:
    """Split a normalised transcript into words exactly like jiwer's default WER transform."""
    return [w for w in _MULTISPACE.sub(" ", text).strip().split(" ") if w]


def align(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int, list[tuple[Optional[int], Optional[int]]]]:
    """Levenshtein alignment as computed by jiwer (rapidfuzz opcodes over word ids).

    Returns (substitutions, deletions, insertions, pairs) where pairs lists
    (ref_index or None, hyp_index or None) in order.
    """
    vocab: dict[str, int] = {}
    r = [vocab.setdefault(w, len(vocab)) for w in ref]
    h = [vocab.setdefault(w, len(vocab)) for w in hyp]
    subs = dels = ins = 0
    pairs: list[tuple[Optional[int], Optional[int]]] = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(r, h):
        if tag == "equal":
            pairs.extend((i1 + k, j1 + k) for k in range(i2 - i1))
        elif tag == "replace":
            subs += i2 - i1  # jiwer's convention; rapidfuzz replace blocks are 1:1
            pairs.extend((i1 + k, j1 + k) for k in range(i2 - i1))
        elif tag == "delete":
            dels += i2 - i1
            pairs.extend((i1 + k, None) for k in range(i2 - i1))
        elif tag == "insert":
            ins += j2 - j1
            pairs.extend((None, j1 + k) for k in range(j2 - j1))
    return subs, dels, ins, pairs


def read_transcripts(path: Path, allow_missing_col: bool = False) -> pd.Series:
    """Read a submission-format CSV the way score.py does (default NA parsing, then fillna(""))."""
    df = pd.read_csv(path)
    if "audio_filename" not in df.columns or ("transcript" not in df.columns and not allow_missing_col):
        sys.exit(f"{path}: expected columns audio_filename,transcript; got {list(df.columns)}")
    dupes = df["audio_filename"][df["audio_filename"].duplicated()]
    if len(dupes):
        sys.exit(f"{path}: duplicate audio_filename rows, e.g. {dupes.iloc[0]}")
    return df.set_index("audio_filename")["transcript"].fillna("").astype(str)


# --------------------------------------------------------------------------- language tagging

def make_lang_tagger(lexicon_path: Optional[str], margin: float = 1.0) -> Optional[Callable[[str], str]]:
    """Return token -> 'es' | 'en' | 'amb' | 'filler' | 'num' | 'unk', or None if no resource.

    Uses an optional lexicon CSV (columns word,lang with lang in {es,en,amb}) first, then word
    frequencies from the ``wordfreq`` package: a word is Spanish (English) when its Zipf frequency
    in Spanish (English) exceeds the other by ``margin``; otherwise it is ambiguous.
    """
    lexicon: dict[str, str] = {}
    if lexicon_path:
        lex = pd.read_csv(lexicon_path)
        lexicon = {str(w).lower(): str(l) for w, l in zip(lex.iloc[:, 0], lex.iloc[:, 1])}
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        zipf_frequency = None
    if not lexicon and zipf_frequency is None:
        return None
    cache: dict[str, str] = {}

    def tag(token: str) -> str:
        word = token.lower().strip(".'-")
        if word in cache:
            return cache[word]
        if not word:
            lang = "unk"
        elif word in FILLERS:
            lang = "filler"
        elif any(ch.isdigit() for ch in word):
            lang = "num"
        elif word in lexicon:
            lang = lexicon[word]
        elif _SPANISH_CHARS & set(word):
            lang = "es"
        elif zipf_frequency is None:
            lang = "unk"
        else:
            es, en = zipf_frequency(word, "es"), zipf_frequency(word, "en")
            if es == 0 and en == 0:
                lang = "unk"
            elif es - en >= margin:
                lang = "es"
            elif en - es >= margin:
                lang = "en"
            else:
                lang = "amb"
        cache[word] = lang
        return lang

    return tag


# --------------------------------------------------------------------------- per-system scoring

@dataclass
class SystemScore:
    name: str
    errors: np.ndarray  # per clip S+D+I
    subs: np.ndarray
    dels: np.ndarray
    ins: np.ndarray
    errors_lowercase: np.ndarray  # per clip errors when casing is ignored
    ref_status: list[list[str]]  # per clip, per reference word: "ok" | "sub" | "del"
    hyp_tokens: list[list[str]]
    raw: list[str] = field(default_factory=list)  # un-normalised hypothesis text per clip


def score_system(name: str, hyp_texts: Sequence[str], ref_tokens: Sequence[list[str]],
                 normalize: Callable[[str], str]) -> SystemScore:
    n = len(ref_tokens)
    errors, subs, dels, ins, errors_lc = (np.zeros(n, dtype=np.int64) for _ in range(5))
    statuses, hyp_tokens = [], []
    for i, (text, ref) in enumerate(zip(hyp_texts, ref_tokens)):
        hyp = tokenize(normalize(text))
        s, d, ins_i, pairs = align(ref, hyp)
        status = ["ok"] * len(ref)
        for r_idx, h_idx in pairs:
            if r_idx is not None:
                if h_idx is None:
                    status[r_idx] = "del"
                elif ref[r_idx] != hyp[h_idx]:
                    status[r_idx] = "sub"
        s_lc, d_lc, i_lc, _ = align([w.lower() for w in ref], [w.lower() for w in hyp])
        subs[i], dels[i], ins[i] = s, d, ins_i
        errors[i] = s + d + ins_i
        errors_lc[i] = s_lc + d_lc + i_lc
        statuses.append(status)
        hyp_tokens.append(hyp)
    return SystemScore(name, errors, subs, dels, ins, errors_lc, statuses, hyp_tokens, list(hyp_texts))


def word_class_error(score: SystemScore, ref_classes: Sequence[list[str]], wanted: str) -> tuple[float, int]:
    """Share of reference words of a class that were substituted or deleted (insertions excluded)."""
    total = wrong = 0
    for status, classes in zip(score.ref_status, ref_classes):
        for st, c in zip(status, classes):
            if c == wanted:
                total += 1
                wrong += st != "ok"
    return (wrong / total if total else float("nan")), total


# --------------------------------------------------------------------------- combination

def _pick(votes: dict, default):
    best = max(votes.values())
    tied = [k for k, v in votes.items() if v == best]
    if default in tied:
        return default
    return sorted(tied, key=lambda k: (k is None, str(k)))[0]


def medoid(hyps: Sequence[list[str]], weights: Sequence[float]) -> int:
    """Index of the hypothesis with the lowest weighted edit distance to the others (MBR)."""
    costs = []
    for i, a in enumerate(hyps):
        costs.append(sum(w * Levenshtein.distance(a, b) for j, (b, w) in enumerate(zip(hyps, weights)) if j != i))
    return int(np.argmin(costs))  # ties go to the earlier system


def rover(hyps: Sequence[list[str]], weights: Sequence[float], pivot: int) -> list[str]:
    """Word-level voting over hypotheses aligned to a pivot (ties keep the pivot's choice).

    Each pivot word position is voted on (a word or deletion) and each gap between pivot words
    is voted on (an inserted phrase or nothing).
    """
    base = hyps[pivot]
    n = len(base)
    slot_votes = [defaultdict(float) for _ in range(n)]
    gap_votes = [defaultdict(float) for _ in range(n + 1)]
    for k, (hyp, w) in enumerate(zip(hyps, weights)):
        covered: list[Optional[str]] = [None] * n
        inserted: dict[int, list[str]] = defaultdict(list)
        if k == pivot:
            covered = list(base)
        else:
            last = -1
            _, _, _, pairs = align(base, hyp)
            for p_idx, h_idx in pairs:
                if p_idx is None:
                    inserted[last + 1].append(hyp[h_idx])
                else:
                    covered[p_idx] = hyp[h_idx] if h_idx is not None else None
                    last = p_idx
        for i in range(n):
            slot_votes[i][covered[i]] += w
        for g in range(n + 1):
            gap_votes[g][tuple(inserted.get(g, ()))] += w
    out: list[str] = []
    for g in range(n + 1):
        out.extend(_pick(gap_votes[g], ()))
        if g < n:
            word = _pick(slot_votes[g], base[g])
            if word is not None:
                out.append(word)
    return out


def combine(scores: Sequence[SystemScore], weights: Sequence[float], method: str) -> list[str]:
    """Per clip combined hypothesis text for ``method`` in {"rover", "mbr"}."""
    texts = []
    for clip in range(len(scores[0].hyp_tokens)):
        hyps = [s.hyp_tokens[clip] for s in scores]
        pivot = medoid(hyps, weights)
        if method == "mbr":
            texts.append(scores[pivot].raw[clip])
        else:
            texts.append(" ".join(rover(hyps, weights, pivot)))
    return texts


# --------------------------------------------------------------------------- statistics

def bootstrap(err: np.ndarray, nref: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Corpus WER for every bootstrap resample (rows of idx index units: clips or groups)."""
    return err[idx].sum(axis=1) / np.maximum(nref[idx].sum(axis=1), 1)


def group_totals(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    uniq, inv = np.unique(groups, return_inverse=True)
    out = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(out, inv, values)
    return out


def fmt(x: float, digits: int = 4, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def parse_by(spec: str) -> tuple[str, Optional[list[float]]]:
    """``col`` for categorical breakdowns, ``col:10,30,60`` to bucket a numeric column."""
    if ":" in spec:
        col, edges = spec.split(":", 1)
        return col, [float(e) for e in edges.split(",") if e]
    return spec, None


def bucket(values: pd.Series, edges: list[float]) -> pd.Series:
    labels = []
    bounds = [-np.inf] + edges + [np.inf]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if np.isinf(lo):
            labels.append(f"<{hi:g}")
        elif np.isinf(hi):
            labels.append(f">={lo:g}")
        else:
            labels.append(f"{lo:g}-{hi:g}")
    return pd.cut(values.astype(float), bins=bounds, labels=labels, right=False).astype(str)


# --------------------------------------------------------------------------- main

def parse_pred_arg(arg: str) -> tuple[str, Path]:
    if "=" in arg and not Path(arg).exists():
        name, path = arg.split("=", 1)
        return name, Path(path)
    return Path(arg).stem, Path(arg)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("preds", nargs="+", help="prediction CSVs (submission format); NAME=PATH to rename")
    ap.add_argument("--ref", required=True, help="reference CSV with audio_filename,transcript")
    ap.add_argument("--scorer", help="path to the official score.py (default: search for a runtime checkout)")
    ap.add_argument("--baseline", help="system the paired deltas are computed against (default: best WER)")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples (0 to skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--meta", help="CSV with audio_filename plus metadata columns (speaker, duration_s, ...)")
    ap.add_argument("--cluster-col", help="meta column to resample as a unit (e.g. speaker) instead of clips")
    ap.add_argument("--by", action="append", default=[],
                    help="meta column to break WER down by; COL:10,30,60 buckets a numeric column")
    ap.add_argument("--combine", help="'all' or comma-separated systems to combine with ROVER and MBR")
    ap.add_argument("--weights", help="comma-separated NAME=WEIGHT votes for --combine (default 1)")
    ap.add_argument("--lang-lexicon", help="CSV word,lang (es/en/amb) overriding the wordfreq tagger")
    ap.add_argument("--no-lang", action="store_true", help="skip the Spanish/English breakdown")
    ap.add_argument("--allow-missing", action="store_true", help="score missing clips as empty transcripts")
    ap.add_argument("--out-dir", help="write summary.json, per_clip.csv and combined CSVs here")
    args = ap.parse_args(argv)

    normalize = load_normalizer(find_scorer(args.scorer))
    ref = read_transcripts(Path(args.ref)).sort_index()
    clips = list(ref.index)
    ref_tokens = [tokenize(normalize(t)) for t in ref.values]
    nref = np.array([len(t) for t in ref_tokens], dtype=np.int64)
    if nref.sum() == 0:
        sys.exit("Reference transcripts are empty after normalisation.")

    systems: list[SystemScore] = []
    for arg in args.preds:
        name, path = parse_pred_arg(arg)
        pred = read_transcripts(path)
        missing = [c for c in clips if c not in pred.index]
        if missing and not args.allow_missing:
            sys.exit(f"{path}: missing {len(missing)} of {len(clips)} reference clips (first: {missing[0]}); "
                     "use --allow-missing to score them as empty")
        if missing:
            print(f"warning: {name} has no prediction for {len(missing)} of {len(clips)} clips; "
                  "scored as empty transcripts", file=sys.stderr)
        texts = [pred.get(c, "") for c in clips]
        systems.append(score_system(name, texts, ref_tokens, normalize))
    names = [s.name for s in systems]
    if len(set(names)) != len(names):
        sys.exit(f"Duplicate system names: {names}; use NAME=PATH to disambiguate")
    inputs = list(systems)

    # Combinations are scored like any other system.
    if args.combine:
        chosen = names if args.combine == "all" else [c.strip() for c in args.combine.split(",")]
        unknown = set(chosen) - set(names)
        if unknown:
            sys.exit(f"--combine: unknown systems {sorted(unknown)}")
        weight_map = {}
        for item in (args.weights or "").split(","):
            if item:
                k, v = item.split("=")
                weight_map[k] = float(v)
        members = [s for s in systems if s.name in chosen]
        weights = [weight_map.get(s.name, 1.0) for s in members]
        if len(members) >= 2:
            label = ",".join(s.name for s in members)
            for method in ("rover", "mbr"):
                texts = combine(members, weights, method)
                systems.append(score_system(f"{method.upper()}({label})", texts, ref_tokens, normalize))

    wer = {s.name: s.errors.sum() / nref.sum() for s in systems}
    order = sorted(systems, key=lambda s: wer[s.name])
    baseline = args.baseline or order[0].name
    if baseline not in wer:
        sys.exit(f"--baseline {baseline} is not one of {list(wer)}")

    # Oracle: best single input system per clip (a floor for any selection-based combination).
    oracle = np.min(np.stack([s.errors for s in inputs]), axis=0)

    # Bootstrap over clips, or over clusters (speakers/sessions) when --cluster-col is given.
    meta = None
    if args.meta:
        meta = pd.read_csv(args.meta).drop_duplicates("audio_filename").set_index("audio_filename").reindex(clips)
    units_err = {s.name: s.errors.astype(np.float64) for s in systems}
    units_err["__oracle__"] = oracle.astype(np.float64)
    units_nref = nref.astype(np.float64)
    if args.cluster_col:
        if meta is None or args.cluster_col not in meta.columns:
            sys.exit("--cluster-col needs --meta with that column")
        groups = meta[args.cluster_col].fillna("unknown").astype(str).to_numpy()
        units_err = {k: group_totals(v, groups) for k, v in units_err.items()}
        units_nref = group_totals(units_nref, groups)
    stats: dict[str, dict] = {}
    if args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, len(units_nref), size=(args.bootstrap, len(units_nref)))
        boot = {k: bootstrap(v, units_nref, idx) for k, v in units_err.items()}
        for s in systems:
            delta = boot[s.name] - boot[baseline]
            stats[s.name] = {
                "ci": np.percentile(boot[s.name], [2.5, 97.5]).tolist(),
                "delta_ci": np.percentile(delta, [2.5, 97.5]).tolist(),
                "p_better": float(np.mean(delta < 0)),
            }
        stats["__oracle__"] = {"ci": np.percentile(boot["__oracle__"], [2.5, 97.5]).tolist()}

    # Diagnostics.
    ref_fill = [["filler" if w.lower().strip(".") in FILLERS else "word" for w in toks] for toks in ref_tokens]
    tagger = None if args.no_lang else make_lang_tagger(args.lang_lexicon)
    ref_lang, ref_switch = None, None
    if tagger:
        ref_lang = [[tagger(w) for w in toks] for toks in ref_tokens]
        ref_switch = []
        for langs in ref_lang:
            prev, marks = None, []
            for lang in langs:
                definite = lang in ("es", "en")
                marks.append("switch" if definite and prev is not None and lang != prev else "no")
                if definite:
                    prev = lang
            ref_switch.append(marks)

    total_ref = int(nref.sum())
    rows = []
    for s in order:
        row = {
            "system": s.name,
            "wer": float(wer[s.name]),
            "sub": float(s.subs.sum() / total_ref),
            "del": float(s.dels.sum() / total_ref),
            "ins": float(s.ins.sum() / total_ref),
            "wer_ignoring_case": float(s.errors_lowercase.sum() / total_ref),
            "filler_error": word_class_error(s, ref_fill, "filler")[0],
        }
        if tagger:
            row["es_error"] = word_class_error(s, ref_lang, "es")[0]
            row["en_error"] = word_class_error(s, ref_lang, "en")[0]
            row["switch_error"] = word_class_error(s, ref_switch, "switch")[0]
        row.update(stats.get(s.name, {}))
        rows.append(row)

    # ---- report
    print(f"Reference: {args.ref} | {len(clips)} clips | {total_ref} words"
          + (f" | bootstrap over {len(units_nref)} {args.cluster_col or 'clip'} units" if stats else ""))
    if tagger:
        counts = defaultdict(int)
        for langs in ref_lang:
            for lang in langs:
                counts[lang] += 1
        n_switch = sum(m == "switch" for marks in ref_switch for m in marks)
        print("Reference words: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
              + f" | switch points {n_switch}")
    print()
    print(f"| system | WER | 95% CI | Δ vs {baseline} | Δ 95% CI | P(beats {baseline}) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        ci = r.get("ci")
        dci = r.get("delta_ci")
        print(f"| {r['system']} | {fmt(r['wer'])} | {fmt(ci[0]) + '–' + fmt(ci[1]) if ci else '–'} | "
              f"{fmt(r['wer'] - wer[baseline], sign=True)} | "
              f"{fmt(dci[0], sign=True) + ' to ' + fmt(dci[1], sign=True) if dci and r['system'] != baseline else '–'} | "
              f"{fmt(r['p_better'], 2) if 'p_better' in r and r['system'] != baseline else '–'} |")
    oracle_wer = oracle.sum() / nref.sum()
    oci = stats.get("__oracle__", {}).get("ci")
    print(f"| oracle (best input per clip) | {fmt(oracle_wer)} | "
          f"{fmt(oci[0]) + '–' + fmt(oci[1]) if oci else '–'} | {fmt(oracle_wer - wer[baseline], sign=True)} | – | – |")
    print()
    head = "| system | sub | del | ins | WER ignoring case | filler error |"
    if tagger:
        head += " es-word error | en-word error | switch-word error |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for r in rows:
        line = (f"| {r['system']} | {fmt(r['sub'])} | {fmt(r['del'])} | {fmt(r['ins'])} | "
                f"{fmt(r['wer_ignoring_case'])} | {fmt(r['filler_error'], 3)} |")
        if tagger:
            line += f" {fmt(r['es_error'], 3)} | {fmt(r['en_error'], 3)} | {fmt(r['switch_error'], 3)} |"
        print(line)

    breakdowns = {}
    for spec in args.by:
        col, edges = parse_by(spec)
        if meta is None or col not in meta.columns:
            sys.exit(f"--by {col}: needs --meta with that column")
        values = bucket(meta[col], edges) if edges else meta[col].fillna("unknown").astype(str)
        levels = sorted(values.unique(), key=str)
        table = {}
        print()
        print("| system | " + " | ".join(f"{col}={lv} (n={int((values == lv).sum())})" for lv in levels) + " |")
        print("|---|" + "---|" * len(levels))
        for s in order:
            cells = []
            for lv in levels:
                mask = (values == lv).to_numpy()
                v = s.errors[mask].sum() / max(nref[mask].sum(), 1)
                table.setdefault(s.name, {})[lv] = float(v)
                cells.append(fmt(v))
            print(f"| {s.name} | " + " | ".join(cells) + " |")
        breakdowns[col] = table

    summary = {
        "reference": str(args.ref),
        "clips": len(clips),
        "reference_words": total_ref,
        "baseline": baseline,
        "systems": rows,
        "oracle_wer": float(oracle_wer),
        "oracle_ci": oci,
        "breakdowns": breakdowns,
    }
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        per_clip = pd.DataFrame({"audio_filename": clips, "ref_words": nref})
        for s in systems:
            per_clip[f"errors:{s.name}"] = s.errors
        per_clip["errors:oracle"] = oracle
        per_clip.to_csv(out / "per_clip.csv", index=False)
        for s in systems:
            if "(" in s.name:
                fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.name).strip("_") + ".csv"
                pd.DataFrame({"audio_filename": clips, "transcript": s.raw}).to_csv(out / fname, index=False)
        print(f"\nWrote {out}/summary.json, per_clip.csv" + (" and combined CSVs" if args.combine else ""))
    return summary


if __name__ == "__main__":
    main()
