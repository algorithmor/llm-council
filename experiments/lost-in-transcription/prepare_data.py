#!/usr/bin/env python3
"""Turn the competition downloads into training / evaluation manifests with 16 kHz FLAC clips.

Inputs (as unpacked from Mozilla Data Collective):

* Bangor Miami: ``MIAMI/chat/*.cha`` CHAT transcripts and ``MIAMI/audios/<name>.mp3`` recordings.
* Official dev set: ``DEV/clips/*`` plus ``DEV/metadata.tsv`` (``audio_filename``, ``transcript``,
  optionally ``speaker``).

Miami utterances are cleaned into verbatim text in the dev transcript style (CHAT markup removed,
fillers and repetitions kept, cut-off words as ``wor...``, the dev spelling conventions applied),
merged into single-speaker clips of up to ~28 s, and split by conversation into train / holdout.
Clips with unintelligible speech or much crosstalk are dropped. Each clip gets a language label
(the majority of its words' @s tags), used as Whisper's language token in training.

Outputs in OUT/: ``clips/*.flac``, ``miami_train.csv``, ``miami_holdout.csv``, ``dev_all.csv``,
``dev_a.csv`` / ``dev_b.csv`` (a fixed split so part of dev can be trained on while the rest is
held out), ``gold_dev_all.csv`` / ``gold_dev_b.csv`` (for compare.py) and ``stats.json``.

    python prepare_data.py --miami MIAMI --dev DEV --out OUT
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
BULLET = re.compile(r"[\x15•](\d+)_(\d+)[\x15•]")
PRECODE = re.compile(r"\[-\s*([a-z]{3})\]")
FILLERS = {"uh", "um", "eh", "ah", "er", "erm", "mm", "mhm", "hm", "hmm", "uhm", "em", "ehm", "uh-huh", "huh"}
TERMINATORS = {
    ".": ".", "?": "?", "!": "!", "+...": "...", "+..?": "?", "+!?": "?", "+/.": "—", "+//.": "—",
    "+/?": "?", "+//?": "?", "+\"/.": ".", "+\".": ".", "+.": ".", "+=.": ".", "+...?": "?",
}
LINKERS = {"+<", "+^", "+,", "++", "+\"", "+"}
STRIP_CHARS = re.compile(r"[ˈˌ↑↓≈≋∬⌈⌉⌊⌋‹›“”\"^~:]")
# Spelling differences between Miami's CHAT transcripts and the dev/test references, as reported by
# another team's audit (dev writes "going to", never "gonna"; "uh", never "ah"). Check them with
# the audit printed by this script before relying on them.
DEFAULT_CONVENTIONS = {
    "expand": {"gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of",
               "lemme": "let me", "gimme": "give me"},
    "rename": {"ah": "uh", "ok": "okay", "o_k": "okay"},
}
AUDIT_PAIRS = [("gonna", "going"), ("wanna", "want"), ("ah", "uh"), ("ok", "okay"), ("um", "uh"),
               ("yeah", "yes"), ("cause", "because"), ("pa'", "para")]


# --------------------------------------------------------------------------- CHAT parsing

@dataclass
class Utterance:
    speaker: str
    start: float  # seconds
    end: float
    text: str
    n_es: int = 0
    n_en: int = 0
    unintelligible: bool = False
    raw: str = field(default="", repr=False)


def _word_language(tag: Optional[str], default: str) -> Optional[str]:
    """Language of one word from its @s tag (None = ambiguous / both languages)."""
    if tag is None:
        return default
    if tag == "":  # bare @s: the other language of the pair
        return "en" if default == "es" else "es"
    if "&" in tag or "+" in tag:
        return None
    return {"spa": "es", "eng": "en"}.get(tag[:3], None)


def style(tokens: list[str]) -> str:
    """Join tokens in the dev style: terminators attached, sentence-initial capitals."""
    out = ""
    for tok in tokens:
        if tok in (".", "?", "!", "...", "—"):
            out = out.rstrip() + tok + " "
        else:
            out += tok + " "
    out = re.sub(r"\s+", " ", out).strip()
    # capitalise after a sentence end, but not after an ellipsis ("wor... and" stays lowercase)
    out = re.sub(r"(^|(?<!\.\.)[.?!]\s+)([a-záéíóúñü])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out if re.search(r"[^\W\d_]", out) else ""


def clean_chat_line(body: str, default_lang: str, conventions: Optional[dict] = None) -> tuple[str, int, int, bool]:
    """CHAT main-tier text -> (verbatim text, n Spanish words, n English words, unintelligible)."""
    conv = conventions if conventions is not None else DEFAULT_CONVENTIONS
    s = BULLET.sub(" ", body)
    m = PRECODE.search(s)
    lang = {"spa": "es", "eng": "en"}.get(m.group(1), default_lang) if m else default_lang
    s = PRECODE.sub(" ", s)
    unintelligible = bool(re.search(r"(?<![\w&])(xxx|yyy|www)(?!\w)", s))
    s = re.sub(r"\[[^\]]*\]", " ", s)  # retracings, overlaps, explanations, replacements, error codes
    s = s.replace("<", " ").replace(">", " ")
    s = re.sub(r"\(\.+\)", " ", s)  # pauses
    tokens, n_es, n_en = [], 0, 0
    for tok in s.split():
        if tok in TERMINATORS:
            tokens.append(TERMINATORS[tok])
            continue
        if tok in LINKERS or tok.startswith("+"):
            continue
        if tok.startswith("&"):
            body_ = tok.lstrip("&")
            if body_[:1] in ("=", "~", "*"):
                continue  # events, nonwords, interposed words
            word = body_.lstrip("-+").split("@")[0]
            word = STRIP_CHARS.sub("", re.sub(r"[()]", "", word)).lower()
            if not word:
                continue
            tokens.append(conv["rename"].get(word, word) if word in FILLERS else word + "...")
            continue
        if tok in ("xxx", "yyy", "www") or tok.startswith("0"):
            continue
        tag = None
        if "@" in tok:
            tok, _, marker = tok.partition("@")
            if marker.startswith("s"):
                tag = marker[2:] if marker.startswith("s:") else ""
        word = STRIP_CHARS.sub("", tok.replace("(", "").replace(")", ""))
        word = word.replace("_", " ").replace("+", " ").strip()
        if not word or not re.search(r"[^\W\d_]", word):
            continue
        low = word.lower()
        if low in conv["expand"]:
            word = conv["expand"][low]
        elif low in conv["rename"]:
            word = conv["rename"][low]
        elif low == "i":
            word = "I"
        if word.lower() not in FILLERS:  # fillers carry no language
            wl = _word_language(tag, lang)
            n_es += wl == "es"
            n_en += wl == "en"
        tokens.append(word)
    return style(tokens), n_es, n_en, unintelligible


def parse_chat(path: Path, conventions: Optional[dict] = None) -> tuple[list[Utterance], str]:
    """Timed utterances of one .cha file (continuation lines joined, dependent tiers skipped)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    default = "en"
    for line in lines:
        if line.startswith("@Languages:"):
            first = re.split(r"[,\s]+", line.split(":", 1)[1].strip())[0]
            default = {"spa": "es", "eng": "en"}.get(first[:3], "en")
            break
    tiers: list[Optional[str]] = []
    for line in lines:
        if line.startswith("*"):
            tiers.append(line)
        elif line.startswith("\t") and tiers and tiers[-1] is not None:
            tiers[-1] += " " + line.strip()
        elif line.startswith(("%", "@")):
            tiers.append(None)
    utts = []
    for tier in tiers:
        if not tier or ":" not in tier:
            continue
        head, body = tier.split(":", 1)
        marks = BULLET.findall(body)
        if not marks:
            continue
        start, end = int(marks[-1][0]) / 1000, int(marks[-1][1]) / 1000
        if end <= start:
            continue
        text, n_es, n_en, unint = clean_chat_line(body, default, conventions)
        utts.append(Utterance(head[1:].strip(), start, end, text, n_es, n_en, unint, body))
    utts.sort(key=lambda u: u.start)
    return utts, default


# --------------------------------------------------------------------------- clip building

def build_clips(utts: list[Utterance], rng: random.Random, max_sec: float = 28.0, min_sec: float = 1.0,
                max_gap: float = 1.5, max_crosstalk: float = 0.2) -> list[dict]:
    """Merge consecutive same-speaker utterances into clips of random length up to max_sec."""
    by_speaker: dict[str, list[Utterance]] = defaultdict(list)
    for u in utts:
        by_speaker[u.speaker].append(u)
    clips = []
    for speaker, own in by_speaker.items():
        others = [(u.start, u.end) for u in utts if u.speaker != speaker]
        i = 0
        while i < len(own):
            target = rng.uniform(min(6.0, max_sec), max_sec)
            group = [own[i]]
            j = i + 1
            while (j < len(own) and own[j].start - group[-1].end <= max_gap
                   and own[j].end - group[0].start <= target):
                group.append(own[j])
                j += 1
            i = j
            start, end = group[0].start, group[-1].end
            dur = end - start
            if dur < min_sec or dur > max_sec + 2 or any(u.unintelligible for u in group):
                continue
            text = " ".join(u.text for u in group if u.text).strip()
            if not text:
                continue
            overlap = sum(max(0.0, min(end, e) - max(start, s)) for s, e in others)
            if overlap / dur > max_crosstalk:
                continue
            n_es, n_en = sum(u.n_es for u in group), sum(u.n_en for u in group)
            clips.append({"speaker": speaker, "start": start, "end": end, "text": text,
                          "n_es": n_es, "n_en": n_en})
    return clips


def load_audio(path: Path) -> np.ndarray:
    try:
        from faster_whisper.audio import decode_audio

        return decode_audio(str(path), sampling_rate=SR)
    except ImportError:
        import librosa

        return librosa.load(str(path), sr=SR, mono=True)[0].astype(np.float32)


def process_conversation(job: tuple) -> list[dict]:
    cha, audio_path, out_dir, seed, max_sec, pad = job
    utts, default = parse_chat(Path(cha))
    rng = random.Random(f"{seed}-{Path(cha).stem}")
    clips = build_clips(utts, rng, max_sec=max_sec)
    if not clips:
        return []
    audio = load_audio(Path(audio_path))
    rows = []
    for k, c in enumerate(clips):
        lo = max(0, int((c["start"] - pad) * SR))
        hi = min(len(audio), int((c["end"] + pad) * SR))
        if hi - lo < SR * 0.5:
            continue
        name = f"{Path(cha).stem}_{k:04d}.flac"
        sf.write(Path(out_dir) / "clips" / name, audio[lo:hi], SR)
        lang = "es" if c["n_es"] > c["n_en"] else "en" if c["n_en"] > c["n_es"] else default
        rows.append({"audio": str(Path(out_dir) / "clips" / name), "text": c["text"], "language": lang,
                     "duration": round((hi - lo) / SR, 3), "source": "miami", "speaker": c["speaker"],
                     "conversation": Path(cha).stem, "n_es": c["n_es"], "n_en": c["n_en"]})
    return rows


# --------------------------------------------------------------------------- dev set

def dev_target(transcript: str) -> str:
    """Training target from a dev reference: drop [annotations] and (?) markers, keep the rest."""
    s = re.sub(r"\[[^\]]*\]", " ", str(transcript))
    s = re.sub(r"\(\?+\)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def prepare_dev(dev_dir: Path, out_dir: Path, seed: int) -> pd.DataFrame:
    meta_path = next((p for p in (dev_dir / "metadata.tsv", dev_dir / "metadata.csv") if p.exists()), None)
    if meta_path is None:
        raise SystemExit(f"{dev_dir}: no metadata.tsv / metadata.csv with audio_filename,transcript")
    meta = pd.read_csv(meta_path, sep="\t" if meta_path.suffix == ".tsv" else ",", keep_default_na=False)
    rows = []
    for r in meta.itertuples(index=False):
        src = dev_dir / "clips" / r.audio_filename
        audio = load_audio(src)
        dst = out_dir / "clips" / f"dev_{Path(r.audio_filename).stem}.flac"
        sf.write(dst, audio, SR)
        rows.append({"audio": str(dst), "text": dev_target(r.transcript), "language": "",
                     "duration": round(len(audio) / SR, 3), "source": "dev",
                     "speaker": str(getattr(r, "speaker", "") or ""), "conversation": "dev",
                     "audio_filename": r.audio_filename, "original_audio": str(src),
                     "transcript": r.transcript})
    dev = pd.DataFrame(rows)
    # Split in half: by speaker when speakers are given, so halves share no voices.
    rng = random.Random(seed)
    if dev["speaker"].str.len().gt(0).all() and dev["speaker"].nunique() >= 4:
        speakers = sorted(dev["speaker"].unique())
        rng.shuffle(speakers)
        half, acc, target = set(), 0.0, dev["duration"].sum() / 2
        for s in speakers:
            if acc < target:
                half.add(s)
                acc += dev.loc[dev["speaker"] == s, "duration"].sum()
        dev["dev_split"] = np.where(dev["speaker"].isin(half), "a", "b")
    else:
        order = list(range(len(dev)))
        rng.shuffle(order)
        dev["dev_split"] = "b"
        dev.loc[order[: len(dev) // 2], "dev_split"] = "a"
    return dev


def language_of_text(text: str, table: dict[str, float]) -> str:
    """Majority language of a dev transcript, from P(Spanish) of words seen in Miami."""
    ps = [table[w] for w in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text.lower()) if w in table]
    return "es" if ps and np.mean(ps) > 0.5 else "en"


# --------------------------------------------------------------------------- main

def audit(miami_text: pd.Series, dev_text: pd.Series) -> list[str]:
    """Per-1k-word frequencies of spelling variants in Miami targets vs dev references."""
    def freqs(texts):
        c = Counter(w for t in texts for w in re.findall(r"[^\W\d_]+'?[^\W\d_]*", str(t).lower()))
        return c, max(sum(c.values()), 1)
    (cm, nm), (cd, nd) = freqs(miami_text), freqs(dev_text)
    lines = []
    for a, b in AUDIT_PAIRS:
        lines.append(f"{a:>6}/{b:<8} Miami {1000 * cm[a] / nm:6.2f}/{1000 * cm[b] / nm:6.2f}   "
                     f"dev {1000 * cd[a] / nd:6.2f}/{1000 * cd[b] / nd:6.2f}  (per 1k words)")
    return lines


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--miami", required=True, help="dir with chat/*.cha and audios/*.mp3")
    ap.add_argument("--dev", required=True, help="dir with clips/ and metadata.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.08, help="share of Miami hours held out by conversation")
    ap.add_argument("--max-sec", type=float, default=28.0)
    ap.add_argument("--pad", type=float, default=0.15, help="seconds of audio kept around each clip")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-conversations", type=int, help="only the first N conversations (quick tests)")
    args = ap.parse_args(argv)

    miami, dev_dir, out = Path(args.miami), Path(args.dev), Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    chas = sorted((miami / "chat").glob("*.cha"))
    audio_for = {p.stem: p for p in (miami / "audios").glob("*") if p.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg")}
    jobs = [(str(c), str(audio_for[c.stem]), str(out), args.seed, args.max_sec, args.pad)
            for c in chas if c.stem in audio_for]
    if args.limit_conversations:
        jobs = jobs[: args.limit_conversations]
    if not jobs:
        raise SystemExit(f"{miami}: found no chat/*.cha with a matching audios/<name>.mp3")
    print(f"Miami: {len(jobs)} conversations with audio ({len(chas)} transcripts)")
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, part in enumerate(pool.map(process_conversation, jobs)):
            rows.extend(part)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} conversations, {len(rows)} clips")
    miami_df = pd.DataFrame(rows)
    if miami_df.empty:
        raise SystemExit("no Miami clips survived filtering")

    # Conversation-level holdout (~holdout_frac of hours).
    rng = random.Random(args.seed)
    hours = miami_df.groupby("conversation")["duration"].sum()
    convs = sorted(hours.index)
    rng.shuffle(convs)
    held, acc = set(), 0.0
    for c in convs:
        if acc >= args.holdout_frac * hours.sum() or len(held) >= len(convs) - 1:
            break
        held.add(c)
        acc += hours[c]
    holdout = miami_df[miami_df["conversation"].isin(held)]
    train = miami_df[~miami_df["conversation"].isin(held)]

    dev = prepare_dev(dev_dir, out, args.seed)
    word_lang = Counter()
    for r in train.itertuples():
        for w in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", r.text.lower()):
            word_lang[(w, r.language)] += 1
    table = {}
    for (w, lang), n in word_lang.items():
        es, en = word_lang[(w, "es")], word_lang[(w, "en")]
        if es + en >= 3:
            table[w] = es / (es + en)
    dev["language"] = [language_of_text(t, table) for t in dev["text"]]

    cols = ["audio", "text", "language", "duration", "source", "speaker", "conversation"]
    train[cols].to_csv(out / "miami_train.csv", index=False)
    holdout[cols].to_csv(out / "miami_holdout.csv", index=False)
    dev_cols = cols + ["audio_filename", "original_audio", "transcript", "dev_split"]
    dev[dev_cols].to_csv(out / "dev_all.csv", index=False)
    for part in ("a", "b"):
        dev[dev["dev_split"] == part][dev_cols].to_csv(out / f"dev_{part}.csv", index=False)
    dev[["audio_filename", "transcript"]].to_csv(out / "gold_dev_all.csv", index=False)
    dev[dev["dev_split"] == "b"][["audio_filename", "transcript"]].to_csv(out / "gold_dev_b.csv", index=False)

    stats = {
        "miami_train_hours": round(train["duration"].sum() / 3600, 2),
        "miami_train_clips": int(len(train)),
        "miami_holdout_hours": round(holdout["duration"].sum() / 3600, 2),
        "miami_holdout_clips": int(len(holdout)),
        "miami_language_mix": {k: int(v) for k, v in train["language"].value_counts().items()},
        "dev_clips": int(len(dev)),
        "dev_minutes": round(dev["duration"].sum() / 60, 1),
        "dev_split_clips": {k: int(v) for k, v in dev["dev_split"].value_counts().items()},
        "dev_language_mix": {k: int(v) for k, v in dev["language"].value_counts().items()},
        "convention_audit": audit(train["text"], dev["transcript"]),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "convention_audit"}, indent=2))
    print("Spelling audit (check the conventions before training):")
    for line in stats["convention_audit"]:
        print("  " + line)
    return stats


if __name__ == "__main__":
    main()
