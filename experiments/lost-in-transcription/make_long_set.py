#!/usr/bin/env python3
"""Build a long-form evaluation set by concatenating labelled clips.

Test clips run to about 4 minutes while dev clips are mostly short, so a system can look fine on
dev and still break on long audio (early stopping, repetition loops, 30 s window seams). This
packs shuffled clips into 60-240 s recordings separated by short silences and joins their
transcripts so each original clip still starts a sentence (the scorer lowercases sentence-initial
letters, so this keeps scoring consistent with the per-clip set).

    python make_long_set.py --gold data/gold.csv --clips data/clips --out-dir long
    # -> long/clips/*.wav, long/gold.csv, long/submission_format.csv, long/meta.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from run_systems import SAMPLE_RATE, load_audio

TERMINAL = (".", "!", "?", "…")


def join_transcripts(parts: list[str]) -> str:
    out = []
    for text in parts:
        text = text.strip()
        if not text:
            continue
        if not text.endswith(TERMINAL):
            text += "."
        out.append(text)
    return " ".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold", required=True, help="CSV audio_filename,transcript")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-sec", type=float, default=60)
    ap.add_argument("--max-sec", type=float, default=240)
    ap.add_argument("--gap-sec", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    gold = pd.read_csv(args.gold).fillna("")
    rows = list(gold[["audio_filename", "transcript"]].itertuples(index=False))
    random.Random(args.seed).shuffle(rows)
    rng = random.Random(args.seed + 1)
    out = Path(args.out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(args.gap_sec * SAMPLE_RATE), dtype=np.float32)

    records, group, texts, seconds = [], [], [], 0.0
    target = rng.uniform(args.min_sec, args.max_sec)

    def flush():
        nonlocal group, texts, seconds, target
        name = f"long_{len(records):03d}.wav"
        audio = np.concatenate([x for a in group for x in (a, gap)][:-1])
        sf.write(out / "clips" / name, audio, SAMPLE_RATE)
        records.append({"audio_filename": name, "transcript": join_transcripts(texts),
                        "duration_s": len(audio) / SAMPLE_RATE, "n_parts": len(group)})
        group, texts, seconds = [], [], 0.0
        target = rng.uniform(args.min_sec, args.max_sec)

    for filename, transcript in rows:
        audio = load_audio(Path(args.clips) / filename)
        group.append(audio)
        texts.append(transcript)
        seconds += len(audio) / SAMPLE_RATE + args.gap_sec
        if seconds >= target:
            flush()
    if group:
        flush()

    table = pd.DataFrame(records)
    table[["audio_filename", "transcript"]].to_csv(out / "gold.csv", index=False)
    table[["audio_filename"]].assign(transcript="").to_csv(out / "submission_format.csv", index=False)
    table[["audio_filename", "duration_s", "n_parts"]].to_csv(out / "meta.csv", index=False)
    print(f"wrote {len(table)} long clips ({table['duration_s'].sum() / 60:.1f} min, "
          f"{table['duration_s'].min():.0f}-{table['duration_s'].max():.0f} s each) to {out}")


if __name__ == "__main__":
    main()
