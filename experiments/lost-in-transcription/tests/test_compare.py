"""Tests for compare.py. Needs the official scorer: set LIT_SCORE_PY or keep a
lost-in-transcription-runtime checkout next to this repo."""

import importlib.util
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rapidfuzz.distance import Levenshtein

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import compare  # noqa: E402

try:
    SCORER = compare.find_scorer(None)
except SystemExit:
    pytest.skip("official score.py not found (set LIT_SCORE_PY)", allow_module_level=True)

spec = importlib.util.spec_from_file_location("official_score", SCORER)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)

REFS = [
    "Hola, ¿cómo estás? I'm good, pero estoy un poco cansada.",
    "Okay so mañana vamos al store — no, a la tienda de mi tía.",
    "Uh, I don't know... ¿Tú qué piensas?",
    "Me dijo que the meeting was at three, pero nadie llegó.",
    "Sí, sí. Yeah, that's what I told her. [laughs]",
    "Mi mamá (?) says que no puedo ir. ¡Qué pena!",
    "It's like, you know, bien raro.",
    "Ay, Dios mío. Mhm. Okay, bye.",
    "Él es de Miami, but he grew up en Puerto Rico.",
    "Um, the thing is que I can't — I can't go.",
]
TRANSLATED = [  # what an English-forced "thread" produces: Spanish translated away
    "Hello, how are you? I'm good, but I'm a little tired.",
    "Okay so tomorrow we go to the store — no, to my aunt's store.",
    "Uh, I don't know... What do you think?",
    "He told me that the meeting was at three, but nobody came.",
    "Yes, yes. Yeah, that's what I told her.",
    "My mom says that I can't go. What a shame!",
    "It's like, you know, very weird.",
    "Oh my God. Mhm. Okay, bye.",
    "He is from Miami, but he grew up in Puerto Rico.",
    "Um, the thing is that I can't — I can't go.",
]


def clip_names(n):
    return [f"clip_{i:03d}.mp3" for i in range(n)]


def write_csv(path, names, texts):
    pd.DataFrame({"audio_filename": names, "transcript": texts}).to_csv(path, index=False)
    return path


def perturb(text, rng):
    """Random word-level edits plus casing/punctuation noise."""
    words = text.split()
    out = []
    for w in words:
        r = rng.random()
        if r < 0.08:
            continue  # deletion
        if r < 0.16:
            out.append(rng.choice(["the", "que", "okay", "no", "sí", "Miami", "uh"]))  # substitution
        elif r < 0.22:
            out.extend([w, rng.choice(["like", "pues", "um"])])  # insertion
        elif r < 0.28:
            out.append(w.upper() if rng.random() < 0.5 else w.lower())  # casing
        elif r < 0.32:
            out.append(w.strip(",.?!¿¡"))  # punctuation
        else:
            out.append(w)
    return " ".join(out)


def official_wer(ref_csv, pred_csv):
    """Replicates score.py main(): align rows by audio_filename, then _word_error_rate."""
    predicted = pd.read_csv(pred_csv).set_index("audio_filename").sort_index()
    actual = pd.read_csv(ref_csv).set_index("audio_filename").sort_index()
    predicted = predicted.loc[actual.index]
    return official._word_error_rate(
        predicted[["transcript"]].fillna("").to_numpy(),
        actual[["transcript"]].fillna("").to_numpy(),
        official.normalize_text,
    )


@pytest.fixture
def dataset(tmp_path):
    rng = random.Random(7)
    names = clip_names(len(REFS))
    refs = write_csv(tmp_path / "ref.csv", names, REFS)
    preds = {
        "exact": write_csv(tmp_path / "exact.csv", names, REFS),
        "noisy1": write_csv(tmp_path / "noisy1.csv", names, [perturb(t, rng) for t in REFS]),
        "noisy2": write_csv(tmp_path / "noisy2.csv", names, [perturb(t, rng) for t in REFS]),
        "noisy3": write_csv(tmp_path / "noisy3.csv", names, [perturb(t, rng) for t in REFS]),
        "translated": write_csv(tmp_path / "translated.csv", names, TRANSLATED),
    }
    return tmp_path, refs, preds


def run(args):
    return compare.main([str(a) for a in args])


def by_name(summary):
    return {row["system"]: row for row in summary["systems"]}


def test_wer_matches_official_scorer(dataset):
    tmp, ref, preds = dataset
    summary = run(["--ref", ref, "--scorer", SCORER, "--bootstrap", 0, *preds.values()])
    rows = by_name(summary)
    for name, path in preds.items():
        assert rows[name]["wer"] == pytest.approx(official_wer(ref, path), abs=1e-12), name
    assert rows["exact"]["wer"] == 0.0


def test_wer_matches_official_cli(dataset):
    tmp, ref, preds = dataset
    out = subprocess.run(
        [sys.executable, str(SCORER), str(ref), "--predicted-path", str(preds["noisy1"])],
        capture_output=True, text=True, check=True,
    ).stdout
    cli = float(re.search(r"WER: ([0-9.]+)", out).group(1))
    summary = run(["--ref", ref, "--scorer", SCORER, "--bootstrap", 0, preds["noisy1"]])
    assert by_name(summary)["noisy1"]["wer"] == pytest.approx(cli, abs=5e-7)


def test_fuzz_alignment_counts_match_jiwer():
    import jiwer

    rng = random.Random(0)
    vocab = ["a", "b", "c", "d", "e", "sí", "no", "the"]
    refs, hyps = [], []
    for _ in range(300):
        refs.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(1, 25))))
        hyps.append(" ".join(rng.choice(vocab) for _ in range(rng.randint(0, 25))))
    out = jiwer.process_words(refs, hyps)
    s = d = i = 0
    for r, h in zip(refs, hyps):
        rt, ht = compare.tokenize(r), compare.tokenize(h)
        s_, d_, i_, pairs = compare.align(rt, ht)
        assert s_ + d_ + i_ == Levenshtein.distance(rt, ht)
        assert sum(p[0] is not None for p in pairs) == len(rt)
        assert sum(p[1] is not None for p in pairs) == len(ht)
        s, d, i = s + s_, d + d_, i + i_
    assert (s, d, i) == (out.substitutions, out.deletions, out.insertions)


def test_tokenize_matches_jiwer_default_transform():
    import jiwer

    for text in ["  hola   que  tal ", "a\tb", "a\t\tb", "one\ntwo", "", "   "]:
        expected = jiwer.transformations.wer_default(text)
        expected = expected[0] if expected else []
        assert compare.tokenize(text) == expected, repr(text)


def test_bootstrap_paired_delta_and_oracle(dataset):
    tmp, ref, preds = dataset
    summary = run(["--ref", ref, "--scorer", SCORER, "--bootstrap", 2000, "--baseline", "noisy1",
                   preds["noisy1"], preds["noisy2"], preds["translated"], f"copy={preds['noisy1']}"])
    rows = by_name(summary)
    lo, hi = rows["noisy2"]["ci"]
    assert lo <= rows["noisy2"]["wer"] <= hi
    # an identical copy of the baseline has a zero paired delta and never "beats" it
    assert rows["copy"]["delta_ci"] == [0.0, 0.0]
    assert rows["copy"]["p_better"] == 0.0
    # translating the Spanish away is clearly worse than light noise
    assert rows["translated"]["delta_ci"][0] > 0
    assert summary["oracle_wer"] <= min(r["wer"] for r in summary["systems"])


def test_cluster_bootstrap_and_breakdown(dataset, tmp_path):
    tmp, ref, preds = dataset
    names = clip_names(len(REFS))
    meta = tmp_path / "meta.csv"
    pd.DataFrame({
        "audio_filename": names,
        "speaker": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5", "s5"],
        "duration_s": [3, 8, 12, 25, 40, 70, 5, 9, 31, 90],
    }).to_csv(meta, index=False)
    summary = run(["--ref", ref, "--scorer", SCORER, "--bootstrap", 500, "--meta", meta,
                   "--cluster-col", "speaker", "--by", "speaker", "--by", "duration_s:10,30,60",
                   preds["noisy1"], preds["exact"]])
    assert set(summary["breakdowns"]["duration_s"]["exact"]) == {"<10", "10-30", "30-60", ">=60"}
    assert all(v == 0 for v in summary["breakdowns"]["speaker"]["exact"].values())


def test_casing_filler_and_language_diagnostics(dataset, tmp_path):
    tmp, ref, preds = dataset
    names = clip_names(len(REFS))
    lower = write_csv(tmp_path / "lower.csv", names, [t.lower() for t in REFS])
    no_fillers = write_csv(tmp_path / "nofill.csv", names, [
        re.sub(r"\b(Uh|Um|Mhm)\b,?\s*", "", t) for t in REFS])
    summary = run(["--ref", ref, "--scorer", SCORER, "--bootstrap", 0,
                   lower, no_fillers, preds["translated"]])
    rows = by_name(summary)
    assert rows["lower"]["wer"] > 0 and rows["lower"]["wer_ignoring_case"] == 0
    assert rows["nofill"]["filler_error"] == 1.0
    pytest.importorskip("wordfreq")
    assert rows["translated"]["es_error"] > 0.8
    assert rows["translated"]["en_error"] < 0.15
    assert rows["translated"]["switch_error"] > 0.5
    assert rows["lower"]["es_error"] < 0.25


def test_rover_votes_and_ties_keep_pivot():
    hyps = [
        "vamos al store mañana".split(),
        "vamos a la store mañana".split(),
        "vamos al store mañana okay".split(),
    ]
    assert compare.rover(hyps, [1, 1, 1], pivot=1) == "vamos al store mañana".split()
    # a lone insertion is outvoted; a lone deletion is outvoted
    assert compare.rover([["a", "b"], ["a", "x", "b"], ["a", "b"]], [1, 1, 1], pivot=1) == ["a", "b"]
    assert compare.rover([["a", "b"], ["a"], ["a", "b"]], [1, 1, 1], pivot=1) == ["a", "b"]
    # with two systems every disagreement is a tie, so the pivot wins
    assert compare.rover([["a", "b"], ["a", "c"]], [1, 1], pivot=1) == ["a", "c"]
    # weights break ties
    assert compare.rover([["a", "b"], ["a", "c"]], [2, 1], pivot=1) == ["a", "b"]


def test_combine_scores_like_a_system(dataset):
    tmp, ref, preds = dataset
    out_dir = tmp / "out"
    summary = run(["--ref", ref, "--scorer", SCORER, "--bootstrap", 0, "--combine", "all",
                   "--out-dir", out_dir, preds["noisy1"], preds["noisy2"], preds["noisy3"]])
    rows = by_name(summary)
    rover_name = "ROVER(noisy1,noisy2,noisy3)"
    assert rover_name in rows and "MBR(noisy1,noisy2,noisy3)" in rows
    # three independent noisy copies of the truth: voting should beat the average member
    members = [rows[n]["wer"] for n in ("noisy1", "noisy2", "noisy3")]
    assert rows[rover_name]["wer"] < np.mean(members)
    # the written combined CSV re-scores to the same WER with the official scorer
    written = out_dir / "ROVER_noisy1_noisy2_noisy3.csv"
    assert rows[rover_name]["wer"] == pytest.approx(official_wer(ref, written), abs=1e-12)
    per_clip = pd.read_csv(out_dir / "per_clip.csv")
    assert (per_clip["errors:oracle"] <= per_clip["errors:noisy1"]).all()


def test_missing_rows(dataset, tmp_path):
    tmp, ref, preds = dataset
    partial = write_csv(tmp_path / "partial.csv", clip_names(3), REFS[:3])
    with pytest.raises(SystemExit):
        run(["--ref", ref, "--scorer", SCORER, "--bootstrap", 0, partial])
    summary = run(["--ref", ref, "--scorer", SCORER, "--bootstrap", 0, "--allow-missing", partial])
    assert 0 < by_name(summary)["partial"]["wer"] < 1
