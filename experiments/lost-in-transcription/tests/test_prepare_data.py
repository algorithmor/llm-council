"""Tests for prepare_data.py: CHAT cleaning, clip building and the end-to-end manifests."""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prepare_data as pdp  # noqa: E402


@pytest.mark.parametrize("body, lang, expected, n_es, n_en, unint", [
    # word-level @s tags against an English default; terminator attached, first letter capitalised
    ("I'm gonna go a@s:spa la@s:spa tienda@s:spa . \x1515210_18560\x15", "en",
     "I'm going to go a la tienda.", 3, 3, False),
    # utterance precode switches the default language; fillers kept, events dropped
    ("[- spa] &-uh no sé &=laughs pero es verdad . \x1521100_23500\x15", "en",
     "Uh no sé pero es verdad.", 5, 0, False),
    # retracing keeps the spoken repetition; self-interruption becomes an em dash
    ("<yeah yeah> [/] yeah I know +/. \x1520000_21000\x15", "en", "Yeah yeah yeah I know—", 0, 5, False),
    # fragments become cut-off words, nonwords and pauses disappear, ah -> uh, ok -> okay
    ("&+wh what (.) &~fu ah ok so@s:eng bueno . \x151_2\x15", "es", "Wh... what uh okay so bueno.", 3, 1, False),
    # unintelligible speech is flagged (the clip gets dropped)
    ("xxx store . \x151_2\x15", "en", "Store.", 0, 1, True),
    # replacements keep the spoken word; overlap / error codes and omitted words vanish
    ("she goed [: went] [*] 0is there [>] ? \x151_2\x15", "en", "She goed there?", 0, 3, False),
    # ambiguous both-language tags count for neither language
    ("no@s:eng&spa me@s:spa digas . \x151_2\x15", "es", "No me digas.", 2, 0, False),
    # (be)cause keeps the full form; compounds and lengthening marks are cleaned
    ("(be)cause New_York is so:@s:eng far ! \x151_2\x15", "es", "Because New York is so far!", 4, 1, False),
])
def test_clean_chat_line(body, lang, expected, n_es, n_en, unint):
    text, es, en, un = pdp.clean_chat_line(body, lang)
    assert text == expected
    assert (es, en, un) == (n_es, n_en, unint)


CHA = """@UTF8
@Begin
@Languages:\tspa, eng
@Participants:\tKAR Karen Adult, MAR Maria Adult
@ID:\tspa|Bangor|KAR||female|||Adult|||
*KAR:\tbueno pues vamos a@s:spa ver . \x150_2000\x15
%aut:\tsomething
*MAR:\tI told her@s:eng that@s:eng
\tit's fine@s:eng . \x152100_4000\x15
*KAR:\tsí sí . \x154100_5000\x15
*KAR:\txxx no sé . \x155100_6000\x15
*MAR:\t&=laughs . \x156100_6500\x15
@End
"""


def test_parse_chat_joins_continuations_and_skips_tiers(tmp_path):
    p = tmp_path / "conv.cha"
    p.write_text(CHA, encoding="utf-8")
    utts, default = pdp.parse_chat(p)
    assert default == "es"
    # untagged words take the file's default language (Spanish here); the laughter-only turn is kept
    # with empty text so it still counts as crosstalk
    assert [u.speaker for u in utts] == ["KAR", "MAR", "KAR", "KAR", "MAR"]
    assert utts[1].text == "I told her that it's fine." and (utts[1].n_en, utts[1].n_es) == (3, 3)
    assert utts[1].start == 2.1 and utts[3].unintelligible and utts[4].text == ""


def test_build_clips_merges_speaker_turns_and_drops_crosstalk():
    U = pdp.Utterance
    utts = [
        U("A", 0.0, 2.0, "Uno.", 1, 0), U("A", 2.5, 4.0, "Dos.", 1, 0),   # merged (gap 0.5 s)
        U("B", 10.0, 12.0, "Hello.", 0, 1),
        U("A", 20.0, 24.0, "Tres cuatro.", 2, 0), U("B", 20.5, 23.5, "Over you.", 0, 2),  # 75% crosstalk
        U("A", 30.0, 32.0, "Xxx.", 0, 0, True),  # unintelligible
    ]
    clips = pdp.build_clips(utts, random.Random(0), max_sec=28.0)
    texts = sorted(c["text"] for c in clips)
    assert texts == ["Hello.", "Uno. Dos."]


def test_end_to_end_manifests(tmp_path):
    miami, dev, out = tmp_path / "miami", tmp_path / "dev", tmp_path / "out"
    (miami / "chat").mkdir(parents=True)
    (miami / "audios").mkdir()
    (dev / "clips").mkdir(parents=True)
    tone = (0.05 * np.sin(2 * np.pi * 200 * np.arange(SR := 16000) / SR)).astype(np.float32)
    for k in range(3):
        (miami / "chat" / f"conv{k}.cha").write_text(CHA, encoding="utf-8")
        sf.write(miami / "audios" / f"conv{k}.wav", np.tile(tone, 8), SR)
    rows = []
    for k in range(6):
        sf.write(dev / "clips" / f"d{k}.wav", tone, SR)
        rows.append({"audio_filename": f"d{k}.wav", "transcript": f"Okay so mañana [laughs] vamos (?) al store {k}.",
                     "speaker": f"s{k % 3}"})
    pd.DataFrame(rows).to_csv(dev / "metadata.tsv", sep="\t", index=False)
    stats = pdp.main(["--miami", str(miami), "--dev", str(dev), "--out", str(out), "--workers", "1",
                      "--holdout-frac", "0.3"])
    train, hold = pd.read_csv(out / "miami_train.csv"), pd.read_csv(out / "miami_holdout.csv")
    assert set(train["conversation"]).isdisjoint(set(hold["conversation"]))
    assert len(train) and len(hold)
    assert all(Path(a).exists() for a in train["audio"])
    dev_all = pd.read_csv(out / "dev_all.csv", keep_default_na=False)
    assert dev_all["text"].iloc[0] == "Okay so mañana vamos al store 0."
    a, b = pd.read_csv(out / "dev_a.csv"), pd.read_csv(out / "dev_b.csv")
    assert len(a) == 3 and len(b) == 3  # 3 speakers is too few for a speaker split, so clips are split
    gold_b = pd.read_csv(out / "gold_dev_b.csv")
    assert list(gold_b.columns) == ["audio_filename", "transcript"] and len(gold_b) == len(b)
    assert stats["miami_train_clips"] == len(train)
