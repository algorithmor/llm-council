"""Tests for run_systems.py helpers and make_long_set.py (no models needed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import compare  # noqa: E402
import make_long_set  # noqa: E402
import run_systems  # noqa: E402


def test_parse_system_keeps_plus_and_paths():
    s = run_systems.parse_system(
        "qwen_merge=qwen3-asr:Qwen/Qwen3-ASR-1.7B?context=runs/dev/wh_es.csv+runs/dev/wh_en.csv&lang=es")
    assert s == {"name": "qwen_merge", "family": "qwen3-asr", "model": "Qwen/Qwen3-ASR-1.7B",
                 "opts": {"context": "runs/dev/wh_es.csv+runs/dev/wh_en.csv", "lang": "es"}}
    s = run_systems.parse_system("ft=whisper:/content/runs/lv3/ct2")
    assert (s["family"], s["model"], s["opts"]) == ("whisper", "/content/runs/lv3/ct2", {})
    with pytest.raises(SystemExit):
        run_systems.parse_system("x=voxtral:mistralai/Voxtral-Mini-3B-2507")
    with pytest.raises(SystemExit):
        run_systems.parse_system("no_family_given")


def test_build_context_joins_threads(tmp_path):
    names = ["a.wav", "b.wav"]
    pd.DataFrame({"audio_filename": names, "transcript": ["hola amigo", "see you"]}).to_csv(tmp_path / "es.csv", index=False)
    pd.DataFrame({"audio_filename": ["a.wav"], "transcript": ["hello friend"]}).to_csv(tmp_path / "en.csv", index=False)
    (tmp_path / "hot.txt").write_text("Miami, Hialeah\n")
    opts = {"context": f"{tmp_path / 'es.csv'}+{tmp_path / 'en.csv'}", "context_file": str(tmp_path / "hot.txt")}
    assert run_systems.build_context(opts, names) == ["Miami, Hialeah\nhola amigo\nhello friend",
                                                      "Miami, Hialeah\nsee you"]
    assert run_systems.build_context({}, names) == ["", ""]


def test_join_transcripts_keeps_each_clip_sentence_initial():
    assert make_long_set.join_transcripts(["hola", "Okay so...", "  ", "Yes!"]) == "hola. Okay so... Yes!"


def test_long_set_scores_like_its_parts(tmp_path):
    try:
        normalize = compare.load_normalizer(compare.find_scorer(None))
    except SystemExit:
        pytest.skip("official score.py not found (set LIT_SCORE_PY)")
    texts = ["Hola, ¿cómo estás?", "I told her — no sé", "Okay so...", "Mi mamá (?) says que no",
             "Uh, Miami is far.", "Pues sí"]
    clips = tmp_path / "clips"
    clips.mkdir()
    names = []
    for i, _ in enumerate(texts):
        names.append(f"c{i}.wav")
        t = np.arange(int(16000 * (2 + i))) / 16000
        sf.write(clips / names[-1], (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 16000)
    pd.DataFrame({"audio_filename": names, "transcript": texts}).to_csv(tmp_path / "gold.csv", index=False)
    make_long_set.main(["--gold", str(tmp_path / "gold.csv"), "--clips", str(clips), "--out-dir",
                        str(tmp_path / "long"), "--min-sec", "5", "--max-sec", "9", "--gap-sec", "0.5"])
    gold = pd.read_csv(tmp_path / "long" / "gold.csv")
    meta = pd.read_csv(tmp_path / "long" / "meta.csv")
    assert meta["n_parts"].sum() == len(texts)
    for name, dur in zip(meta["audio_filename"], meta["duration_s"]):
        assert sf.info(tmp_path / "long" / "clips" / name).duration == pytest.approx(dur, abs=1e-3)
    # every word of every clip appears once, and the joined text normalises to the same words
    per_clip = sorted(w for t in texts for w in compare.tokenize(normalize(t)))
    joined = sorted(w for t in gold["transcript"] for w in compare.tokenize(normalize(t)))
    assert joined == per_clip
