#!/usr/bin/env python3
"""Transcribe a labelled set with several ASR approaches, one prediction CSV per approach.

Each ``--system`` is ``NAME=FAMILY:MODEL[?key=value&key=value...]``:

  whisper    faster-whisper / CTranslate2 (hub size like large-v3, a CT2 repo id, or a local
             fine-tuned export). Options: lang=auto|es|en (default auto), beam=5, vad=0|1,
             prev=1|0 (condition on previous text), multi=0|1 (re-detect language per 30 s
             window), batch=N (batched VAD pipeline), prompt_file=PATH (initial prompt).
  qwen3-asr  Qwen3-ASR through the qwen-asr package. Options: lang=auto|es|en, backend=vllm|
             transformers (default: vllm when a GPU and vLLM are present), batch=N,
             max_new_tokens=2048, gpu_mem=0.85, context=A.csv+B.csv (per-clip context built from
             other systems' predictions, e.g. the Spanish- and English-forced Whisper outputs),
             context_file=PATH (one context text for every clip, e.g. hotwords).

Every system runs in its own subprocess so GPU memory is released between systems, and a
system whose CSV already exists is skipped unless --overwrite. Output per system:
``OUT/NAME.csv`` (submission format, so ``compare.py OUT/*.csv`` scores every system), plus
``OUT/info/NAME.timing.json`` (load time, decode time, real-time factor),
``OUT/info/NAME.details.csv`` (Qwen's detected language) and ``OUT/info/meta.csv`` (clip
durations for ``compare.py --meta OUT/info/meta.csv --by duration_s:10,30,60``).

Example (Colab / GPU)::

    python run_systems.py --manifest data/submission_format.csv --clips data/clips --out-dir preds \\
        --system "wh_auto=whisper:large-v3" --system "wh_es=whisper:large-v3?lang=es" \\
        --system "wh_en=whisper:large-v3?lang=en" \\
        --system "qwen_auto=qwen3-asr:Qwen/Qwen3-ASR-1.7B" \\
        --system "qwen_merge=qwen3-asr:Qwen/Qwen3-ASR-1.7B?context=preds/wh_es.csv+preds/wh_en.csv"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SAMPLE_RATE = 16000
QWEN_LANGS = {"es": "Spanish", "en": "English"}


# --------------------------------------------------------------------------- helpers

def parse_system(spec: str) -> dict:
    """``NAME=FAMILY:MODEL?k=v&k=v`` -> {"name", "family", "model", "opts"}."""
    if "=" not in spec or ":" not in spec.split("=", 1)[1]:
        raise SystemExit(f"--system {spec!r}: expected NAME=FAMILY:MODEL[?key=value&...]")
    name, rest = spec.split("=", 1)
    family, rest = rest.split(":", 1)
    model, _, query = rest.partition("?")
    family = family.strip().lower()
    if family not in ("whisper", "qwen3-asr"):
        raise SystemExit(f"--system {spec!r}: unknown family {family!r} (whisper, qwen3-asr)")
    opts = {}
    for item in filter(None, query.split("&")):  # no URL decoding: '+' separates context CSVs
        key, _, value = item.partition("=")
        opts[key.strip()] = value.strip()
    return {"name": name.strip(), "family": family, "model": model.strip(), "opts": opts}


def load_audio(path: Path) -> np.ndarray:
    """Mono 16 kHz float32, decoded with PyAV (mp3/ogg/opus/m4a/wav) or librosa as a fallback."""
    try:
        from faster_whisper.audio import decode_audio

        return decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    except ImportError:
        import librosa

        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        try:
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except ImportError:
            return False


def flag(opts: dict, key: str, default: bool) -> bool:
    return str(opts.get(key, int(default))).lower() in ("1", "true", "yes")


def read_manifest(path: Path, limit: Optional[int]) -> list[str]:
    names = pd.read_csv(path)["audio_filename"].astype(str).tolist()
    return names[:limit] if limit else names


def build_context(opts: dict, names: list[str]) -> list[str]:
    """Per-clip context: the given prediction CSVs' transcripts (one per line) and/or a fixed text."""
    fixed = Path(opts["context_file"]).read_text().strip() if "context_file" in opts else ""
    per_clip = [[] for _ in names]
    for csv in filter(None, opts.get("context", "").split("+")):
        pred = pd.read_csv(csv).set_index("audio_filename")["transcript"].fillna("").astype(str)
        for i, n in enumerate(names):
            per_clip[i].append(pred.get(n, ""))
    return ["\n".join(filter(None, [fixed, *parts])) for parts in per_clip]


# --------------------------------------------------------------------------- families

def run_whisper(system: dict, audios: list[np.ndarray], log) -> list[str]:
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    opts = system["opts"]
    gpu = cuda_available()
    t0 = time.time()
    model = WhisperModel(system["model"], device="cuda" if gpu else "cpu",
                         compute_type=opts.get("compute_type", "float16" if gpu else "int8"))
    log(f"model loaded in {time.time() - t0:.0f}s", load_seconds=time.time() - t0)
    lang = opts.get("lang", "auto")
    kwargs = dict(
        language=None if lang == "auto" else lang,
        beam_size=int(opts.get("beam", 5)),
        condition_on_previous_text=flag(opts, "prev", True),
        multilingual=flag(opts, "multi", False),
        initial_prompt=Path(opts["prompt_file"]).read_text().strip() if "prompt_file" in opts else None,
    )
    batch = int(opts.get("batch", 0))
    pipe = BatchedInferencePipeline(model) if batch > 0 else None
    texts = []
    for i, audio in enumerate(audios):
        if pipe:  # the batched pipeline always segments with VAD
            segments, _ = pipe.transcribe(audio, batch_size=batch, vad_filter=True, **kwargs)
        else:
            segments, _ = model.transcribe(audio, vad_filter=flag(opts, "vad", False), **kwargs)
        texts.append(" ".join(s.text.strip() for s in segments).strip())
        if (i + 1) % 25 == 0 or i + 1 == len(audios):
            log(f"{i + 1}/{len(audios)} clips")
    return texts


def run_qwen(system: dict, audios: list[np.ndarray], names: list[str], log) -> tuple[list[str], list[str]]:
    from qwen_asr import Qwen3ASRModel

    opts = system["opts"]
    gpu = cuda_available()
    backend = opts.get("backend")
    if backend is None:  # find_spec checks for vLLM without paying for its import
        backend = "vllm" if gpu and importlib.util.find_spec("vllm") else "transformers"
    max_new = int(opts.get("max_new_tokens", 2048))  # qwen-asr's transformers default (512) truncates long clips
    t0 = time.time()
    if backend == "vllm":
        model = Qwen3ASRModel.LLM(model=system["model"], gpu_memory_utilization=float(opts.get("gpu_mem", 0.85)),
                                  max_model_len=int(opts.get("max_model_len", 8192)),
                                  max_inference_batch_size=int(opts.get("batch", -1)), max_new_tokens=max_new)
    else:
        import torch

        model = Qwen3ASRModel.from_pretrained(
            system["model"], dtype=torch.bfloat16 if gpu else torch.float32,
            device_map="cuda:0" if gpu else "cpu",
            max_inference_batch_size=int(opts.get("batch", 16)), max_new_tokens=max_new)
    log(f"{backend} model loaded in {time.time() - t0:.0f}s", load_seconds=time.time() - t0)
    lang = opts.get("lang", "auto")
    language = None if lang == "auto" else QWEN_LANGS.get(lang, lang)
    contexts = build_context(opts, names)
    texts, langs = [], []
    step = 64
    for start in range(0, len(audios), step):
        chunk = slice(start, start + step)
        results = model.transcribe(audio=[(a, SAMPLE_RATE) for a in audios[chunk]],
                                   context=contexts[chunk], language=language)
        texts.extend(r.text.strip() for r in results)
        langs.extend(r.language for r in results)
        log(f"{min(start + step, len(audios))}/{len(audios)} clips")
    return texts, langs


# --------------------------------------------------------------------------- driver

def worker(system: dict, args) -> None:
    """Transcribe every clip with one system and write NAME.csv + NAME.timing.json."""
    out_dir = Path(args.out_dir)
    info_dir = out_dir / "info"
    names = read_manifest(Path(args.manifest), args.limit)
    load = {"seconds": 0.0}

    def log(msg, load_seconds=None):
        if load_seconds is not None:
            load["seconds"] = load_seconds
        print(f"[{system['name']}] {msg}", flush=True)

    t0 = time.time()
    audios = [load_audio(Path(args.clips) / n) for n in names]
    audio_seconds = float(sum(len(a) for a in audios) / SAMPLE_RATE)
    log(f"decoded {len(audios)} clips ({audio_seconds / 60:.1f} min) in {time.time() - t0:.0f}s")

    t1 = time.time()
    details = None
    if system["family"] == "whisper":
        texts = run_whisper(system, audios, log)
    else:
        texts, langs = run_qwen(system, audios, names, log)
        details = pd.DataFrame({"audio_filename": names, "language": langs})
    decode_seconds = time.time() - t1 - load["seconds"]

    pd.DataFrame({"audio_filename": names, "transcript": texts}).to_csv(out_dir / f"{system['name']}.csv", index=False)
    if details is not None:
        details.to_csv(info_dir / f"{system['name']}.details.csv", index=False)
    timing = {
        "system": system, "clips": len(names), "audio_seconds": audio_seconds,
        "load_seconds": load["seconds"], "decode_seconds": decode_seconds,
        "rtfx": audio_seconds / max(decode_seconds, 1e-9), "gpu": cuda_available(),
    }
    (info_dir / f"{system['name']}.timing.json").write_text(json.dumps(timing, indent=2))
    log(f"decoded {audio_seconds / 60:.1f} min of audio in {decode_seconds:.0f}s "
        f"({timing['rtfx']:.1f}x real time; model load {load['seconds']:.0f}s)")


def write_meta(args) -> None:
    meta_path = Path(args.out_dir) / "info" / "meta.csv"
    names = read_manifest(Path(args.manifest), args.limit)
    if meta_path.exists() and not args.overwrite and set(names) <= set(pd.read_csv(meta_path)["audio_filename"]):
        return
    durations = [len(load_audio(Path(args.clips) / n)) / SAMPLE_RATE for n in names]
    pd.DataFrame({"audio_filename": names, "duration_s": durations}).to_csv(meta_path, index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="CSV with an audio_filename column (submission_format.csv or gold)")
    ap.add_argument("--clips", required=True, help="directory holding the audio files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--system", action="append", default=[], help="NAME=FAMILY:MODEL[?key=value&...]; repeatable")
    ap.add_argument("--limit", type=int, help="only the first N clips (quick checks)")
    ap.add_argument("--overwrite", action="store_true", help="re-run systems whose CSV exists")
    ap.add_argument("--in-process", action="store_true", help="run systems in this process (no GPU isolation)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    (Path(args.out_dir) / "info").mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(json.loads(args.worker), args)
        return
    systems = [parse_system(s) for s in args.system]
    if not systems:
        ap.error("give at least one --system")
    if len({s["name"] for s in systems}) != len(systems):
        ap.error("system names must be unique")
    write_meta(args)
    failed = []
    for system in systems:
        if (Path(args.out_dir) / f"{system['name']}.csv").exists() and not args.overwrite:
            print(f"[{system['name']}] exists, skipping (use --overwrite)")
            continue
        if args.in_process:
            worker(system, args)
            continue
        cmd = [sys.executable, __file__, "--manifest", args.manifest, "--clips", args.clips,
               "--out-dir", args.out_dir, "--worker", json.dumps(system)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if subprocess.run(cmd).returncode != 0:
            failed.append(system["name"])
    if failed:
        sys.exit(f"failed systems: {', '.join(failed)}")


if __name__ == "__main__":
    main()
