"""Lost in Transcription submission: a fine-tuned Whisper (CTranslate2) run with faster-whisper.

Zip this file together with the exported ``model/`` directory (main.py at the zip root). It reads
``submission_format.csv`` and ``clips/`` from the data directory and writes ``submission.csv``.
Logs carry only counts and timings: the rules forbid clip names or other test information in logs.
"""

import os
import time
from pathlib import Path

import pandas as pd
from faster_whisper import BatchedInferencePipeline, WhisperModel

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("LIT_DATA_DIR", "/code_execution/data"))
SUBMISSION_PATH = Path(os.environ.get("LIT_SUBMISSION_PATH", "/code_execution/submission/submission.csv"))
BEAM_SIZE = int(os.environ.get("LIT_BEAM_SIZE", "5"))
BATCH_SIZE = int(os.environ.get("LIT_BATCH_SIZE", "16"))


def load_model() -> WhisperModel:
    device = os.environ.get("LIT_DEVICE")
    if device is None:
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return WhisperModel(str(HERE / "model"), device=device, compute_type="float16" if device == "cuda" else "int8")


def main() -> None:
    t0 = time.time()
    submission = pd.read_csv(DATA_DIR / "submission_format.csv")
    pipeline = BatchedInferencePipeline(load_model())
    print(f"model loaded in {time.time() - t0:.0f}s; {len(submission)} clips", flush=True)
    texts, failures = [], 0
    for i, name in enumerate(submission["audio_filename"]):
        try:
            # VAD splits long clips into <=30 s chunks that are decoded as a batch; the language is
            # detected per clip, as in training.
            segments, _ = pipeline.transcribe(str(DATA_DIR / "clips" / name), batch_size=BATCH_SIZE,
                                              beam_size=BEAM_SIZE, vad_filter=True, language=None)
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as exc:  # keep going; one bad clip must not cost the whole run
            failures += 1
            print(f"clip {i}: {type(exc).__name__}", flush=True)
            text = ""
        # Empty transcripts are rejected by the platform; "." normalises to nothing when scored.
        texts.append(text or ".")
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(submission)} clips in {time.time() - t0:.0f}s", flush=True)
    submission["transcript"] = texts
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission[["audio_filename", "transcript"]].to_csv(SUBMISSION_PATH, index=False)
    print(f"wrote {len(submission)} rows ({failures} failures) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
