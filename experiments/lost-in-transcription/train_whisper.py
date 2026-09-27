#!/usr/bin/env python3
"""LoRA fine-tuning of Whisper on clips made by prepare_data.py.

Runs on one GPU, on several with torchrun, or on CPU for smoke tests. Precision follows the GPU:
bf16 on Ampere or newer, fp16 on older cards such as Kaggle's T4 and P100. Every ``--eval-steps``
it decodes a fixed sample of the eval set and scores it with the official normaliser, keeps the
checkpoint with the lowest WER, and writes it to ``OUTPUT/best_adapter``.

    torchrun --standalone --nproc_per_node=2 train_whisper.py \\
        --train data/miami_train.csv,data/dev_a.csv --eval data/miami_holdout.csv \\
        --scorer lost-in-transcription-runtime/score.py --output /tmp/runs/lv3

Defaults follow what worked for another team on this data (LoRA rank 64 on the upper half of the
encoder plus the decoder, lr 1e-4, effective batch 48); they saw gains flatten after 100-200 steps.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from rapidfuzz.distance import Levenshtein
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
                          WhisperForConditionalGeneration, WhisperProcessor)

SR = 16000
MAX_SAMPLES = 30 * SR
LORA_LEAVES = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}


# --------------------------------------------------------------------------- data

def codec_roundtrip(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    """Encode to Opus at a low bitrate and back, like a phone voice note."""
    import av

    buf = io.BytesIO()
    out = av.open(buf, "w", format="ogg")
    stream = out.add_stream("libopus", rate=SR, options={"compression_level": "3", "application": "voip"})
    stream.layout = "mono"
    stream.bit_rate = rng.choice([12000, 16000, 24000, 32000])
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    for i in range(0, len(pcm), 320):  # 20 ms frames
        chunk = pcm[i:i + 320]
        if len(chunk) < 320:
            chunk = np.pad(chunk, (0, 320 - len(chunk)))
        frame = av.AudioFrame.from_ndarray(chunk[None, :], format="s16", layout="mono")
        frame.sample_rate, frame.pts = SR, i
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()
    buf.seek(0)
    decoded = []
    with av.open(buf) as inp:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for frame in inp.decode(audio=0):
            decoded.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        decoded.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    y = np.concatenate(decoded).astype(np.float32) / 32768 if decoded else np.zeros_like(audio)
    return y[: len(audio)] if len(y) >= len(audio) else np.pad(y, (0, len(audio) - len(y)))


class ClipDataset(torch.utils.data.Dataset):
    """Rows of audio,text,language[,source] -> Whisper log-mel features and label ids."""

    def __init__(self, df: pd.DataFrame, processor: WhisperProcessor, augment_prob: float = 0.0):
        self.rows = df.to_dict("records")
        self.fe = processor.feature_extractor
        self.tok = processor.tokenizer
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio, sr = sf.read(r["audio"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        audio = audio[:MAX_SAMPLES]
        if self.augment_prob and r.get("source") == "miami" and random.random() < self.augment_prob:
            rng = random.Random(random.random())
            audio = codec_roundtrip(audio, rng) * (10 ** (rng.uniform(-6, 6) / 20))
        feats = self.fe(audio, sampling_rate=SR).input_features[0]
        lang = r.get("language") if isinstance(r.get("language"), str) and r.get("language") else "en"
        self.tok.set_prefix_tokens(language=lang, task="transcribe", predict_timestamps=False)
        labels = self.tok(str(r["text"])).input_ids[:440]
        return {"input_features": feats, "labels": labels}


class Collator:
    def __init__(self, decoder_start_token_id: int, dtype: torch.dtype):
        self.start = decoder_start_token_id
        self.dtype = dtype

    def __call__(self, features):
        feats = torch.tensor(np.stack([f["input_features"] for f in features]), dtype=self.dtype)
        width = max(len(f["labels"]) for f in features)
        labels = torch.full((len(features), width), -100, dtype=torch.long)
        for i, f in enumerate(features):
            labels[i, : len(f["labels"])] = torch.tensor(f["labels"])
        if (labels[:, 0] == self.start).all():  # the model prepends <|startoftranscript|> itself
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


def read_rows(spec: str) -> pd.DataFrame:
    frames = [pd.read_csv(p, keep_default_na=False) for p in spec.split(",") if p]
    df = pd.concat(frames, ignore_index=True)
    return df[df["text"].astype(str).str.strip() != ""].reset_index(drop=True)


# --------------------------------------------------------------------------- model

def lora_targets(model, encoder_from_layer: int, decoder: bool) -> list[str]:
    """Attention and MLP projections of encoder layers >= encoder_from_layer (and the decoder)."""
    names = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if not isinstance(module, torch.nn.Linear) or "layers" not in parts or parts[-1] not in LORA_LEAVES:
            continue
        side, idx = parts[parts.index("layers") - 1], int(parts[parts.index("layers") + 1])
        if (side == "encoder" and idx >= encoder_from_layer) or (side == "decoder" and decoder):
            names.append(name)
    return names


def load_normalizer(path: str):
    spec = importlib.util.spec_from_file_location("lit_official_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_text


def make_metrics(tokenizer, normalize):
    def words(text):
        return [w for w in re.sub(r"\s\s+", " ", normalize(text)).strip().split(" ") if w]

    def compute(pred):
        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        pred_ids = np.where(pred_ids < 0, tokenizer.pad_token_id, pred_ids)
        label_ids = np.where(pred.label_ids < 0, tokenizer.pad_token_id, pred.label_ids)
        hyps = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        errors = total = 0
        for h, r in zip(hyps, refs):
            rw = words(r)
            errors += Levenshtein.distance(rw, words(h))
            total += len(rw)
        return {"wer": errors / max(total, 1)}

    return compute


class PrintEval(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero and metrics:
            print(f"[eval] step {state.global_step}: WER {metrics.get('eval_wer', float('nan')):.4f}", flush=True)


# --------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", required=True, help="comma-separated CSVs from prepare_data.py")
    ap.add_argument("--eval", required=True, help="CSV used to pick the best checkpoint")
    ap.add_argument("--scorer", required=True, help="official score.py (for its normaliser)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--eval-samples", type=int, default=160, help="eval clips decoded at each evaluation")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8, help="per-device micro-batch")
    ap.add_argument("--effective-batch", type=int, default=48)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--encoder-from-layer", type=int, default=16)
    ap.add_argument("--no-decoder-lora", action="store_true")
    ap.add_argument("--augment-prob", type=float, default=0.5, help="Opus round trip on Miami clips")
    ap.add_argument("--workers", type=int, help="dataloader workers per process (default: CPUs / processes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-eval-on-start", action="store_true", help="skip the step-0 (zero-shot) evaluation")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --output")
    ap.add_argument("--cpu", action="store_true", help="smoke tests without a GPU")
    args = ap.parse_args(argv)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    gpu = torch.cuda.is_available() and not args.cpu
    # is_bf16_supported() also counts emulation (True on a T4); real bf16 needs Ampere or newer
    bf16 = gpu and torch.cuda.get_device_capability(0)[0] >= 8
    fp16 = gpu and not bf16
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    accum = max(1, round(args.effective_batch / (args.batch_size * world)))
    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 2) // world)
    random.seed(args.seed)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = None  # detect the language, as at inference
    model.generation_config.task = "transcribe"
    targets = lora_targets(model, args.encoder_from_layer, not args.no_decoder_lora)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout, target_modules=targets, bias="none"))

    train_df = read_rows(args.train)
    eval_df = read_rows(args.eval).sample(frac=1.0, random_state=args.seed).head(args.eval_samples)
    rank0 = int(os.environ.get("RANK", "0")) == 0
    if rank0:
        model.print_trainable_parameters()
        print(f"train clips {len(train_df)} ({train_df['duration'].sum() / 3600:.1f} h) | eval clips {len(eval_df)} | "
              f"{'bf16' if bf16 else 'fp16' if fp16 else 'fp32'} | {world} process(es) x batch {args.batch_size} "
              f"x accum {accum} = {args.batch_size * world * accum}", flush=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 8),
        gradient_accumulation_steps=accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        bf16=bf16,
        fp16=fp16,
        use_cpu=not gpu,
        gradient_checkpointing=gpu,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_on_start=not args.no_eval_on_start,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=workers if gpu else 0,
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ClipDataset(train_df, processor, args.augment_prob),
        eval_dataset=ClipDataset(eval_df, processor),
        data_collator=Collator(model.config.decoder_start_token_id, dtype),
        compute_metrics=make_metrics(processor.tokenizer, load_normalizer(args.scorer)),
        callbacks=[PrintEval()],
    )
    resume = args.resume and any(Path(args.output).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)

    if trainer.is_world_process_zero():
        best_dir = Path(args.output) / "best_adapter"
        trainer.model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        history = [h for h in trainer.state.log_history if "eval_wer" in h]
        summary = {
            "base_model": args.model,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_wer": trainer.state.best_metric,
            "eval_history": [{"step": h["step"], "wer": h["eval_wer"]} for h in history],
            "train_loss": [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h],
            "args": vars(args),
        }
        (Path(args.output) / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"best eval WER {summary['best_eval_wer']} at {summary['best_checkpoint']}; adapter -> {best_dir}")


if __name__ == "__main__":
    main()
