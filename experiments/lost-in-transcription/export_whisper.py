#!/usr/bin/env python3
"""Merge a Whisper LoRA adapter and convert the result to CTranslate2 for faster-whisper.

The output directory is what the competition runtime loads (faster-whisper 1.2.1 on
ctranslate2 4.8.2), and it includes tokenizer.json and preprocessor_config.json so nothing is
downloaded at inference time.

    python export_whisper.py --adapter /tmp/runs/lv3/best_adapter --out-ct2 /kaggle/working/model
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", required=True, help="LoRA adapter directory (train_whisper.py best_adapter)")
    ap.add_argument("--base", help="base model (default: the one recorded in the adapter)")
    ap.add_argument("--out-ct2", required=True)
    ap.add_argument("--out-hf", help="also keep the merged Hugging Face model here")
    ap.add_argument("--quantization", default="float16", help="float16 | int8_float16 | int8 | float32")
    args = ap.parse_args(argv)

    base = args.base or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    hf_dir = Path(args.out_hf) if args.out_hf else Path(tempfile.mkdtemp(prefix="merged_"))
    model = WhisperForConditionalGeneration.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.generation_config.forced_decoder_ids = None
    model.save_pretrained(hf_dir, safe_serialization=True)
    WhisperProcessor.from_pretrained(base).save_pretrained(hf_dir)
    if not (hf_dir / "tokenizer.json").exists():  # faster-whisper needs the fast tokenizer file offline
        WhisperTokenizerFast.from_pretrained(base).save_pretrained(hf_dir)

    from ctranslate2.converters import TransformersConverter

    out = Path(args.out_ct2)
    TransformersConverter(str(hf_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(out), quantization=args.quantization, force=True)
    (out / "export_info.json").write_text(json.dumps({"base": base, "adapter": str(args.adapter),
                                                     "quantization": args.quantization}, indent=2))
    if not args.out_hf:
        shutil.rmtree(hf_dir, ignore_errors=True)

    from faster_whisper import WhisperModel  # fail here, not on the platform, if the export is broken

    WhisperModel(str(out), device="cpu", compute_type="int8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    print(f"CTranslate2 model ({args.quantization}, {size:.2f} GB) -> {out}")


if __name__ == "__main__":
    main()
