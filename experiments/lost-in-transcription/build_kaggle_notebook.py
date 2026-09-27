#!/usr/bin/env python3
"""Generate kaggle_whisper_lora.ipynb from the scripts in this folder.

The notebook embeds prepare_data.py, train_whisper.py, export_whisper.py, run_systems.py,
compare.py and submission/main.py as %%writefile cells, so it runs on Kaggle without cloning this
repo and always ships the tested code. Re-run this after editing any of those scripts.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

INTRO = """\
# Lost in Transcription (Spanish–English): fine-tune Whisper large-v3 with LoRA on Kaggle

What this notebook does:

1. Installs the package versions the competition runtime uses (transformers 4.57.6, peft 0.20.0,
   faster-whisper 1.2.1, ctranslate2 4.8.2).
2. Finds the competition data: the Bangor Miami corpus and the official dev set.
3. Turns Miami's CHAT transcripts into training clips of up to 28 s in the dev transcript style,
   and splits the dev set into two halves.
4. Fine-tunes Whisper large-v3 with LoRA on Miami plus dev half A. The checkpoint with the lowest
   WER on held-out Miami conversations is kept. Miami clips get a random Opus round trip so they
   sound more like phone voice notes.
5. Exports the model to CTranslate2. Scores it and zero-shot large-v3 on dev half B with the
   official scorer, using the submission's own decoding settings.
6. Writes `submission.zip` (`main.py` + `model/`) to the output, ready to upload.

**Setup**

- *Settings → Accelerator*: **GPU T4 ×2** (P100 also works, more slowly). *Internet*: **on**.
- **Data**: download the Miami and dev archives from the competition's data page and add them to
  this notebook as a **private** Kaggle dataset (*Add Input*). Tarballs or extracted folders both
  work. Alternatively, fill in `MDC_DATASET_IDS` below and store your Mozilla Data Collective API
  key as a Kaggle secret named `MDC_API_KEY` (*Add-ons → Secrets*).
- **Time**: about 1.5–2 h on T4 ×2 with the defaults.

**For the final model**, set `DEV_FOR_TRAINING = "all"` and run again. The submitted model then
has seen every test-like dev clip, but there is no dev score left to check it against.
"""

CONFIG = """\
import os
from pathlib import Path

SMOKE = os.environ.get("LIT_SMOKE") == "1"  # tiny CPU run used to test this notebook

# ---- where things live (the environment variables are only for testing outside Kaggle)
INPUT_ROOT = Path(os.environ.get("LIT_INPUT_ROOT", "/kaggle/input"))  # attached datasets
WORK = Path(os.environ.get("LIT_WORK", "/kaggle/working"))           # saved output (20 GB limit)
SCRATCH = Path(os.environ.get("LIT_SCRATCH", "/tmp/lit"))            # big intermediate files
MDC_DATASET_IDS = {"miami": "", "enspa_dev": ""}  # only to download with the MDC API instead

# ---- experiment
BASE_MODEL = "openai/whisper-large-v3"
ZERO_SHOT_MODEL = "large-v3"       # faster-whisper name of the same model, for the baseline
DEV_FOR_TRAINING = "half"          # "none": score on all of dev | "half": train on A, score on B
                                   # "all": final model, no dev score
RUN_ZERO_SHOT_BASELINE = True      # score untuned large-v3 on the same dev clips (~5 min)
MAX_STEPS = 300                    # another team saw gains flatten after 100-200 steps
EVAL_STEPS = 50
EVAL_SAMPLES = 160                 # held-out Miami clips decoded at each evaluation
LR = 1e-4
LORA_R, LORA_ALPHA = 64, 64
ENCODER_FROM_LAYER = 16            # LoRA on encoder layers 16-31 plus the whole decoder
BATCH_PER_GPU = 8                  # fits a 16 GB T4 in fp16 with gradient checkpointing
EFFECTIVE_BATCH = 48
AUGMENT_PROB = 0.5                 # share of Miami clips passed through an Opus round trip
CT2_QUANTIZATION = "float16"       # "int8_float16" halves the zip if size becomes a problem

if SMOKE:
    BASE_MODEL, ZERO_SHOT_MODEL = "openai/whisper-tiny", "tiny"
    MAX_STEPS, EVAL_STEPS, EVAL_SAMPLES = 4, 2, 4
    BATCH_PER_GPU, EFFECTIVE_BATCH, ENCODER_FROM_LAYER = 2, 4, 2

for d in (WORK, SCRATCH):
    d.mkdir(parents=True, exist_ok=True)
"""

INSTALL = """\
import importlib.util
import subprocess
import sys

if not SMOKE:
    !pip install -q "transformers==4.57.6" "peft==0.20.0" "ctranslate2==4.8.2" "faster-whisper==1.2.1" jiwer typer wordfreq rapidfuzz soundfile

# CTranslate2 opens CUDA 12 cuBLAS/cuDNN by name at the first GPU computation. Put the pip-installed
# copies on the library path so the evaluation subprocesses find them.
lib_dirs = []
for pkg in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"):
    try:
        spec = importlib.util.find_spec(pkg)
    except ModuleNotFoundError:
        spec = None
    for p in (spec.submodule_search_locations or []) if spec else []:
        if (Path(p) / "lib").is_dir():
            lib_dirs.append(str(Path(p) / "lib"))
os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).strip(":")

# Query GPUs with nvidia-smi so this process never opens a CUDA context (it would sit on GPU 0
# during training).
try:
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
except FileNotFoundError:
    gpus = []
N_GPUS = len(gpus)
import transformers, peft, faster_whisper, ctranslate2
print(f"python {sys.version.split()[0]} | transformers {transformers.__version__} | peft {peft.__version__} | "
      f"faster-whisper {faster_whisper.__version__} | ctranslate2 {ctranslate2.__version__}")
print(f"{N_GPUS} GPU(s): {gpus or 'none - CPU only'}")
"""

DATA = """\
import json
import shutil
import tarfile
import zipfile

RAW = SCRATCH / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def mdc_download(dataset_id, name):
    import requests
    from kaggle_secrets import UserSecretsClient

    key = UserSecretsClient().get_secret("MDC_API_KEY")
    r = requests.post(f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download",
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=60)
    r.raise_for_status()
    target = RAW / f"{name}.tar.gz"
    with requests.get(r.json()["downloadUrl"], stream=True, timeout=600) as dl, open(target, "wb") as f:
        dl.raise_for_status()
        for chunk in dl.iter_content(1 << 20):
            f.write(chunk)
    print(f"downloaded {name}: {target.stat().st_size / 1e6:.0f} MB")


def extract_archives(root):
    for arc in sorted(p for p in root.rglob("*") if p.is_file() and p.name.endswith((".tar.gz", ".tgz", ".tar", ".zip"))):
        dest = RAW / arc.name.split(".")[0]
        if dest.exists():
            continue
        print("extracting", arc.name)
        if arc.suffix == ".zip":
            with zipfile.ZipFile(arc) as z:
                z.extractall(dest)
        else:
            with tarfile.open(arc) as t:
                try:
                    t.extractall(dest, filter="data")
                except TypeError:  # Python without extraction filters
                    t.extractall(dest)


for name, dataset_id in MDC_DATASET_IDS.items():
    if dataset_id and not (RAW / f"{name}.tar.gz").exists():
        mdc_download(dataset_id, name)
if INPUT_ROOT.exists():
    extract_archives(INPUT_ROOT)
extract_archives(RAW)

roots = [r for r in (INPUT_ROOT, RAW) if r.exists()]
candidates = [d for r in roots for d in [r, *r.rglob("*")] if d.is_dir()]
MIAMI_DIR = next((d for d in candidates if (d / "audios").is_dir() and any((d / "chat").glob("*.cha"))), None)
DEV_DIR = next((d for d in candidates if (d / "clips").is_dir()
                and ((d / "metadata.tsv").is_file() or (d / "metadata.csv").is_file())), None)
assert MIAMI_DIR, "No Miami data found: expected a folder with chat/*.cha and audios/ (see the setup notes)"
assert DEV_DIR, "No dev set found: expected a folder with metadata.tsv (or .csv) and clips/ (see the setup notes)"
print("Miami:", MIAMI_DIR, f"({len(list((MIAMI_DIR / 'chat').glob('*.cha')))} transcripts)")
print("Dev:  ", DEV_DIR, f"({len(list((DEV_DIR / 'clips').iterdir()))} clips)")
"""

SCORER = """\
import urllib.request

SCORER = SCRATCH / "score.py"  # the organisers' scorer, so every WER here matches the leaderboard's
if not SCORER.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/drivendataorg/lost-in-transcription-runtime/main/score.py", SCORER)
print("scorer:", SCORER)
"""

PREP = """\
PREP = SCRATCH / "prepared"
if not (PREP / "stats.json").exists():
    !python prepare_data.py --miami "{MIAMI_DIR}" --dev "{DEV_DIR}" --out "{PREP}" --workers {os.cpu_count()}
stats = json.loads((PREP / "stats.json").read_text())
print(json.dumps({k: v for k, v in stats.items() if k != "convention_audit"}, indent=2))
print("\\n".join(stats["convention_audit"]))

train_sets = [PREP / "miami_train.csv"]
DEV_EVAL_GOLD = {"none": PREP / "gold_dev_all.csv", "half": PREP / "gold_dev_b.csv", "all": None}[DEV_FOR_TRAINING]
if DEV_FOR_TRAINING == "half":
    train_sets.append(PREP / "dev_a.csv")
elif DEV_FOR_TRAINING == "all":
    train_sets.append(PREP / "dev_all.csv")
print("train on:", [p.name for p in train_sets], "| dev score on:", DEV_EVAL_GOLD.name if DEV_EVAL_GOLD else "nothing")
"""

BASELINE = """\
EVAL_DIR = WORK / "dev_eval"
if RUN_ZERO_SHOT_BASELINE and DEV_EVAL_GOLD is not None:
    # Same decoding as the submission: VAD chunks, batch 16, beam 5, language detected per clip.
    !python run_systems.py --manifest "{DEV_EVAL_GOLD}" --clips "{DEV_DIR / 'clips'}" --out-dir "{EVAL_DIR}" --system "zeroshot=whisper:{ZERO_SHOT_MODEL}?batch=16"
"""

TRAIN = """\
RUN = SCRATCH / "run"
cpu_flag = "" if N_GPUS else "--cpu"
train_arg = ",".join(str(p) for p in train_sets)
!torchrun --standalone --nproc_per_node={max(1, N_GPUS)} train_whisper.py --train "{train_arg}" --eval "{PREP / 'miami_holdout.csv'}" --scorer "{SCORER}" --output "{RUN}" --model {BASE_MODEL} --max-steps {MAX_STEPS} --eval-steps {EVAL_STEPS} --eval-samples {EVAL_SAMPLES} --lr {LR} --batch-size {BATCH_PER_GPU} --effective-batch {EFFECTIVE_BATCH} --lora-r {LORA_R} --lora-alpha {LORA_ALPHA} --encoder-from-layer {ENCODER_FROM_LAYER} --augment-prob {AUGMENT_PROB} --resume {cpu_flag}

summary = json.loads((RUN / "summary.json").read_text())
print("held-out Miami WER by step (step 0 = zero-shot):")
for row in summary["eval_history"]:
    print(f"  step {row['step']:>4}: {row['wer']:.4f}")
print("best:", summary["best_checkpoint"], summary["best_eval_wer"])
shutil.copytree(RUN / "best_adapter", WORK / "best_adapter", dirs_exist_ok=True)
_ = shutil.copy(RUN / "summary.json", WORK / "train_summary.json")
"""

EXPORT = """\
CT2 = SCRATCH / "ct2_model"
!python export_whisper.py --adapter "{RUN / 'best_adapter'}" --out-ct2 "{CT2}" --quantization {CT2_QUANTIZATION}
"""

EVALUATE = """\
if DEV_EVAL_GOLD is not None:
    !python run_systems.py --manifest "{DEV_EVAL_GOLD}" --clips "{DEV_DIR / 'clips'}" --out-dir "{EVAL_DIR}" --overwrite --system "finetuned=whisper:{CT2}?batch=16"
    !python compare.py --ref "{DEV_EVAL_GOLD}" {EVAL_DIR}/*.csv --scorer "{SCORER}" --meta "{EVAL_DIR / 'info' / 'meta.csv'}" --by duration_s:10,30,60 --out-dir "{WORK / 'dev_eval_report'}"
else:
    print("DEV_FOR_TRAINING = 'all': every dev clip was trained on, so there is no dev score.")
"""

PACKAGE = """\
import pandas as pd

SUB = SCRATCH / "submission_src"
shutil.rmtree(SUB, ignore_errors=True)
SUB.mkdir(parents=True)
shutil.copy("main.py", SUB / "main.py")
shutil.copytree(CT2, SUB / "model")
ZIP = WORK / "submission.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:  # weights barely compress; storing is faster
    for f in sorted(SUB.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(SUB))
print(f"{ZIP} ({ZIP.stat().st_size / 1e9:.2f} GB)")

# Run the zip the way the platform does: unpack, run main.py against a copy of the data layout.
CHECK = SCRATCH / "zip_check"
shutil.rmtree(CHECK, ignore_errors=True)
(CHECK / "data").mkdir(parents=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(CHECK / "src")
gold = pd.read_csv(DEV_EVAL_GOLD if DEV_EVAL_GOLD is not None else PREP / "gold_dev_all.csv")
gold[["audio_filename"]].assign(transcript="").to_csv(CHECK / "data" / "submission_format.csv", index=False)
(CHECK / "data" / "clips").symlink_to(DEV_DIR / "clips")
!cd "{CHECK / 'src'}" && LIT_DATA_DIR="{CHECK / 'data'}" LIT_SUBMISSION_PATH="{CHECK / 'submission.csv'}" python main.py
if DEV_EVAL_GOLD is not None:
    !python "{SCORER}" "{DEV_EVAL_GOLD}" --predicted-path "{CHECK / 'submission.csv'}"
"""

OUTRO = """\
## Next steps

- **Download** `submission.zip` from this notebook's *Output* and upload it on the competition's
  submission page. `best_adapter/`, `train_summary.json` and `dev_eval_report/` are saved as well.
- **Read the dev report** above. Keep a change only if the paired Δ interval against the
  alternative excludes 0.
- **Final model**: set `DEV_FOR_TRAINING = "all"` and run again.
- **Other checkpoints**: to score one besides the best, point `export_whisper.py --adapter` at
  `/tmp/lit/run/checkpoint-<step>` in the same session, then re-run the evaluation cell.
"""

SCRIPTS = [
    ("prepare_data.py", "prepare_data.py"),
    ("train_whisper.py", "train_whisper.py"),
    ("export_whisper.py", "export_whisper.py"),
    ("run_systems.py", "run_systems.py"),
    ("compare.py", "compare.py"),
    ("submission/main.py", "main.py"),
]


def hidden(source: str):
    cell = new_code_cell(source)
    cell.metadata["jupyter"] = {"source_hidden": True}
    return cell


def build() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(INTRO), new_code_cell(CONFIG), new_code_cell(INSTALL),
             new_markdown_cell("## Code\nThe scripts below are written to files so `torchrun` can launch them. "
                               "Their sources are collapsed; expand a cell to read or edit it.")]
    for src, dst in SCRIPTS:
        cells.append(hidden(f"%%writefile {dst}\n" + (HERE / src).read_text()))
    cells += [
        new_markdown_cell("## 1. Data"), new_code_cell(DATA), new_code_cell(SCORER), new_code_cell(PREP),
        new_markdown_cell("## 2. Zero-shot baseline on the dev clips held out from training"), new_code_cell(BASELINE),
        new_markdown_cell("## 3. Fine-tune (LoRA)"), new_code_cell(TRAIN),
        new_markdown_cell("## 4. Export to CTranslate2 and score on dev"), new_code_cell(EXPORT), new_code_cell(EVALUATE),
        new_markdown_cell("## 5. Package and check the submission"), new_code_cell(PACKAGE),
        new_markdown_cell(OUTRO),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


if __name__ == "__main__":
    out = HERE / "kaggle_whisper_lora.ipynb"
    nbformat.write(build(), out)
    print(f"wrote {out}")
