# ChildNet GUI

A browser GUI for [ChildNet](https://github.com/MartinPernus/ChildNet) (Pernuš et al., *IEEE Access* 2023): upload a photo of each parent, then explore the predicted child with sliders for the dominant parent, age, gender and more. It runs locally or on Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/algorithmor/llm-council/blob/master/childnet_gui/ChildNet_GUI.ipynb)

## Controls

| Control | What it does | From |
|---|---|---|
| **Dominant parent** (−1 to +1) | Pulls the child towards the mother (−1) or the father (+1). 0 is ChildNet's own blend. | ChildNet (`--move2parent`) |
| **Age group** and **gender** | Sets the child's age group: 0–2, 3–6, 7–9, 10–14, 15–19, 20–29, 30–39, 40–49, 50–69 or 70+ years. Untick the box for ChildNet's unedited output. | ChildNet (`--child_age`, `--child_gender`) |
| **Age/gender edit strength** | Scales that edit: 1 is ChildNet's, 0 is none, above 1 exaggerates it. | Added |
| **Dominant parent per facial level** | Separate sliders for coarse (face shape, pose), medium (facial features) and fine (colouring, skin, hair texture) layers, e.g. the father's face shape with the mother's colouring. | Added, on ChildNet's coarse/medium/fine split |
| **Variability** and **seed** | Samples different children: 0 always gives the same child, 0.5 is ChildNet's sampling mode. | ChildNet (`--sample`), rate adjustable |
| **Siblings** tab | Several samples at once, using seeds *seed*, *seed*+1, … | Added |
| **Sweep** tab | A strip across the dominant parent or all age groups, like the paper's figures. | Added |
| **Model's view** tab | The face crops ChildNet receives and how its encoder reconstructs each parent. If a parent's reconstruction doesn't resemble them, the child won't either. | Added |
| **Model weights** | Trained on Next of Kin (NoKDB) or Families in the Wild (FIW). | ChildNet (`--model_weights`) |
| **Auto-align faces** | Finds the face and crops it FFHQ-style, like ChildNet's example images, so everyday photos work. | Added |

With the same settings, the GUI produces exactly the images ChildNet's own `main.py` does; the tests check this bit for bit.

## Run on Google Colab

Open [`ChildNet_GUI.ipynb`](ChildNet_GUI.ipynb) with the badge above, choose a GPU runtime, run all cells and open the `https://….gradio.live` link the last cell prints. The notebook downloads ChildNet and its weights and installs everything.

## Run locally

You need Python 3.10+, 8 GB of RAM and about 8 GB of free disk while the weights download (they take 3.6 GB once unpacked). An NVIDIA GPU is recommended. It also runs on CPU: on a 4-core machine each slider change took 2–4 s, and a new parent photo about a second longer.

```bash
# 1. ChildNet and its pretrained weights
git clone https://github.com/MartinPernus/ChildNet
(cd ChildNet && bash download.sh)

# 2. The GUI's dependencies, from the root of this repository.
#    For GPU support, install PyTorch for your CUDA version first (https://pytorch.org).
pip install -r childnet_gui/requirements.txt

# 3. Start it, then open http://127.0.0.1:7860
python -m childnet_gui --childnet-dir /path/to/ChildNet
```

On Windows without bash, download the archive linked in ChildNet's `download.sh` and extract it into `ChildNet/checkpoints/`.

| Option | Default | |
|---|---|---|
| `--childnet-dir PATH` | `$CHILDNET_DIR`, `.` or `./ChildNet` | ChildNet checkout containing `checkpoints/` |
| `--weights nokdb\|fiw` | first one available | weights to start with; switch in the GUI |
| `--device DEVICE` | `cuda` if available, else `cpu` | e.g. `cuda:1`; `mps` (Apple silicon) is untested |
| `--share` | off | also serve on a public `gradio.live` link |
| `--host`, `--port` | `127.0.0.1`, `7860` | `--host 0.0.0.0` makes it reachable from other machines |
| `--no-align` | | don't align faces, just centre-crop photos |
| `--landmarks PATH` | `<childnet-dir>/checkpoints/shape_predictor_68_face_landmarks.dat` | dlib's landmark model, downloaded on first run if missing |
| `--swap-gender` | | use if choosing *Girl* produces boys (see below) |
| `--cuda-ops` | off | build StyleGAN2's custom CUDA kernels (needs `nvcc`) |

## How it works

- [`engine.py`](engine.py) imports ChildNet from your checkout without modifying it. It re-implements only ChildNet's few-line latent blending step, so the dominant parent can be set per layer, and checks against the original in the tests. Parent latents are cached, so a slider change only re-runs the small latent-space networks and the StyleGAN decoder.
- Nothing needs compiling. ChildNet's encoder compiles two CUDA extensions at import time that inference never calls, so they are stubbed out. StyleGAN2's custom kernels are replaced by its built-in PyTorch reference implementations; `--cuda-ops` builds the kernels instead.
- Checkpoints load from any working directory and on PyTorch 2.6+, whose `torch.load` rejects pickled objects by default. The full unpickler is used only as a fallback, and only for files inside your ChildNet folder.
- [`align.py`](align.py) detects 68 face landmarks with dlib and crops and rotates the face the way FFHQ images are aligned, which is what ChildNet's encoder was trained on.

## Caveats

- **The gender labels are inferred.** ChildNet only documents gender as class 0 or 1, and this couldn't be checked against the pretrained weights while building the GUI. It assumes 0 = girl and 1 = boy, the order of the FFHQ-Aging labels. If *Girl* produces boys, restart with `--swap-gender`.
- **The age groups** are FFHQ-Aging's. The paper's figures label classes 3, 6 and 9 as 10–14, 30–39 and 70–120 years, consistent with this order.
- Results depend heavily on the photos. Clear, well-lit, roughly frontal faces work best; check the *Model's view* tab.
- These images are predictions shaped by ChildNet's training data, not genetic forecasts.
- Photos are processed on your machine (or your Colab VM). With `--share`, anyone who has the link can use the app while it runs.

## Tests

```bash
pip install pytest
CHILDNET_DIR=/path/to/ChildNet python -m pytest childnet_gui/tests
```

The tests run ChildNet's real code with random-weight checkpoints written to a temporary folder (about 4 GB), so they need no download. They check that the GUI reproduces ChildNet's own outputs exactly, and exercise every control and GUI event handler. To include the face-detection test, set `CHILDNET_LANDMARKS` to dlib's `shape_predictor_68_face_landmarks.dat`. Without `CHILDNET_DIR`, only the tests that don't need ChildNet run.

## Credits and licences

- [ChildNet](https://github.com/MartinPernus/ChildNet) is MIT-licensed by Martin Pernuš. It bundles [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) code under NVIDIA's non-commercial licence and [e4e](https://github.com/omertov/encoder4editing) (MIT).
- The face alignment in `align.py` is adapted from NVIDIA's [FFHQ dataset scripts](https://github.com/NVlabs/ffhq-dataset) (CC BY-NC-SA 4.0).
- dlib's [68-point landmark model](https://github.com/davisking/dlib-models) is trained on iBUG 300-W, whose licence excludes commercial use.

Taken together: use this for research and personal projects, not commercially.
