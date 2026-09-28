"""Gradio GUI for ChildNet: predict a child's face from photos of both parents.

    python -m childnet_gui --childnet-dir /path/to/ChildNet [--share]
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from .align import LANDMARKS_FILE, load_aligner
from .engine import AGE_GROUPS, ChildNetEngine, Controls, center_square, encoder_input, to_pil

WEIGHT_LABELS = {"nokdb": "Next of Kin (NoKDB)", "fiw": "Families in the Wild (FIW)"}
# ChildNet only documents gender as class 0 or 1. Girl = 0, Boy = 1 is inferred
# from the FFHQ-Aging labels it was trained with; --swap-gender flips it.
GENDERS = ("Girl", "Boy")
# Order of the control components passed to every generating event handler.
CONTROL_NAMES = ("dominance", "per_level", "coarse", "medium", "fine", "age_gender", "age", "gender", "strength", "variability", "seed")
SAMPLING_RATE = 0.5  # ChildNet's --sample dropout rate
SWEEP_LABELS = {"Dominant parent": "Sweep (-1 = mother, +1 = father)", "Age group": "Sweep (age groups in years)"}
PARENT_CACHE_SIZE = 8
THEME = gr.themes.Soft(primary_hue="indigo")
# Gradio 6 moved `theme` from the Blocks constructor to launch().
THEME_IN_LAUNCH = "theme" in inspect.signature(gr.Blocks.launch).parameters


def age_label(age: float) -> str:
    return f"Age group: {AGE_GROUPS[int(age)]} years"


class ChildNetApp:
    """Event handlers and layout of the GUI around a loaded ChildNetEngine."""

    def __init__(self, engine: ChildNetEngine, aligner=None, align_note: str = "", swap_gender: bool = False):
        self.engine = engine
        self.aligner = aligner
        self.align_note = align_note
        self.gender_classes = {name: (1 - i if swap_gender else i) for i, name in enumerate(GENDERS)}
        self._faces: OrderedDict[tuple, tuple[Image.Image, str]] = OrderedDict()

    # --------------------------------------------------------------- inputs

    def face(self, path: str, role: str, align: bool) -> tuple[Image.Image, str]:
        """The face crop the model gets for an uploaded photo, plus a status note."""
        align = bool(align and self.aligner)
        key = (path, os.path.getmtime(path), align)
        if key not in self._faces:
            photo = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            face = self.aligner(photo) if align else None
            note = f"No face found in the {role} photo, so it was centre-cropped instead." if align and face is None else ""
            self._faces[key] = (center_square(photo) if face is None else face, note)
            while len(self._faces) > PARENT_CACHE_SIZE:
                self._faces.popitem(last=False)
        self._faces.move_to_end(key)
        return self._faces[key]

    def parents(self, father: str | None, mother: str | None, align: bool):
        """((father_face, mother_face), notes), or (None, hint) until both photos are there."""
        if not father or not mother:
            return None, "Add a photo of each parent (or click the example pair) to begin."
        (f, f_note), (m, m_note) = self.face(father, "father", align), self.face(mother, "mother", align)
        return (f, m), " ".join(note for note in (f_note, m_note) if note)

    def controls(self, dominance, per_level, coarse, medium, fine, age_gender, age, gender, strength, variability, seed) -> Controls:
        return Controls(
            dominance=dominance,
            level_dominance=(coarse, medium, fine) if per_level else None,
            age=int(age) if age_gender else None,
            gender=self.gender_classes[gender] if age_gender else None,
            edit_strength=strength,
            variability=variability,
            seed=int(seed or 0),
        )

    def status(self, start: float, *notes: str) -> str:
        engine = self.engine
        timing = f"{time.perf_counter() - start:.2f} s on `{engine.device}` with {WEIGHT_LABELS[engine.weights]} weights."
        return " ".join((timing, *(n for n in notes if n)))

    # --------------------------------------------------------------- events

    def generate(self, father, mother, align, *control_values):
        parents, notes = self.parents(father, mother, align)
        if parents is None:
            return None, notes
        start = time.perf_counter()
        [child] = self.engine.generate(*parents, self.controls(*control_values))
        return child, self.status(start, notes)

    def generate_live(self, live, last_request, *args):
        """Regenerate after a control change, unless nothing that matters changed.

        One action can fire several change events (an example sets both
        parents), so this session's last request is remembered and repeats are skipped.
        """
        request = (self.engine.weights, *args)
        if not live or request == last_request:
            return gr.update(), gr.update(), last_request
        return *self.generate(*args), request

    def siblings(self, father, mother, align, count, *control_values):
        parents, notes = self.parents(father, mother, align)
        if parents is None:
            return [], notes
        start = time.perf_counter()
        controls, sampling_note = self.controls(*control_values), ""
        if controls.variability == 0:
            controls = replace(controls, variability=SAMPLING_RATE)
            sampling_note = f"Variability is 0, which would make identical siblings, so ChildNet's sampling rate {SAMPLING_RATE} was used."
        images = self.engine.generate(*parents, controls, num_samples=int(count))
        gallery = [(image, f"seed {controls.seed + i}") for i, image in enumerate(images)]
        return gallery, self.status(start, notes, sampling_note)

    def sweep(self, father, mother, align, target, steps, *control_values):
        parents, notes = self.parents(father, mother, align)
        if parents is None:
            return [], notes
        start = time.perf_counter()
        values = dict(zip(CONTROL_NAMES, control_values, strict=True))
        base = self.controls(**values)
        if target == "Age group":
            gender = self.gender_classes[values["gender"]]
            runs = [(replace(base, age=age, gender=gender), f"{group} years") for age, group in enumerate(AGE_GROUPS)]
        else:
            steps = int(steps)
            dominances = [round(-1 + 2 * i / (steps - 1), 2) for i in range(steps)]
            runs = [(replace(base, dominance=d, level_dominance=None), f"{d:+.2f}") for d in dominances]
        gallery = [(self.engine.generate(*parents, controls)[0], caption) for controls, caption in runs]
        return gallery, self.status(start, notes)

    def model_view(self, father, mother, align):
        parents, notes = self.parents(father, mother, align)
        if parents is None:
            return [], notes
        start = time.perf_counter()
        gallery = []
        for role, face in zip(("Father", "Mother"), parents, strict=True):
            gallery.append((to_pil(encoder_input(face)), f"{role}: model input"))
            gallery.append((self.engine.reconstruct(face), f"{role}: reconstruction"))
        return gallery, self.status(start, notes)

    def load_weights(self, weights: str) -> str:
        start = time.perf_counter()
        try:
            self.engine.load(weights)
        except FileNotFoundError as e:
            raise gr.Error(str(e)) from e
        return f"Loaded {WEIGHT_LABELS[weights]} weights in {time.perf_counter() - start:.1f} s."

    # --------------------------------------------------------------- layout

    def build(self) -> gr.Blocks:
        engine = self.engine
        examples = [str(engine.childnet_dir / "imgs" / f"{p}.jpg") for p in ("father", "mother")]
        blocks_kwargs = {} if THEME_IN_LAUNCH else {"theme": THEME}

        with gr.Blocks(title="ChildNet", **blocks_kwargs) as demo:
            gr.Markdown(
                "# ChildNet: what might your child look like?\n"
                "Upload a photo of each parent and move the sliders. Powered by "
                "[ChildNet](https://github.com/MartinPernus/ChildNet) (Pernuš et al., IEEE Access 2023), "
                f"running on `{engine.device}`."
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    with gr.Row():
                        father = gr.Image(label="Father", type="filepath", height=260)
                        mother = gr.Image(label="Mother", type="filepath", height=260)
                    if all(os.path.isfile(p) for p in examples):
                        gr.Examples([examples], inputs=[father, mother], label="Example parents from the ChildNet repo")
                    with gr.Row():
                        align = gr.Checkbox(
                            label="Auto-align faces",
                            value=self.aligner is not None,
                            interactive=self.aligner is not None,
                            info=self.align_note,
                        )
                        weights = gr.Dropdown(
                            [(WEIGHT_LABELS[w], w) for w in engine.available_weights()],
                            value=engine.weights,
                            label="Model weights",
                        )

                    dominance = gr.Slider(
                        -1, 1, value=0, step=0.05, label="Dominant parent",
                        info="-1 = like the mother, 0 = ChildNet's own blend, +1 = like the father",
                    )
                    with gr.Group():
                        age_gender = gr.Checkbox(label="Set the child's age and gender", value=True)
                        age = gr.Slider(0, len(AGE_GROUPS) - 1, value=2, step=1, label=age_label(2))
                        gender = gr.Radio(list(GENDERS), value=GENDERS[0], label="Gender")
                        strength = gr.Slider(
                            0, 2, value=1, step=0.05, label="Age/gender edit strength",
                            info="1 = ChildNet's edit, 0 = no edit, above 1 = exaggerated",
                        )
                    with gr.Accordion("Dominant parent per facial level", open=False):
                        per_level = gr.Checkbox(label="Set the dominant parent separately for each level", value=False)
                        coarse = gr.Slider(-1, 1, value=0, step=0.05, interactive=False, label="Coarse: face shape and pose")
                        medium = gr.Slider(-1, 1, value=0, step=0.05, interactive=False, label="Medium: facial features")
                        fine = gr.Slider(-1, 1, value=0, step=0.05, interactive=False, label="Fine: colouring, skin and hair texture")
                    with gr.Accordion("Variability", open=False):
                        variability = gr.Slider(
                            0, 0.75, value=0, step=0.05, label="Variability",
                            info="Dropout rate while sampling: 0 = always the same child, 0.5 = ChildNet's --sample mode",
                        )
                        with gr.Row():
                            seed = gr.Number(value=0, precision=0, label="Seed")
                            dice = gr.Button("New random seed")
                    live = gr.Checkbox(label="Update automatically when a control changes", value=True)

                with gr.Column(scale=4):
                    with gr.Tabs():
                        with gr.Tab("Child"):
                            child = gr.Image(label="Predicted child", type="pil", format="png", height=512, interactive=False)
                            generate_btn = gr.Button("Generate", variant="primary")
                        with gr.Tab("Siblings"):
                            count = gr.Slider(2, 8, value=4, step=1, label="Number of siblings")
                            siblings_btn = gr.Button("Generate siblings", variant="primary")
                            siblings_gallery = gr.Gallery(label="Siblings (random samples)", columns=4, format="png")
                        with gr.Tab("Sweep"):
                            target = gr.Radio(["Dominant parent", "Age group"], value="Dominant parent", label="Vary")
                            steps = gr.Slider(3, 9, value=5, step=1, label="Steps")
                            sweep_btn = gr.Button("Run sweep", variant="primary")
                            sweep_gallery = gr.Gallery(label=SWEEP_LABELS["Dominant parent"], columns=5, format="png")
                        with gr.Tab("Model's view"):
                            gr.Markdown(
                                "The face crops ChildNet receives, and how its encoder reconstructs each parent. "
                                "If a reconstruction doesn't look like the parent, the child won't inherit their features."
                            )
                            view_btn = gr.Button("Show", variant="primary")
                            view_gallery = gr.Gallery(label="Model's view", columns=2, format="png")
                    status = gr.Markdown()

            parent_inputs = [father, mother, align]
            control_inputs = [dominance, per_level, coarse, medium, fine, age_gender, age, gender, strength, variability, seed]
            generate_inputs = [*parent_inputs, *control_inputs]
            last_request = gr.State(None)
            live_update = dict(fn=self.generate_live, inputs=[live, last_request, *generate_inputs], outputs=[child, status, last_request])

            generate_btn.click(self.generate, generate_inputs, [child, status])
            gr.on(
                [father.change, mother.change, align.change, gender.change, seed.change]
                + [s.release for s in (dominance, coarse, medium, fine, age, strength, variability)],
                trigger_mode="always_last",
                **live_update,
            )
            # These update other controls first, so they regenerate only after that.
            # (Each output needs its own update dict; Gradio consumes their keys.)
            age_gender.change(
                lambda on: [gr.update(interactive=on) for _ in range(3)], age_gender, [age, gender, strength]
            ).then(**live_update)
            per_level.change(
                lambda on, d: [gr.update(interactive=not on)] + [gr.update(interactive=on, value=d) for _ in range(3)],
                [per_level, dominance],
                [dominance, coarse, medium, fine],
            ).then(**live_update)
            weights.change(self.load_weights, weights, status).then(**live_update)

            siblings_btn.click(self.siblings, [*parent_inputs, count, *control_inputs], [siblings_gallery, status])
            sweep_btn.click(self.sweep, [*parent_inputs, target, steps, *control_inputs], [sweep_gallery, status])
            view_btn.click(self.model_view, parent_inputs, [view_gallery, status])
            dice.click(lambda: random.randrange(2**31), None, seed)
            age.change(lambda a: gr.update(label=age_label(a)), age, age)
            target.change(
                lambda t: (gr.update(visible=t == "Dominant parent"), gr.update(label=SWEEP_LABELS[t])),
                target,
                [steps, sweep_gallery],
            )
        return demo


def find_childnet_dir(path: str | None) -> Path:
    candidates = [path] if path else [os.environ.get("CHILDNET_DIR"), ".", "ChildNet"]
    for candidate in filter(None, candidates):
        if (Path(candidate) / "models" / "childnet.py").is_file():
            return Path(candidate).resolve()
    raise SystemExit(
        "Could not find ChildNet. Clone https://github.com/MartinPernus/ChildNet, run `bash download.sh` "
        "in it, then pass its path with --childnet-dir (or set CHILDNET_DIR)."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gradio GUI for ChildNet kinship face synthesis.")
    parser.add_argument("--childnet-dir", help="ChildNet checkout with downloaded checkpoints (default: $CHILDNET_DIR, ., ./ChildNet)")
    parser.add_argument("--weights", choices=sorted(WEIGHT_LABELS), help="model weights to start with (default: first available)")
    parser.add_argument("--device", help="torch device, e.g. cuda, cuda:1, cpu, mps (default: cuda if available, else cpu)")
    parser.add_argument("--cuda-ops", action="store_true", help="build StyleGAN2's custom CUDA kernels (needs nvcc; slightly faster)")
    parser.add_argument("--no-align", action="store_true", help="don't auto-align faces; photos are centre-cropped")
    parser.add_argument("--landmarks", type=Path, help=f"dlib landmark model (default: <childnet-dir>/checkpoints/{LANDMARKS_FILE}, downloaded if missing)")
    parser.add_argument("--swap-gender", action="store_true", help="use if choosing Girl produces boys and vice versa")
    parser.add_argument("--share", action="store_true", help="also serve on a public gradio.live link (needed on Colab)")
    parser.add_argument("--host", default=None, help="interface to listen on, e.g. 0.0.0.0 (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="port to listen on (default: 7860 or the next free one)")
    args = parser.parse_args(argv)

    childnet_dir = find_childnet_dir(args.childnet_dir)
    engine = ChildNetEngine(childnet_dir, device=args.device, cuda_ops=args.cuda_ops)
    available = engine.available_weights()
    if not available:
        missing = ", ".join(engine.missing_checkpoints("nokdb"))
        raise SystemExit(f"No ChildNet checkpoints in {childnet_dir / 'checkpoints'} (missing {missing}). Run `bash download.sh` in {childnet_dir}.")
    weights = args.weights or available[0]
    print(f"Loading ChildNet ({WEIGHT_LABELS[weights]}) on {engine.device} ...", flush=True)
    try:
        engine.load(weights)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    if args.no_align:
        aligner, note = None, "Face alignment is off (--no-align)."
    else:
        aligner, note = load_aligner(args.landmarks or childnet_dir / "checkpoints" / LANDMARKS_FILE)
    print(note, flush=True)

    demo = ChildNetApp(engine, aligner, note, args.swap_gender).build()
    launch_kwargs = {"theme": THEME} if THEME_IN_LAUNCH else {}
    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port,
        show_error=True,
        allowed_paths=[str(childnet_dir / "imgs")],
        **launch_kwargs,
    )
