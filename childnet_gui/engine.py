"""Inference wrapper around the official ChildNet code, used by the GUI.

ChildNet (https://github.com/MartinPernus/ChildNet) is imported from a checkout
directory and used unmodified. On top of its forward pass this module adds:

* parent-latent caching, so moving a slider only re-runs the small latent-space
  networks and the StyleGAN decoder instead of re-encoding both parents;
* the controls ChildNet exposes (dominant parent, age, gender, dropout
  sampling) plus two latent-space extensions: per-level dominance and an
  age/gender edit strength;
* portability fixes: no CUDA extension builds (so it also runs on CPU) and
  checkpoint loading on PyTorch >= 2.6.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import pickle
import sys
import threading
import types
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Pretrained ChildNet variants: trained on Next of Kin and on Families in the Wild.
WEIGHTS = ("nokdb", "fiw")
# ChildNet's 10 age classes are the FFHQ-Aging age groups, youngest first.
AGE_GROUPS = ("0-2", "3-6", "7-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-69", "70+")
# W+ layer ranges that ChildNet's StructuralMerge models separately (18 layers at 1024 px).
LEVELS = (("coarse", slice(0, 4)), ("medium", slice(4, 8)), ("fine", slice(8, None)))
# Checkpoints every variant needs, next to checkpoints/childnet_<weights>.pt.
SHARED_CHECKPOINTS = ("G_kwargs.pt", "encoder_state_dict.pt", "latent_avg.pt", "disentanglement.pt")
ENCODER_SIZE = 256  # e4e's input resolution
LATENT_CACHE_SIZE = 16


@dataclass(frozen=True)
class Controls:
    """Everything that shapes a child image apart from the two parents."""

    # ChildNet's move2parent: -1 = mother, 0 = the model's own blend, +1 = father.
    dominance: float = 0.0
    # Optional (coarse, medium, fine) dominance that replaces `dominance` per level.
    level_dominance: tuple[float, float, float] | None = None
    # Age class (index into AGE_GROUPS) and gender class; both or neither.
    age: int | None = None
    gender: int | None = None
    # Scales the age/gender latent edit; 1 reproduces ChildNet.
    edit_strength: float = 1.0
    # Dropout rate of the kinship module at inference: 0 is deterministic,
    # 0.5 is ChildNet's --sample mode.
    variability: float = 0.0
    seed: int = 0


class ChildNetEngine:
    """Loads ChildNet from a checkout and turns parent photos into child images.

    Public methods are thread-safe: a lock serialises all model access, since
    sampling temporarily reconfigures the model's dropout layers.
    """

    def __init__(self, childnet_dir: str | os.PathLike, device: str | None = None, cuda_ops: bool = False):
        self.childnet_dir = Path(childnet_dir).expanduser().resolve()
        if not (self.childnet_dir / "models" / "childnet.py").is_file():
            raise FileNotFoundError(
                f"{self.childnet_dir} is not a ChildNet checkout (models/childnet.py is missing). "
                "Clone https://github.com/MartinPernus/ChildNet and pass its path."
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights: str | None = None
        self._childnet_cls = _import_childnet(self.childnet_dir, cuda_ops)
        self._model = None
        self._latents: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ loading

    def missing_checkpoints(self, weights: str) -> list[str]:
        names = (*SHARED_CHECKPOINTS, f"childnet_{weights}.pt")
        return [name for name in names if not (self.childnet_dir / "checkpoints" / name).is_file()]

    def available_weights(self) -> list[str]:
        return [w for w in WEIGHTS if not self.missing_checkpoints(w)]

    def load(self, weights: str) -> None:
        """Load a pretrained variant ("nokdb" or "fiw"), replacing the current one."""
        if weights not in WEIGHTS:
            raise ValueError(f"Unknown weights {weights!r}; expected one of {WEIGHTS}")
        with self._lock:
            if weights == self.weights:
                return
            missing = self.missing_checkpoints(weights)
            if missing:
                raise FileNotFoundError(
                    f"Missing ChildNet checkpoints in {self.childnet_dir / 'checkpoints'}: {', '.join(missing)}. "
                    "Run `bash download.sh` inside the ChildNet folder."
                )
            # Free the previous model first: each variant holds a ~1 GB encoder.
            self._model, self.weights = None, None
            self._latents.clear()
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            with _childnet_checkpoints(self.childnet_dir):
                model = self._childnet_cls(weights)
            self._model = model.to(self.device).eval()
            self.weights = weights

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("No ChildNet weights loaded; call load() first.")
        return self._model

    # ---------------------------------------------------------------- inference

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """e4e W+ latent (1, 18, 512) of a face image, cached per image."""
        digest = hashlib.sha1(image.convert("RGB").tobytes()).hexdigest()
        with self._lock:
            key = (self.weights, image.size, digest)
            latent = self._latents.get(key)
            if latent is None:
                latent = self.model.kinship_model.e4e(encoder_input(image).to(self.device))
                self._latents[key] = latent
                while len(self._latents) > LATENT_CACHE_SIZE:
                    self._latents.popitem(last=False)
            self._latents.move_to_end(key)
            return latent

    @torch.no_grad()
    def child_latent(self, w_father: torch.Tensor, w_mother: torch.Tensor, controls: Controls = Controls()) -> torch.Tensor:
        """ChildNet's latent-space pipeline (kinship module + age/gender edit)."""
        with self._lock:
            model = self.model
            gene_model = model.kinship_model.gene_model
            with _dropout(gene_model, _check_range("variability", controls.variability, 0.0, 0.95)):
                if controls.variability > 0:
                    torch.manual_seed(int(controls.seed))
                dominance = _layer_dominance(controls, w_father.shape[1], w_father.device)
                w = _merge_parents(gene_model, w_father, w_mother, dominance)
            if controls.age is not None and controls.gender is not None:
                if controls.age not in range(len(AGE_GROUPS)) or controls.gender not in (0, 1):
                    raise ValueError(f"age must be 0-{len(AGE_GROUPS) - 1} and gender 0 or 1")
                age = torch.full((w.shape[0],), controls.age, dtype=torch.long, device=w.device)
                gender = torch.full((w.shape[0],), controls.gender, dtype=torch.long, device=w.device)
                edited, delta = model.disentangle(w, age, gender)
                w = edited if controls.edit_strength == 1 else w + controls.edit_strength * delta
            return w

    @torch.no_grad()
    def synthesize(self, w: torch.Tensor) -> list[Image.Image]:
        """Decode W+ latents with ChildNet's StyleGAN2 generator (1024x1024)."""
        with self._lock:
            return [to_pil(self.model.kinship_model.decoder(w[i : i + 1])) for i in range(w.shape[0])]

    def generate(self, father: Image.Image, mother: Image.Image, controls: Controls = Controls(), num_samples: int = 1) -> list[Image.Image]:
        """Child images for a parent pair; sample i uses seed `controls.seed + i`."""
        with self._lock:
            w_father, w_mother = self.encode(father), self.encode(mother)
            images = []
            for i in range(num_samples):
                w = self.child_latent(w_father, w_mother, replace(controls, seed=controls.seed + i))
                images += self.synthesize(w)
            return images

    def reconstruct(self, image: Image.Image) -> Image.Image:
        """How the encoder sees a parent: decode its latent without any kinship modelling."""
        with self._lock:
            return self.synthesize(self.encode(image))[0]


# ---------------------------------------------------------------------- images


def encoder_input(image: Image.Image) -> torch.Tensor:
    """(1, 3, 256, 256) tensor in [0, 1], as ChildNet's encoder receives it.

    Non-square images are centre-cropped rather than squashed. Resizing uses the
    same area interpolation as e4e's own downsampling, done here on the CPU so
    that the device never sees full-resolution photos.
    """
    image = center_square(image.convert("RGB"))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]
    if x.shape[-1] != ENCODER_SIZE:
        x = F.interpolate(x, (ENCODER_SIZE, ENCODER_SIZE), mode="area")
    return x


def center_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return image.crop((left, top, left + side, top + side)) if w != h else image


def to_pil(x: torch.Tensor) -> Image.Image:
    """First image of a [0, 1] NCHW batch, rounded like torchvision's save_image."""
    x = x[0].detach().float().mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
    return Image.fromarray(x.permute(1, 2, 0).numpy())


# --------------------------------------------------------------- latent space


def _layer_dominance(controls: Controls, num_layers: int, device: torch.device) -> torch.Tensor:
    """Per-layer move2parent values, shaped (1, num_layers, 1)."""
    values = controls.level_dominance or (controls.dominance,) * len(LEVELS)
    dominance = torch.empty(1, num_layers, 1, device=device)
    for (_, layers), value in zip(LEVELS, values, strict=True):
        dominance[:, layers] = _check_range("dominance", value, -1.0, 1.0)
    return dominance


def _merge_parents(gene_model, w_father: torch.Tensor, w_mother: torch.Tensor, dominance: torch.Tensor) -> torch.Tensor:
    """ChildNet's GeneModel.forward, with move2parent given per W+ layer.

    With the same value on every layer this matches the original exactly: the
    attention weights alpha are pushed towards 1 (father) or 0 (mother).
    """
    alpha = torch.sigmoid(gene_model.merge_mu_att(w_father, w_mother))
    alpha = torch.where(dominance > 0, alpha + dominance * (1 - alpha), alpha + dominance * alpha)
    w = alpha * w_father + (1 - alpha) * w_mother
    return w + gene_model.merge_mu_res(w_father, w_mother) * gene_model.w_std


@contextlib.contextmanager
def _dropout(module: torch.nn.Module, p: float):
    """Run `module`'s dropout layers at rate `p` (0 disables them, as in eval mode)."""
    layers = [m for m in module.modules() if isinstance(m, torch.nn.Dropout)]
    saved = [(m.p, m.training) for m in layers]
    for m in layers:
        m.p = p
        m.train(p > 0)
    try:
        yield
    finally:
        for m, (rate, training) in zip(layers, saved, strict=True):
            m.p = rate
            m.train(training)


def _check_range(name: str, value: float, low: float, high: float) -> float:
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
    return value


# ------------------------------------------------------------- ChildNet import


def _import_childnet(childnet_dir: Path, cuda_ops: bool):
    """Import ChildNet's `models` package from `childnet_dir` and return its ChildNet class.

    The package has no __init__.py and a very generic name, so it is registered
    explicitly instead of relying on sys.path, where any other `models` package
    would shadow it.
    """
    package = sys.modules.get("models")
    models_dir = str(childnet_dir / "models")
    if package is None:
        spec = importlib.machinery.ModuleSpec("models", None, is_package=True)
        spec.submodule_search_locations = [models_dir]
        package = importlib.util.module_from_spec(spec)
        sys.modules["models"] = package
    elif models_dir not in list(getattr(package, "__path__", [])):
        raise ImportError(f"A different 'models' package is already imported: {package}")
    _install_e4e_op_stub()
    childnet = importlib.import_module("models.childnet")
    if not cuda_ops:
        # StyleGAN2-ADA builds its CUDA kernels on first use and falls back to
        # reference PyTorch ops when that fails. Skip the build: it needs nvcc,
        # can take minutes, and hangs on stale locks after an interrupted build.
        from models.stylegan.torch_utils.ops import bias_act, upfirdn2d

        bias_act._init = upfirdn2d._init = lambda: False
    return childnet.ChildNet


def _install_e4e_op_stub() -> None:
    """Register e4e's `op` package without compiling its CUDA extensions.

    `models/e4e/models/stylegan2/op` JIT-compiles two CUDA extensions at import
    time, which fails without nvcc (always, on CPU-only machines). ChildNet only
    imports `EqualLinear` from that code and uses it without an activation, which
    never calls these ops, so stand-ins that raise if ever called are enough.
    """
    name = "models.e4e.models.stylegan2.op"
    if name in sys.modules:
        return

    def unavailable(*args, **kwargs):
        raise RuntimeError("e4e's fused CUDA ops are stubbed out by childnet_gui; ChildNet inference does not use them.")

    op = types.ModuleType(name)
    op.FusedLeakyReLU = op.fused_leaky_relu = op.upfirdn2d = unavailable
    sys.modules[name] = op


@contextlib.contextmanager
def _childnet_checkpoints(childnet_dir: Path):
    """Context for constructing ChildNet from any working directory, quickly.

    * Resolves ChildNet's relative `checkpoints/...` loads against `childnet_dir`.
    * Keeps them loadable on PyTorch >= 2.6, where torch.load defaults to
      weights_only=True and rejects pickled Python objects; the full unpickler
      is only a fallback, and only for files inside `childnet_dir`.
    * Skips work the checkpoints overwrite anyway: random weight initialisation
      and the 100k-sample W statistics that seed GeneModel.w_std. ChildNet loads
      every state dict strictly, so all parameters and buffers get replaced.
    """
    from models.stylegan.utils import StyleGAN

    original_load = torch.load

    def load(f, *args, **kwargs):
        if isinstance(f, (str, os.PathLike)):
            f = childnet_dir / f
            try:
                return original_load(f, *args, **kwargs)
            except pickle.UnpicklingError:
                if "weights_only" in kwargs or childnet_dir not in f.resolve().parents:
                    raise
                return original_load(f, *args, weights_only=False, **kwargs)
        return original_load(f, *args, **kwargs)

    def skip_init(tensor, *args, **kwargs):
        return tensor

    def placeholder_stats(self, n_sample=None):
        return torch.zeros(1, 512), torch.ones(1, 512)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(torch, "load", load))
        stack.enter_context(mock.patch.object(StyleGAN, "get_latent_stats", placeholder_stats))
        for init in ("kaiming_uniform_", "uniform_", "normal_"):
            stack.enter_context(mock.patch.object(torch.nn.init, init, skip_init))
        yield
