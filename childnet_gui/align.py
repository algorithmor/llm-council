"""Optional FFHQ-style face alignment for arbitrary parent photos.

ChildNet's encoder (e4e) was trained on FFHQ-aligned faces, like the examples
in the ChildNet repo. This module finds a face with dlib and crops/rotates it
the same way. Without dlib the GUI falls back to a centre crop, which only
works for photos already framed tightly around the face.
"""

from __future__ import annotations

import bz2
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

LANDMARKS_URL = "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2"
LANDMARKS_FILE = "shape_predictor_68_face_landmarks.dat"
DETECT_SIZE = 1024  # detect on a copy no larger than this, for speed on big photos


def download_landmarks(dest: Path) -> Path:
    """Download and unpack dlib's 68-point landmark model (~100 MB) to `dest`."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(LANDMARKS_URL, timeout=60) as response, open(partial, "wb") as out:
        shutil.copyfileobj(bz2.BZ2File(response), out)
    partial.replace(dest)
    return dest


def load_aligner(landmarks_path: Path, download: bool = True) -> tuple[FaceAligner | None, str]:
    """A FaceAligner, or None plus the reason alignment is unavailable."""
    try:
        import dlib  # noqa: F401
    except ImportError:
        return None, "Face alignment is off: install dlib (`pip install dlib-bin`) to enable it."
    if not landmarks_path.is_file():
        if not download:
            return None, f"Face alignment is off: landmark model not found at {landmarks_path}."
        print(f"Downloading dlib's face landmark model to {landmarks_path} ...", flush=True)
        try:
            download_landmarks(landmarks_path)
        except OSError as e:
            return None, f"Face alignment is off: could not download the landmark model ({e})."
    return FaceAligner(landmarks_path), "Faces are aligned FFHQ-style with dlib."


class FaceAligner:
    def __init__(self, landmarks_path: Path):
        import dlib

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(landmarks_path))

    def landmarks(self, image: Image.Image) -> np.ndarray | None:
        """dlib's 68 (x, y) landmarks of the largest face, or None if there is no face."""
        image = image.convert("RGB")
        scale = min(1.0, DETECT_SIZE / max(image.size))
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        pixels = np.asarray(image)
        faces = self._detector(pixels, 1)
        if not faces:
            return None
        face = max(faces, key=lambda rect: rect.area())
        points = self._predictor(pixels, face).parts()
        return np.array([(p.x, p.y) for p in points], dtype=np.float64) / scale

    def __call__(self, image: Image.Image, size: int = 256) -> Image.Image | None:
        """The FFHQ-aligned face crop, or None if no face was found."""
        landmarks = self.landmarks(image)
        return None if landmarks is None else ffhq_align(image, landmarks, size)


def ffhq_align(image: Image.Image, landmarks: np.ndarray, size: int = 256) -> Image.Image:
    """Crop and rotate a face the way FFHQ images (and e4e's inputs) are aligned.

    Adapted from `recreate_aligned_images` in NVIDIA's ffhq-dataset download
    script (Karras et al., "A Style-Based Generator Architecture for Generative
    Adversarial Networks", CVPR 2019), (c) NVIDIA Corporation, licensed
    CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/), and so
    is this function. Changes: e4e's output settings, a Pillow padding blur
    instead of SciPy's, and restructured code.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    eye_left, eye_right = lm[36:42].mean(0), lm[42:48].mean(0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm[48] + lm[54]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = image.convert("RGB")
    shrink = int(np.floor(qsize / size * 0.5))
    if shrink > 1:
        img = img.resize((int(np.rint(img.width / shrink)), int(np.rint(img.height / shrink))), Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop to the face plus a border.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        max(int(np.floor(min(quad[:, 0]))) - border, 0),
        max(int(np.floor(min(quad[:, 1]))) - border, 0),
        min(int(np.ceil(max(quad[:, 0]))) + border, img.width),
        min(int(np.ceil(max(quad[:, 1]))) + border, img.height),
    )
    if crop[2] - crop[0] < img.width or crop[3] - crop[1] < img.height:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad with blurred reflections where the crop leaves the photo.
    pad = (
        max(-int(np.floor(min(quad[:, 0]))) + border, 0),
        max(-int(np.floor(min(quad[:, 1]))) + border, 0),
        max(int(np.ceil(max(quad[:, 0]))) - img.width + border, 0),
        max(int(np.ceil(max(quad[:, 1]))) - img.height + border, 0),
    )
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        arr = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = arr.shape
        yy, xx, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(w - 1 - xx) / pad[2]),
            1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(h - 1 - yy) / pad[3]),
        )
        blurred = np.float32(_to_image(arr).filter(ImageFilter.GaussianBlur(float(qsize * 0.02))))
        arr += (blurred - arr) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        arr += (np.median(arr, axis=(0, 1)) - arr) * np.clip(mask, 0.0, 1.0)
        img = _to_image(arr)
        quad += pad[:2]

    return img.transform((size, size), Image.Transform.QUAD, (quad + 0.5).flatten(), Image.Resampling.BILINEAR)


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(np.clip(np.rint(arr), 0, 255)), "RGB")
