import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ..align import LANDMARKS_FILE, FaceAligner, ffhq_align, load_aligner


def synthetic_face(angle_deg: float, center=(500.0, 400.0), eye_distance=100.0, size=(1000, 800)):
    """An image with red/blue dots on the eyes, and 68 landmarks of a face rotated by `angle_deg`."""
    a = np.deg2rad(angle_deg)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    upright = {"eye_left": (-eye_distance / 2, 0.0), "eye_right": (eye_distance / 2, 0.0),
               "mouth_left": (-30.0, eye_distance), "mouth_right": (30.0, eye_distance)}
    points = {k: rotation @ np.array(v) + center for k, v in upright.items()}
    landmarks = np.zeros((68, 2))
    landmarks[36:42], landmarks[42:48] = points["eye_left"], points["eye_right"]
    landmarks[48], landmarks[54] = points["mouth_left"], points["mouth_right"]
    image = Image.new("RGB", size, (128, 128, 128))
    draw = ImageDraw.Draw(image)
    for key, color in (("eye_left", (255, 0, 0)), ("eye_right", (0, 0, 255))):
        x, y = points[key]
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
    return image, landmarks


def centroid(image: Image.Image, color) -> np.ndarray:
    pixels = np.asarray(image).astype(int)
    mask = np.abs(pixels - color).sum(-1) < 120
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


@pytest.mark.parametrize("angle", [0, 20, -35])
def test_ffhq_align_levels_and_frames_the_face(angle):
    image, landmarks = synthetic_face(angle)
    aligned = ffhq_align(image, landmarks, size=256)
    assert aligned.size == (256, 256)
    left, right = centroid(aligned, (255, 0, 0)), centroid(aligned, (0, 0, 255))
    # FFHQ framing: the crop is 4x the eye distance wide, centred 0.1 * eye-to-mouth below the eyes.
    np.testing.assert_allclose(left, [96, 121.6], atol=2)
    np.testing.assert_allclose(right, [160, 121.6], atol=2)


def test_ffhq_align_pads_faces_at_the_edge():
    image, landmarks = synthetic_face(10, center=(100.0, 90.0), size=(400, 300))  # crop reaches ~180 px past the edges
    aligned = ffhq_align(image, landmarks, size=128)
    assert aligned.size == (128, 128)
    np.testing.assert_allclose(centroid(aligned, (255, 0, 0)), [48, 60.8], atol=2)


def test_load_aligner_without_landmark_model(tmp_path):
    pytest.importorskip("dlib")
    aligner, note = load_aligner(tmp_path / LANDMARKS_FILE, download=False)
    assert aligner is None and "not found" in note


def landmark_model() -> Path:
    candidates = [os.environ.get("CHILDNET_LANDMARKS"), Path(os.environ.get("CHILDNET_DIR", "")) / "checkpoints" / LANDMARKS_FILE]
    for candidate in filter(None, candidates):
        if Path(candidate).is_file():
            return Path(candidate)
    pytest.skip("set CHILDNET_LANDMARKS to dlib's shape_predictor_68_face_landmarks.dat")


def test_face_aligner_recovers_a_face_from_a_casual_photo():
    pytest.importorskip("dlib")
    source = Path(os.environ.get("CHILDNET_DIR", "")) / "imgs" / "father.jpg"
    if not source.is_file():
        pytest.skip("set CHILDNET_DIR to a ChildNet checkout")
    aligner = FaceAligner(landmark_model())
    face = Image.open(source).convert("RGB")
    background = (90, 110, 130)
    photo = Image.new("RGB", (1600, 1200), background)
    tilted = face.resize((420, 420)).rotate(18, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=background)
    photo.paste(tilted, (900, 250))

    recovered, reference = aligner(photo), aligner(face)
    mse = np.mean((np.float32(recovered) - np.float32(reference)) ** 2)
    assert 10 * np.log10(255**2 / mse) > 20  # PSNR: the same crop, up to resampling
    assert aligner(Image.new("RGB", (640, 480), background)) is None
