import gradio as gr
import pytest
from PIL import Image

from ..app import CONTROL_NAMES, ChildNetApp, find_childnet_dir
from ..engine import AGE_GROUPS, Controls

DEFAULTS = dict(dominance=0.0, per_level=False, coarse=0.0, medium=0.0, fine=0.0, age_gender=True,
                age=2, gender="Girl", strength=1.0, variability=0.0, seed=0)


def values(**overrides):
    """UI control values in the order the event handlers receive them."""
    return [{**DEFAULTS, **overrides}[name] for name in CONTROL_NAMES]


@pytest.fixture
def app(engine):
    return ChildNetApp(engine)


def test_controls_map_ui_values(app):
    assert app.controls(*values(age_gender=False)) == Controls()
    assert app.controls(*values(dominance=0.3, age=4.0, gender="Boy", seed=7.0)) == Controls(dominance=0.3, age=4, gender=1, seed=7)
    assert app.controls(*values(per_level=True, coarse=0.1, medium=0.2, fine=0.3)).level_dominance == (0.1, 0.2, 0.3)


def test_swap_gender(engine):
    assert ChildNetApp(engine, swap_gender=True).controls(*values(gender="Girl")).gender == 1


def test_generate_needs_both_parents(app, parent_paths):
    child, status = app.generate(parent_paths[0], None, True, *values())
    assert child is None and "each parent" in status


def test_generate(app, parent_paths):
    child, status = app.generate(*parent_paths, True, *values(dominance=-0.5))
    assert isinstance(child, Image.Image) and child.size == (1024, 1024)
    assert "cpu" in status


def test_live_updates_skip_repeats_and_respect_the_toggle(app, parent_paths):
    args = (*parent_paths, False, *values())
    child, _, request = app.generate_live(True, None, *args)
    assert isinstance(child, Image.Image)
    for live, last in ((True, request), (False, None)):
        child, status, kept = app.generate_live(live, last, *args)
        assert child == gr.update() and status == gr.update() and kept == last


def test_faces_are_centre_cropped_without_an_aligner(app, tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (300, 200)).save(path)
    face, note = app.face(str(path), "father", align=True)
    assert face.size == (200, 200) and note == ""


def test_face_note_when_no_face_is_found(engine, tmp_path):
    path = tmp_path / "blank.png"
    Image.new("RGB", (300, 200)).save(path)
    face, note = ChildNetApp(engine, aligner=lambda image: None).face(str(path), "mother", align=True)
    assert face.size == (200, 200) and "mother" in note


def test_siblings_use_sampling_even_at_zero_variability(app, parent_paths):
    gallery, status = app.siblings(*parent_paths, False, 3, *values(seed=5))
    assert [caption for _, caption in gallery] == ["seed 5", "seed 6", "seed 7"]
    assert len({image.tobytes() for image, _ in gallery}) == 3
    assert "sampling rate" in status


def test_sweeps(app, parent_paths):
    gallery, _ = app.sweep(*parent_paths, False, "Dominant parent", 5, *values())
    assert [caption for _, caption in gallery] == ["-1.00", "-0.50", "+0.00", "+0.50", "+1.00"]
    gallery, _ = app.sweep(*parent_paths, False, "Age group", 5, *values(age_gender=False, gender="Boy"))
    assert [caption for _, caption in gallery] == [f"{group} years" for group in AGE_GROUPS]


def test_model_view(app, parent_paths):
    gallery, _ = app.model_view(*parent_paths, False)
    assert [caption for _, caption in gallery] == [
        "Father: model input", "Father: reconstruction", "Mother: model input", "Mother: reconstruction"
    ]
    assert gallery[0][0].size == (256, 256)


def test_build(app):
    assert isinstance(app.build(), gr.Blocks)


def test_find_childnet_dir(childnet_dir, tmp_path):
    assert find_childnet_dir(str(childnet_dir)) == childnet_dir
    with pytest.raises(SystemExit, match="download.sh"):
        find_childnet_dir(str(tmp_path))
