import pickle
import sys
from dataclasses import replace

import pytest
import torch
from PIL import Image

from ..engine import AGE_GROUPS, ChildNetEngine, Controls, _childnet_checkpoints, encoder_input, to_pil


def latent(engine, parents, **controls):
    w_father, w_mother = (engine.encode(p) for p in parents)
    return engine.child_latent(w_father, w_mother, Controls(**controls))


@pytest.mark.parametrize("move2parent", [None, -1.0, -0.35, 0.5, 1.0])
@pytest.mark.parametrize("age_gender", [None, (0, 1), (3, 0), (len(AGE_GROUPS) - 1, 1)])
def test_matches_childnet(engine, parents, move2parent, age_gender):
    age, gender = age_gender or (None, None)
    model = engine.model
    with torch.no_grad():  # ChildNet's own pipeline, as its main.py runs it
        expected, _ = model.kinship_model.encoder_forward(*(encoder_input(p) for p in parents), move2parent=move2parent)
        if age_gender:
            expected, _ = model.disentangle(expected, torch.tensor([age]), torch.tensor([gender]))
    assert torch.equal(latent(engine, parents, dominance=move2parent or 0.0, age=age, gender=gender), expected)


def test_generate_matches_childnet_forward(engine, parents):
    with torch.no_grad():
        expected = engine.model(*(encoder_input(p) for p in parents), torch.tensor([2]), torch.tensor([1]), -0.6)
    [child] = engine.generate(*parents, Controls(dominance=-0.6, age=2, gender=1))
    assert child.size == (1024, 1024)
    assert child.tobytes() == to_pil(expected).tobytes()


def test_sampling_matches_childnet_sample_mode(engine, parents):
    kinship = engine.model.kinship_model
    kinship.eval_mode_with_sampling()  # what ChildNet(sample=True) does
    try:
        torch.manual_seed(123)
        with torch.no_grad():
            expected, _ = kinship.encoder_forward(*(encoder_input(p) for p in parents))
    finally:
        kinship.eval()
    assert torch.equal(latent(engine, parents, variability=0.5, seed=123), expected)


def test_variability_and_seeds(engine, parents):
    dropouts = [m for m in engine.model.modules() if isinstance(m, torch.nn.Dropout)]
    before = [(m.p, m.training) for m in dropouts]
    assert torch.equal(latent(engine, parents, seed=1), latent(engine, parents, seed=2))
    assert torch.equal(latent(engine, parents, variability=0.3, seed=1), latent(engine, parents, variability=0.3, seed=1))
    assert not torch.equal(latent(engine, parents, variability=0.3, seed=1), latent(engine, parents, variability=0.3, seed=2))
    assert [(m.p, m.training) for m in dropouts] == before


def test_samples_use_consecutive_seeds(engine, parents):
    controls = Controls(variability=0.5, seed=10)
    samples = engine.generate(*parents, controls, num_samples=3)
    [third] = engine.generate(*parents, replace(controls, seed=12))
    assert samples[2].tobytes() == third.tobytes()
    assert samples[0].tobytes() != samples[1].tobytes()


def test_uniform_level_dominance_equals_global(engine, parents):
    assert torch.equal(latent(engine, parents, dominance=0.4), latent(engine, parents, level_dominance=(0.4, 0.4, 0.4)))


@pytest.mark.parametrize("level, layers", [(0, slice(0, 4)), (1, slice(4, 8)), (2, slice(8, 18))])
def test_level_dominance_only_changes_its_layers(engine, parents, level, layers):
    base = latent(engine, parents)
    values = [0.0, 0.0, 0.0]
    values[level] = 0.8
    changed = latent(engine, parents, level_dominance=tuple(values))
    unchanged = torch.ones(18, dtype=torch.bool)
    unchanged[layers] = False
    assert torch.equal(changed[:, unchanged], base[:, unchanged])
    assert not torch.equal(changed[:, layers], base[:, layers])


def test_edit_strength_scales_the_edit(engine, parents):
    plain = latent(engine, parents)
    full = latent(engine, parents, age=4, gender=0)
    assert torch.equal(latent(engine, parents, age=4, gender=0, edit_strength=0.0), plain)
    doubled = latent(engine, parents, age=4, gender=0, edit_strength=2.0)
    torch.testing.assert_close(doubled - plain, 2 * (full - plain))


@pytest.mark.parametrize(
    "controls",
    [dict(dominance=1.5), dict(level_dominance=(0.1, 0.2)), dict(age=10, gender=0), dict(age=1, gender=2), dict(variability=0.99)],
)
def test_rejects_out_of_range_controls(engine, parents, controls):
    with pytest.raises(ValueError):
        latent(engine, parents, **controls)


def test_encode_is_cached_by_content(engine, parents):
    assert engine.encode(parents[0]) is engine.encode(parents[0].copy())


def test_encoder_input_centre_crops():
    image = Image.new("RGB", (300, 200), "red")
    image.paste("blue", (0, 0, 50, 200))
    image.paste("blue", (250, 0, 300, 200))
    x = encoder_input(image)
    assert x.shape == (1, 3, 256, 256)
    assert torch.equal(x[0, 0], torch.ones(256, 256)) and not x[0, 1:].any()


def test_every_tensor_comes_from_a_checkpoint(engine, childnet_dir):
    checkpoints = childnet_dir / "checkpoints"
    expected = {f"kinship_model.{k}": v for k, v in torch.load(checkpoints / "childnet_nokdb.pt").items()}
    expected |= {f"disentangle.{k}": v for k, v in torch.load(checkpoints / "disentanglement.pt").items()}
    state = engine.model.state_dict()
    assert state.keys() == expected.keys()
    assert all(torch.equal(state[k], expected[k]) for k in state)


def test_switching_weights(engine, parents):
    nokdb = engine.generate(*parents)[0]
    engine.load("fiw")
    try:
        assert engine.weights == "fiw"
        assert engine.generate(*parents)[0].tobytes() != nokdb.tobytes()
    finally:
        engine.load("nokdb")
    assert engine.generate(*parents)[0].tobytes() == nokdb.tobytes()


def test_missing_checkpoints_are_reported(engine):
    path = engine.childnet_dir / "checkpoints" / "childnet_fiw.pt"
    path.rename(path.with_suffix(".bak"))
    try:
        assert engine.missing_checkpoints("fiw") == ["childnet_fiw.pt"]
        assert engine.available_weights() == ["nokdb"]
        with pytest.raises(FileNotFoundError, match="download.sh"):
            engine.load("fiw")
        assert engine.weights == "nokdb"
    finally:
        path.with_suffix(".bak").rename(path)


def test_rejects_a_directory_without_childnet(tmp_path):
    with pytest.raises(FileNotFoundError, match="ChildNet checkout"):
        ChildNetEngine(tmp_path)


def test_e4e_cuda_ops_are_stubbed(engine):
    op = sys.modules["models.e4e.models.stylegan2.op"]
    with pytest.raises(RuntimeError, match="stubbed"):
        op.upfirdn2d(torch.zeros(1))


def test_checkpoint_loader_allows_pickled_objects_only_inside_childnet(engine, childnet_dir, tmp_path):
    from models.stylegan.dnnlib import EasyDict  # ChildNet's own class, as in G_kwargs-style pickles

    inside = childnet_dir / "checkpoints" / "easydict_test.pt"
    outside = tmp_path / "easydict_test.pt"
    for path in (inside, outside):
        torch.save(EasyDict(a=1), path)
    try:
        with _childnet_checkpoints(childnet_dir):
            assert torch.load("checkpoints/easydict_test.pt") == {"a": 1}
            if tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 6):  # weights_only by default
                with pytest.raises(pickle.UnpicklingError):
                    torch.load(outside)
    finally:
        inside.unlink()
