"""Fixtures that run the real ChildNet code with small random-weight checkpoints.

Set CHILDNET_DIR to a ChildNet checkout to run the tests that need it (they
copy its code to a temp dir and need ~4 GB of free disk there).
"""

import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from ..engine import ChildNetEngine
from .fake_checkpoints import write_fake_checkpoints


@pytest.fixture(scope="session")
def childnet_dir(tmp_path_factory) -> Path:
    source = Path(os.environ.get("CHILDNET_DIR", "")).expanduser()
    if not (source / "models" / "childnet.py").is_file():
        pytest.skip("set CHILDNET_DIR to a ChildNet checkout to run this test")
    copy = tmp_path_factory.mktemp("childnet")
    for name in ("models", "imgs"):
        shutil.copytree(source / name, copy / name)
    write_fake_checkpoints(copy, small=True)
    return copy


@pytest.fixture(scope="session")
def engine(childnet_dir) -> ChildNetEngine:
    engine = ChildNetEngine(childnet_dir, device="cpu")
    engine.load("nokdb")
    return engine


@pytest.fixture(scope="session")
def parent_paths(childnet_dir) -> tuple[str, str]:
    return str(childnet_dir / "imgs" / "father.jpg"), str(childnet_dir / "imgs" / "mother.jpg")


@pytest.fixture(scope="session")
def parents(parent_paths) -> tuple[Image.Image, Image.Image]:
    return tuple(Image.open(path).convert("RGB") for path in parent_paths)
