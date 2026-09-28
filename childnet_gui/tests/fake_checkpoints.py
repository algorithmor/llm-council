"""Random-weight checkpoints in ChildNet's on-disk format.

Lets the GUI and its tests run ChildNet's real code without the pretrained
weights. The images are meaningless, but every tensor flows through the same
modules, shapes and loading paths as with the real checkpoints.

    python -m childnet_gui.tests.fake_checkpoints /path/to/childnet-copy
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from unittest import mock

import torch

from ..engine import SHARED_CHECKPOINTS, WEIGHTS, _import_childnet


def generator_kwargs(small: bool = False) -> dict:
    """Constructor kwargs of the FFHQ 1024px StyleGAN2 (config-f) generator.

    `small` shrinks the synthesis channels, which makes the decoder much faster
    while keeping the 18-layer W+ space the rest of ChildNet depends on.
    """
    channels = dict(channel_base=1024, channel_max=32) if small else dict(channel_base=32768, channel_max=512)
    return dict(
        z_dim=512,
        c_dim=0,
        w_dim=512,
        img_resolution=1024,
        img_channels=3,
        mapping_kwargs=dict(num_layers=8),
        synthesis_kwargs=dict(num_fp16_res=0, conv_clamp=None, **channels),
    )


def write_fake_checkpoints(childnet_dir: str | Path, small: bool = False, seed: int = 0) -> None:
    """Write random checkpoints for every ChildNet variant into `childnet_dir`/checkpoints.

    Refuses to overwrite existing files, so real checkpoints are never clobbered.
    """
    checkpoints = Path(childnet_dir).resolve() / "checkpoints"
    targets = [checkpoints / name for name in (*SHARED_CHECKPOINTS, *(f"childnet_{w}.pt" for w in WEIGHTS))]
    existing = [str(t) for t in targets if t.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing checkpoints: {existing}")
    checkpoints.mkdir(parents=True, exist_ok=True)

    childnet_cls = _import_childnet(checkpoints.parent, cuda_ops=False)
    g_kwargs = generator_kwargs(small)

    def fake_load(f, *args, **kwargs):
        return copy.deepcopy(g_kwargs) if str(f).endswith("G_kwargs.pt") else mock.MagicMock()

    # Build ChildNet with its random initialisation by skipping every checkpoint load.
    torch.manual_seed(seed)
    with mock.patch.object(torch, "load", fake_load), mock.patch.object(torch.nn.Module, "load_state_dict"):
        model = childnet_cls("nokdb")

    torch.save(g_kwargs, checkpoints / "G_kwargs.pt")
    encoder = model.kinship_model.e4e.state_dict()
    torch.save(encoder.pop("latent_avg")[0].clone(), checkpoints / "latent_avg.pt")
    torch.save(encoder, checkpoints / "encoder_state_dict.pt")
    torch.save(model.disentangle.state_dict(), checkpoints / "disentanglement.pt")

    for i, weights in enumerate(WEIGHTS):
        if i:  # give each variant its own kinship module
            with torch.no_grad():
                for p in model.kinship_model.gene_model.parameters():
                    p.add_(0.1 * torch.randn_like(p))
        torch.save(model.kinship_model.state_dict(), checkpoints / f"childnet_{weights}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("childnet_dir", help="a ChildNet checkout without real checkpoints")
    parser.add_argument("--small", action="store_true", help="shrink the decoder for speed")
    args = parser.parse_args()
    write_fake_checkpoints(args.childnet_dir, small=args.small)
    print(f"Wrote fake checkpoints to {Path(args.childnet_dir) / 'checkpoints'}")
