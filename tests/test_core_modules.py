import tempfile
from pathlib import Path

import torch

from medlatent.modules import BoundaryEmbeddings, LatentDistiller, LatentProjector


def test_latent_distiller_projects_same_hidden_size_and_roundtrips():
    module = LatentDistiller(hidden_size=16)
    x = torch.randn(2, 4, 16)

    y = module(x)

    assert y.shape == (2, 4, 16)
    assert module.num_trainable_params() == 16 * 16 + 2 * 16 + 16

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "distiller.pt"
        module.save(path)
        loaded = LatentDistiller.load(path)
        assert loaded.hidden_size == 16
        for left, right in zip(module.parameters(), loaded.parameters()):
            assert torch.allclose(left, right)


def test_latent_projector_maps_encoder_to_host_space_and_roundtrips():
    module = LatentProjector(encoder_dim=12, host_dim=20)
    x = torch.randn(3, 5, 12)

    y = module(x)

    assert y.shape == (3, 5, 20)
    assert module.num_trainable_params() == 12 * 20 + 2 * 12

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "projector.pt"
        module.save(path)
        loaded = LatentProjector.load(path)
        assert loaded.encoder_dim == 12
        assert loaded.host_dim == 20
        for left, right in zip(module.parameters(), loaded.parameters()):
            assert torch.allclose(left, right)


def test_boundary_embeddings_wrap_latent_tokens():
    boundary = BoundaryEmbeddings(hidden_size=8)
    latents = torch.randn(2, 3, 8)

    wrapped = boundary.wrap(latents)

    assert wrapped.shape == (2, 5, 8)
    assert torch.allclose(wrapped[:, 1:4], latents)
    assert torch.allclose(wrapped[:, 0], boundary.begin.expand(2, -1))
    assert torch.allclose(wrapped[:, -1], boundary.end.expand(2, -1))
