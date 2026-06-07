"""Trainable latent-interface modules used by MedLatent.

The LLM backbones are frozen. MedLatent-H trains only a same-family
``LatentDistiller`` plus boundary embeddings. MedLatent-X trains only
cross-family ``LatentProjector`` modules plus host-side boundary embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


class LatentDistiller(nn.Module):
    """Same-backbone hidden-to-embedding distiller.

    The distiller maps a hidden state back into the frozen backbone's input
    embedding space. Repeated autoregressive use yields ``num_latents`` compact
    latent positions whose KV entries are transmitted to the host.
    """

    module_type = "LatentDistiller"

    def __init__(self, hidden_size: int, *, identity_init: bool = True):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.hidden_size, eps=1e-6),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
        )
        if identity_init:
            with torch.no_grad():
                self.projection[1].weight.copy_(torch.eye(self.hidden_size))

        self.latent_begin = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.latent_begin, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden)

    def begin_embedding(self, *, dtype=None, device=None) -> torch.Tensor:
        emb = self.latent_begin
        if dtype is not None:
            emb = emb.to(dtype=dtype)
        if device is not None:
            emb = emb.to(device=device)
        return emb.view(1, 1, -1)

    def state(self) -> dict[str, Any]:
        return {
            "module_type": self.module_type,
            "hidden_size": self.hidden_size,
            "projection_state_dict": self.projection.state_dict(),
            "latent_begin": self.latent_begin.detach().cpu(),
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "LatentDistiller":
        ckpt: Mapping[str, Any] = torch.load(path, map_location=map_location)
        if ckpt.get("module_type") != cls.module_type:
            raise ValueError(f"Expected {cls.module_type} checkpoint, got {ckpt.get('module_type')!r}")
        module = cls(int(ckpt["hidden_size"]))
        module.projection.load_state_dict(ckpt["projection_state_dict"])
        module.latent_begin.data.copy_(ckpt["latent_begin"].to(module.latent_begin.dtype))
        return module

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LatentProjector(nn.Module):
    """Cross-family projector from encoder-family latents to host embeddings."""

    module_type = "LatentProjector"

    def __init__(self, encoder_dim: int, host_dim: int):
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.host_dim = int(host_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.encoder_dim, eps=1e-6),
            nn.Linear(self.encoder_dim, self.host_dim, bias=False),
        )

    def forward(self, latent_hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(latent_hidden)

    def state(self) -> dict[str, Any]:
        return {
            "module_type": self.module_type,
            "encoder_dim": self.encoder_dim,
            "host_dim": self.host_dim,
            "projection_state_dict": self.projection.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "LatentProjector":
        ckpt: Mapping[str, Any] = torch.load(path, map_location=map_location)
        if ckpt.get("module_type") != cls.module_type:
            raise ValueError(f"Expected {cls.module_type} checkpoint, got {ckpt.get('module_type')!r}")
        module = cls(encoder_dim=int(ckpt["encoder_dim"]), host_dim=int(ckpt["host_dim"]))
        module.projection.load_state_dict(ckpt["projection_state_dict"])
        return module

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BoundaryEmbeddings(nn.Module):
    """Learned begin/end markers delimiting one hospital latent block."""

    module_type = "BoundaryEmbeddings"

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.begin = nn.Parameter(torch.zeros(self.hidden_size))
        self.end = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.begin, mean=0.0, std=0.02)
        nn.init.normal_(self.end, mean=0.0, std=0.02)

    def wrap(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        if latent_tokens.ndim != 3:
            raise ValueError(f"latent_tokens must be [batch, num_latents, hidden], got {tuple(latent_tokens.shape)}")
        if latent_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {latent_tokens.shape[-1]}")
        batch = latent_tokens.shape[0]
        begin = self.begin.to(dtype=latent_tokens.dtype, device=latent_tokens.device).view(1, 1, -1)
        end = self.end.to(dtype=latent_tokens.dtype, device=latent_tokens.device).view(1, 1, -1)
        return torch.cat([begin.expand(batch, -1, -1), latent_tokens, end.expand(batch, -1, -1)], dim=1)

    def state(self) -> dict[str, Any]:
        return {
            "module_type": self.module_type,
            "hidden_size": self.hidden_size,
            "begin": self.begin.detach().cpu(),
            "end": self.end.detach().cpu(),
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state(), path)

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "BoundaryEmbeddings":
        ckpt: Mapping[str, Any] = torch.load(path, map_location=map_location)
        if ckpt.get("module_type") != cls.module_type:
            raise ValueError(f"Expected {cls.module_type} checkpoint, got {ckpt.get('module_type')!r}")
        module = cls(int(ckpt["hidden_size"]))
        module.begin.data.copy_(ckpt["begin"].to(module.begin.dtype))
        module.end.data.copy_(ckpt["end"].to(module.end.dtype))
        return module
