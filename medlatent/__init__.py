"""MedLatent: compact latent KV communication for cross-hospital diagnosis."""

from .modules import BoundaryEmbeddings, LatentDistiller, LatentProjector

__all__ = [
    "BoundaryEmbeddings",
    "LatentDistiller",
    "LatentProjector",
]
