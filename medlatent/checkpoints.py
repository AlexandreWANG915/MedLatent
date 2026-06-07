"""Checkpoint helpers for MedLatent trainable interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


def save_projector_set(path: str | Path, projectors: Mapping[str, torch.nn.Module], metadata: Mapping[str, object]) -> None:
    payload = {
        "module_type": "MedLatentProjectorSet",
        "metadata": dict(metadata),
        "projectors": {name: module.state_dict() for name, module in projectors.items()},
    }
    torch.save(payload, path)


def load_projector_set(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    if payload.get("module_type") != "MedLatentProjectorSet":
        raise ValueError(f"Expected MedLatentProjectorSet, got {payload.get('module_type')!r}")
    return payload
