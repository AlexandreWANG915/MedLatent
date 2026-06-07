"""Forward-pass entry points for MedLatent-H and MedLatent-X.

The CLI training scripts own model loading and batching. This module keeps the
paper-level interfaces explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .modules import BoundaryEmbeddings, LatentProjector


@dataclass
class MedLatentHBatch:
    hospital_input_ids: Mapping[int, torch.Tensor]
    hospital_attention_masks: Mapping[int, torch.Tensor]
    host_question_ids: torch.Tensor
    target_ids: torch.Tensor


@dataclass
class MedLatentXBatch:
    native_blocks: Mapping[int, torch.Tensor]
    foreign_blocks: Mapping[int, tuple[str, torch.Tensor]]
    host_question_ids: torch.Tensor
    target_ids: torch.Tensor


def project_foreign_blocks(
    foreign_blocks: Mapping[int, tuple[str, torch.Tensor]],
    projectors: Mapping[str, LatentProjector],
    boundary: BoundaryEmbeddings,
) -> dict[int, torch.Tensor]:
    projected: dict[int, torch.Tensor] = {}
    for hospital_id, (family, hidden_states) in foreign_blocks.items():
        if family not in projectors:
            raise KeyError(f"No projector registered for encoder family {family!r}")
        projected[hospital_id] = boundary.wrap(projectors[family](hidden_states))
    return projected
