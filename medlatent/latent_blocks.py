"""Utilities for compact latent block construction and aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LatentBlock:
    """One hospital's transmitted compact latent object."""

    hospital_id: int
    family: str
    hidden_states: torch.Tensor
    kv_cache: object | None = None


def concatenate_latent_tokens(blocks: list[torch.Tensor]) -> torch.Tensor:
    if not blocks:
        raise ValueError("At least one latent block is required")
    return torch.cat(blocks, dim=1)


def slice_last_positions_from_legacy_cache(past_key_values, num_positions: int):
    """Keep only the last ``num_positions`` from a tuple-style HF KV cache."""
    sliced = []
    for layer in past_key_values:
        if len(layer) < 2:
            raise ValueError("Each cache layer must contain key and value tensors")
        key, value, *rest = layer
        sliced.append((key[..., -num_positions:, :], value[..., -num_positions:, :], *rest))
    return tuple(sliced)
