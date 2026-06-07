"""On-disk latent cache for MedLatent-X cross-family alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch


class FamilyLatentCache:
    """Lazy loader for encoder-family compact latent hidden states.

    Cache keys use ``"{case_id}#{hospital_id}"`` and values are bf16 tensors
    shaped ``[num_latents, encoder_dim]``.
    """

    module_type = "MedLatentLatentCache"

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("module_type") != self.module_type:
            raise ValueError(f"Expected {self.module_type} manifest, got {self.manifest.get('module_type')!r}")
        self._shards: dict[str, Mapping[str, torch.Tensor]] = {}

    def validate(self, expected: Mapping[str, object]) -> None:
        for key, value in expected.items():
            actual = self.manifest.get(key)
            if actual != value:
                raise ValueError(f"Manifest mismatch for {key}: expected {value!r}, got {actual!r}")

    def has_split(self, split: str) -> bool:
        return split in self.manifest.get("splits", {})

    def _load_split(self, split: str) -> Mapping[str, torch.Tensor]:
        if split in self._shards:
            return self._shards[split]
        split_info = self.manifest.get("splits", {}).get(split)
        if split_info is None:
            raise KeyError(f"Cache split {split!r} is not present")
        shard_path = self.cache_dir / split_info["path"]
        shard = torch.load(shard_path, map_location="cpu")
        self._validate_shard(split, shard)
        entries = shard["entries"]
        self._shards[split] = entries
        return entries

    def _validate_shard(self, split: str, shard: Mapping[str, object]) -> None:
        if shard.get("split") != split:
            raise ValueError(f"Shard split mismatch: expected {split!r}, got {shard.get('split')!r}")
        for key in ("dtype", "num_latents", "encoder_dim"):
            if shard.get(key) != self.manifest.get(key):
                raise ValueError(f"Shard {key} mismatch: expected {self.manifest.get(key)!r}, got {shard.get(key)!r}")
        entries = shard.get("entries")
        if not isinstance(entries, dict):
            raise ValueError("Shard entries must be a dictionary")

    def lookup(self, split: str, case_id: str, hospital_id: int) -> torch.Tensor:
        entries = self._load_split(split)
        key = f"{case_id}#{hospital_id}"
        if key not in entries:
            raise KeyError(f"Missing latent cache entry {key!r} in split {split!r}")
        value = entries[key]
        expected_shape = (int(self.manifest["num_latents"]), int(self.manifest["encoder_dim"]))
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Entry {key!r} has shape {tuple(value.shape)}, expected {expected_shape}")
        return value

    def lookup_batch(
        self,
        split: str,
        case_ids: list[str],
        hospital_id: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        stacked = torch.stack([self.lookup(split, case_id, hospital_id) for case_id in case_ids], dim=0)
        if device is not None or dtype is not None:
            stacked = stacked.to(device=device, dtype=dtype)
        return stacked
