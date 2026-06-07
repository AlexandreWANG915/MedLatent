"""Small CPU runtime used to verify MedLatent CLIs end-to-end.

This is not a scientific experiment. It exercises the same trainable modules
as the paper method without requiring private data or downloaded LLM weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import exact_match
from .modules import BoundaryEmbeddings, LatentDistiller


class ToyDiagnosisHead(nn.Module):
    module_type = "ToyDiagnosisHead"

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_labels = int(num_labels)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, wrapped_latents: torch.Tensor) -> torch.Tensor:
        pooled = wrapped_latents.mean(dim=1)
        return self.classifier(pooled)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "module_type": self.module_type,
                "hidden_size": self.hidden_size,
                "num_labels": self.num_labels,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "ToyDiagnosisHead":
        ckpt = torch.load(path, map_location=map_location)
        if ckpt.get("module_type") != cls.module_type:
            raise ValueError(f"Expected {cls.module_type}, got {ckpt.get('module_type')!r}")
        module = cls(ckpt["hidden_size"], ckpt["num_labels"])
        module.load_state_dict(ckpt["state_dict"])
        return module


def _make_toy_batch(*, seed: int, batch_size: int = 24, hidden_size: int = 16, num_labels: int = 3):
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(num_labels, hidden_size, generator=generator)
    labels = torch.arange(batch_size) % num_labels
    hidden = prototypes[labels] + 0.05 * torch.randn(batch_size, hidden_size, generator=generator)
    names = [f"Disease {chr(ord('A') + int(label))}" for label in labels]
    return hidden, labels, names


def train_toy_medlatent_h(output_dir: str | Path, *, steps: int = 16, seed: int = 42) -> dict[str, float]:
    torch.manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    hidden_size = 16
    num_labels = 3
    distiller = LatentDistiller(hidden_size)
    boundary = BoundaryEmbeddings(hidden_size)
    head = ToyDiagnosisHead(hidden_size, num_labels)
    optimizer = torch.optim.AdamW(
        list(distiller.parameters()) + list(boundary.parameters()) + list(head.parameters()),
        lr=3e-3,
        weight_decay=0.01,
    )

    last_loss = 0.0
    for step in range(max(1, int(steps))):
        hidden, labels, _ = _make_toy_batch(seed=seed + step, hidden_size=hidden_size, num_labels=num_labels)
        latents = torch.stack([distiller(hidden) for _ in range(3)], dim=1)
        logits = head(boundary.wrap(latents))
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())

    distiller.save(output / "distiller_final.pt")
    boundary.save(output / "boundary_final.pt")
    head.save(output / "toy_head.pt")
    metadata = {"runtime": "toy", "steps": int(steps), "seed": int(seed), "final_loss": last_loss}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def evaluate_toy_diagnosis(checkpoint_dir: str | Path, output: str | Path | None = None, *, seed: int = 123) -> dict:
    ckpt = Path(checkpoint_dir)
    distiller = LatentDistiller.load(ckpt / "distiller_final.pt")
    boundary = BoundaryEmbeddings.load(ckpt / "boundary_final.pt")
    head = ToyDiagnosisHead.load(ckpt / "toy_head.pt")

    hidden, labels, names = _make_toy_batch(seed=seed, hidden_size=distiller.hidden_size)
    with torch.no_grad():
        latents = torch.stack([distiller(hidden) for _ in range(3)], dim=1)
        pred_ids = head(boundary.wrap(latents)).argmax(dim=-1)
    predictions = [f"Disease {chr(ord('A') + int(idx))}" for idx in pred_ids]
    correct = [exact_match(pred, gold) for pred, gold in zip(predictions, names)]
    metrics = {
        "runtime": "toy",
        "num_examples": len(names),
        "accuracy": sum(correct) / len(correct),
        "predictions": predictions[:5],
        "references": names[:5],
    }
    if output is not None:
        Path(output).write_text(json.dumps(metrics, indent=2))
    return metrics
