"""Passive-observer reconstruction helpers."""

from __future__ import annotations

from .metrics import token_f1


def reconstruction_report(reconstruction: str, source_prompt: str) -> dict[str, float]:
    return {"token_f1": token_f1(reconstruction, source_prompt)}
