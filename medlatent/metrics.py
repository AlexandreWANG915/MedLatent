"""Lightweight metrics used by MedLatent evaluation scripts."""

from __future__ import annotations

from collections import Counter


def token_f1(prediction: str, reference: str) -> float:
    pred = prediction.lower().split()
    ref = reference.lower().split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> bool:
    return prediction.strip().casefold() == reference.strip().casefold()
