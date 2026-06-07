import json
from pathlib import Path

import pytest
import torch

from medlatent.cache import FamilyLatentCache
from medlatent.retrieval import HpoCosineRetriever


def test_hpo_cosine_retriever_ranks_by_normalized_similarity():
    term_embeddings = {
        "HP:1": [1.0, 0.0],
        "HP:2": [0.0, 1.0],
        "HP:3": [1.0, 1.0],
    }
    ic = {"HP:1": 1.0, "HP:2": 1.0, "HP:3": 1.0}
    records = [
        {"case_id": "a", "hpo_codes": ["HP:1"], "disease": "A"},
        {"case_id": "b", "hpo_codes": ["HP:2"], "disease": "B"},
        {"case_id": "c", "hpo_codes": ["HP:3"], "disease": "C"},
    ]

    retriever = HpoCosineRetriever(term_embeddings, ic)
    ranked = retriever.rank(["HP:1"], records, top_k=3)

    assert [item.record["case_id"] for item in ranked] == ["a", "c", "b"]
    assert ranked[0].score == pytest.approx(1.0)


def test_family_latent_cache_validates_manifest_and_lookup(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    tensor = torch.randn(4, 6, dtype=torch.bfloat16)
    torch.save(
        {
            "split": "train",
            "dtype": "bfloat16",
            "num_latents": 4,
            "encoder_dim": 6,
            "entries": {"case-1#2": tensor},
        },
        cache_dir / "train.pt",
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "module_type": "MedLatentLatentCache",
                "schema_version": 1,
                "encoder_family": "llama32_3b",
                "distiller_checkpoint": "${MEDLATENT_ARTIFACTS}/llama/distiller_final.pt",
                "num_latents": 4,
                "encoder_dim": 6,
                "dtype": "bfloat16",
                "retrieval": "cosine_hpo",
                "splits": {"train": {"path": "train.pt", "entries": 1}},
            }
        )
    )

    cache = FamilyLatentCache(cache_dir)
    cache.validate({"encoder_family": "llama32_3b", "num_latents": 4, "retrieval": "cosine_hpo"})
    got = cache.lookup("train", "case-1", 2)

    assert got.shape == (4, 6)
    assert got.dtype == torch.bfloat16


def test_family_latent_cache_rejects_wrong_manifest_type(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({"module_type": "LegacyLatentCache"}))

    with pytest.raises(ValueError, match="MedLatentLatentCache"):
        FamilyLatentCache(tmp_path)
