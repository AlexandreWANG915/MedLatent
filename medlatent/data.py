"""Small data structures for cross-hospital rare-disease diagnosis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    hpo_codes: list[str]
    phenotype_text: str
    disease_name: str
    omim_id: str | None = None


def load_case_records(path: str | Path) -> list[CaseRecord]:
    rows = json.loads(Path(path).read_text())
    records: list[CaseRecord] = []
    for row in rows:
        records.append(
            CaseRecord(
                case_id=str(row.get("case_id", row.get("id"))),
                hpo_codes=list(row.get("hpo_codes", [])),
                phenotype_text=str(row.get("phenotypes_str", row.get("phenotype_text", ""))),
                disease_name=str(row.get("diseases_prompt", row.get("diseases_str", row.get("disease_name", "")))),
                omim_id=row.get("omim_id") or row.get("disease_code"),
            )
        )
    return records


def records_as_dicts(records: Sequence[CaseRecord]) -> list[dict]:
    return [
        {
            "case_id": record.case_id,
            "hpo_codes": record.hpo_codes,
            "phenotype_text": record.phenotype_text,
            "disease": record.disease_name,
            "omim_id": record.omim_id,
        }
        for record in records
    ]
