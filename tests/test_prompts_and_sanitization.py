from pathlib import Path

from medlatent.prompts import build_hospital_prompt, build_host_question


def test_medlatent_prompts_match_single_answer_contract():
    hospital = build_hospital_prompt(
        hospital_id=2,
        case_disease="Marfan syndrome",
        case_phenotype="Tall stature, Aortic aneurysm",
        query_phenotype="Aortic dissection, Lens subluxation",
    )
    host = build_host_question("Aortic dissection, Lens subluxation")

    assert "Hospital 2" in hospital
    assert "Marfan syndrome" in hospital
    assert "Patient's phenotype: Aortic dissection, Lens subluxation" in host
    assert "<answer>" in host
    assert "Only the disease name" in host


def test_artifact_contains_no_personal_absolute_paths():
    root = Path(__file__).resolve().parents[1]
    sensitive_terms = ["/" + "data" + "/" + "ios" + "4132", "ios" + "4132"]
    checked = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".md", ".yaml", ".toml", ".txt"}:
            text = path.read_text(errors="ignore")
            checked.append(path)
            for term in sensitive_terms:
                assert term not in text

    assert checked
