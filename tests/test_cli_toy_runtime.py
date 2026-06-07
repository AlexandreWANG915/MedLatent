import json
import subprocess
import sys
from pathlib import Path


def test_toy_training_and_diagnosis_eval_run_end_to_end(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "run"

    train = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "train_medlatent_h.py"),
            "--toy",
            "--output_dir",
            str(output_dir),
            "--steps",
            "4",
            "--seed",
            "7",
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "toy training complete" in train.stdout
    assert (output_dir / "distiller_final.pt").exists()
    assert (output_dir / "boundary_final.pt").exists()
    assert (output_dir / "toy_head.pt").exists()

    eval_out = output_dir / "diagnosis_metrics.json"
    evaluate = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "evaluate_diagnosis.py"),
            "--toy",
            "--checkpoint_dir",
            str(output_dir),
            "--output",
            str(eval_out),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "toy diagnosis evaluation complete" in evaluate.stdout
    metrics = json.loads(eval_out.read_text())
    assert metrics["num_examples"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
