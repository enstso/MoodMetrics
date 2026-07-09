import json

from scripts.generate_report import generate_pdf_report


def test_generate_pdf_report_writes_pdf(tmp_path):
    evaluation_path = tmp_path / "evaluation.json"
    output_path = tmp_path / "evaluation_report.pdf"
    evaluation_path.write_text(
        json.dumps(
            {
                "samples": 100,
                "labels": {
                    "positive": {
                        "train_samples": 75,
                        "test_samples": 25,
                        "accuracy": 0.92,
                        "precision": 0.9,
                        "recall": 0.88,
                        "f1_score": 0.89,
                        "confusion_matrix": {
                            "labels": ["absent", "present"],
                            "values": [[14, 1], [1, 9]],
                        },
                    },
                    "negative": {
                        "train_samples": 75,
                        "test_samples": 25,
                        "accuracy": 0.88,
                        "precision": 0.86,
                        "recall": 0.84,
                        "f1_score": 0.85,
                        "confusion_matrix": {
                            "labels": ["absent", "present"],
                            "values": [[13, 2], [1, 9]],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    generated_path = generate_pdf_report(str(evaluation_path), str(output_path))

    assert generated_path == output_path
    assert output_path.read_bytes().startswith(b"%PDF")
