import json

from scripts.evaluate_model import render_confusion_matrix_png
from scripts.generate_report import (
    generate_pdf_report,
    interpret_confusion,
    per_class_from_confusion,
)


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


def test_generate_pdf_report_embeds_figures(tmp_path):
    render_confusion_matrix_png([[14, 1], [1, 9]], "positif", tmp_path / "cm_pos.png")
    render_confusion_matrix_png([[13, 2], [1, 9]], "négatif", tmp_path / "cm_neg.png")

    def label(figure):
        return {
            "train_samples": 75,
            "test_samples": 25,
            "accuracy": 0.92,
            "precision": 0.9,
            "recall": 0.88,
            "f1_score": 0.89,
            "per_class": per_class_from_confusion([[14, 1], [1, 9]]),
            "confusion_matrix": {"labels": ["absent", "present"], "values": [[14, 1], [1, 9]]},
            "figure": figure,
        }

    evaluation_path = tmp_path / "evaluation.json"
    output_path = tmp_path / "report.pdf"
    evaluation_path.write_text(
        json.dumps(
            {
                "samples": 200,
                "test_size": 0.2,
                "random_state": 42,
                "dataset": {
                    "samples": 200,
                    "positive": 80,
                    "negative": 90,
                    "neutral": 30,
                    "note": "Corpus de test.",
                },
                "labels": {"positive": label("cm_pos.png"), "negative": label("cm_neg.png")},
            }
        ),
        encoding="utf-8",
    )

    generate_pdf_report(str(evaluation_path), str(output_path))

    # Un PDF avec deux images intégrées est nettement plus volumineux qu'un PDF texte.
    assert output_path.read_bytes().startswith(b"%PDF")
    assert output_path.stat().st_size > 20_000


def test_interpret_confusion_uses_real_counts_and_agreement():
    metrics = {"confusion_matrix": {"labels": ["absent", "present"], "values": [[20, 1], [3, 8]]}}
    text = interpret_confusion("positive", metrics)

    assert "8 vrais positifs" in text
    assert "1 faux positif (FP)" in text  # accord au singulier
    assert "3 faux négatifs" in text
    assert "sur-prédire" not in text  # 3 FN > 1 FP => modèle prudent, pas de sur-prédiction
