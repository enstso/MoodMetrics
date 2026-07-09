import argparse
import json
from pathlib import Path
from time import perf_counter

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from moodmetrics.config import Config
from moodmetrics.modeling import ModelSpec, iter_model_specs
from scripts.evaluate_model import build_evaluation_pool, stratify_if_possible


DEFAULT_OPTIMIZATION_PATH = "reports/optimization_results.json"
LABEL_COLUMNS = ("positive", "negative")


def benchmark_classifier(
    texts: list[str],
    labels: list[bool],
    spec: ModelSpec,
    test_size: float,
    random_state: int,
) -> dict[str, object]:
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_if_possible(labels),
    )

    train_started_at = perf_counter()
    model = spec.build().fit(train_texts, train_labels)
    fit_time_seconds = perf_counter() - train_started_at

    predict_started_at = perf_counter()
    predictions = model.predict(test_texts)
    predict_time_seconds = perf_counter() - predict_started_at

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "fit_time_seconds": round(fit_time_seconds, 4),
        "predict_time_seconds": round(predict_time_seconds, 4),
    }


def aggregate_label_results(labels: dict[str, dict[str, object]]) -> dict[str, float]:
    return {
        "accuracy": round(
            sum(float(labels[label]["accuracy"]) for label in LABEL_COLUMNS)
            / len(LABEL_COLUMNS),
            4,
        ),
        "f1_score": round(
            sum(float(labels[label]["f1_score"]) for label in LABEL_COLUMNS)
            / len(LABEL_COLUMNS),
            4,
        ),
        "fit_time_seconds": round(
            sum(float(labels[label]["fit_time_seconds"]) for label in LABEL_COLUMNS),
            4,
        ),
        "predict_time_seconds": round(
            sum(float(labels[label]["predict_time_seconds"]) for label in LABEL_COLUMNS),
            4,
        ),
    }


def benchmark_model_specs(
    texts: list[str],
    positives: list[bool],
    negatives: list[bool],
    candidate_names: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[dict[str, object]]:
    columns = {"positive": positives, "negative": negatives}
    attempts = []
    for spec in iter_model_specs(candidate_names):
        labels = {
            label: benchmark_classifier(
                texts,
                columns[label],
                spec,
                test_size=test_size,
                random_state=random_state,
            )
            for label in LABEL_COLUMNS
        }
        attempts.append(
            {
                "model": spec.to_metadata(),
                "aggregate": aggregate_label_results(labels),
                "labels": labels,
            }
        )

    return sorted(
        attempts,
        key=lambda attempt: (
            float(attempt["aggregate"]["f1_score"]),
            float(attempt["aggregate"]["accuracy"]),
            -float(attempt["aggregate"]["fit_time_seconds"]),
        ),
        reverse=True,
    )


def optimize_from_database(
    database_url: str,
    output_path: str = DEFAULT_OPTIMIZATION_PATH,
    supplementary_path: str | None = None,
    candidate_names: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    texts, positives, negatives, stats = build_evaluation_pool(
        database_url,
        supplementary_path,
    )
    if len(texts) < 8:
        raise ValueError("Au moins huit tweets annotés sont nécessaires pour optimiser.")

    attempts = benchmark_model_specs(
        texts,
        positives,
        negatives,
        candidate_names=candidate_names,
        test_size=test_size,
        random_state=random_state,
    )
    optimization = {
        "samples": len(texts),
        "test_size": test_size,
        "random_state": random_state,
        "dataset": {
            "database_rows": stats["database_rows"],
            "database_unique": stats["database_unique"],
            "duplicates_removed": stats["duplicates_removed"],
            "supplement_added": stats["supplement_added"],
        },
        "best_model": attempts[0]["model"],
        "attempts": attempts,
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(optimization, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return optimization


def print_summary(optimization: dict[str, object]) -> None:
    print(f"Optimisation sur {optimization['samples']} tweets uniques.")
    for rank, attempt in enumerate(optimization["attempts"], start=1):
        aggregate = attempt["aggregate"]
        model = attempt["model"]
        print(
            f"{rank}. {model['name']} - f1={aggregate['f1_score']} "
            f"accuracy={aggregate['accuracy']} fit={aggregate['fit_time_seconds']}s"
        )
    print(f"Meilleur modèle: {optimization['best_model']['name']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare plusieurs pipelines de sentiment analysis."
    )
    parser.add_argument("--output", default=DEFAULT_OPTIMIZATION_PATH)
    parser.add_argument("--candidate", action="append", dest="candidates")
    args = parser.parse_args()

    result = optimize_from_database(
        Config.DATABASE_URL,
        output_path=args.output,
        supplementary_path=Config.SUPPLEMENTARY_DATASET_PATH,
        candidate_names=args.candidates,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
