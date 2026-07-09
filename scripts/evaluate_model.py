import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sans affichage, adapté à l'exécution en conteneur

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from moodmetrics.config import Config
from moodmetrics.database import Tweet, create_session_factory
from moodmetrics.datasets import read_tweet_rows
from scripts.train_model import train_classifier


CONFUSION_LABELS = [False, True]
CONFUSION_LABEL_NAMES = ["absent", "present"]

# Nom lisible des deux classifieurs pour les figures et le rapport.
CLASSIFIER_TITLES = {
    "positive": "Sentiment positif",
    "negative": "Sentiment négatif",
}
FIGURE_FILENAMES = {
    "positive": "confusion_matrix_positive.png",
    "negative": "confusion_matrix_negative.png",
}


def load_training_rows(database_url: str) -> list[Tweet]:
    session_factory, _ = create_session_factory(database_url)
    with session_factory() as session:
        return session.query(Tweet).order_by(Tweet.id).all()


def stratify_if_possible(labels: list[bool]) -> list[bool] | None:
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    return labels


def per_class_metrics(test_labels, predictions) -> dict[str, dict[str, float]]:
    """Précision, rappel et F1 pour chaque classe via ``classification_report``."""
    report = classification_report(
        test_labels,
        predictions,
        labels=CONFUSION_LABELS,
        target_names=CONFUSION_LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {}
    for name in CONFUSION_LABEL_NAMES:
        entry = report[name]
        metrics[name] = {
            "precision": round(float(entry["precision"]), 4),
            "recall": round(float(entry["recall"]), 4),
            "f1_score": round(float(entry["f1-score"]), 4),
            "support": int(entry["support"]),
        }
    return metrics


def detect_warnings(test_labels, accuracy: float) -> list[str]:
    """Signale les configurations où les métriques ne sont pas significatives."""
    warnings: list[str] = []
    label_counts = Counter(test_labels)
    if len(label_counts) < 2:
        warnings.append(
            "Le jeu de test ne contient qu'une seule classe : les métriques ne sont "
            "pas représentatives (jeu de données trop petit ou trop déséquilibré)."
        )
    if len(test_labels) < 10:
        warnings.append(
            f"Le jeu de test ne compte que {len(test_labels)} exemples : les métriques "
            "sont fragiles et à interpréter avec prudence."
        )
    if accuracy == 1.0:
        warnings.append(
            "Aucune erreur sur le jeu de test (métriques parfaites) : le corpus est "
            "probablement trop facile ou trop gabarité pour être représentatif de "
            "tweets réels. Résultat à consolider avec davantage de données variées."
        )
    return warnings


def render_confusion_matrix_png(
    values: list[list[int]],
    title: str,
    destination: Path,
) -> Path:
    """Enregistre une matrice de confusion annotée au format PNG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(values, cmap="Blues")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Prédiction du modèle", fontsize=11)
    ax.set_ylabel("Valeur réelle", fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["absent", "présent"])
    ax.set_yticklabels(["absent", "présent"])

    # Annotation de chaque cellule, avec une couleur de texte contrastée.
    maximum = max((max(row) for row in values), default=0)
    for row in range(2):
        for column in range(2):
            count = values[row][column]
            color = "white" if count > maximum / 2 else "#0f172a"
            ax.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                color=color,
                fontsize=16,
                fontweight="bold",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    return destination


def evaluate_classifier(
    texts: list[str],
    labels: list[bool],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, object]:
    if len(set(labels)) < 2:
        raise ValueError("Chaque label doit contenir au moins deux classes.")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_if_possible(labels),
    )
    model = train_classifier(train_texts, train_labels)
    predictions = model.predict(test_texts)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    accuracy = round(float(accuracy_score(test_labels, predictions)), 4)

    return {
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "accuracy": accuracy,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "per_class": per_class_metrics(test_labels, predictions),
        "confusion_matrix": {
            "labels": CONFUSION_LABEL_NAMES,
            "values": confusion_matrix(
                test_labels,
                predictions,
                labels=CONFUSION_LABELS,
            ).tolist(),
        },
        "warnings": detect_warnings(test_labels, accuracy),
    }


def build_evaluation_pool(
    database_url: str,
    supplementary_path: str | None = None,
) -> tuple[list[str], list[bool], list[bool], dict[str, object]]:
    """Construit le pool d'évaluation : tweets de la base dédupliqués par texte,
    complétés par un jeu de tweets réalistes supplémentaires.

    La déduplication évite qu'un même texte se retrouve à la fois dans le train et
    le test (fuite de données), ce qui gonfle artificiellement les métriques.
    """
    rows = load_training_rows(database_url)
    pool: dict[str, tuple[bool, bool]] = {}
    for tweet in rows:
        pool.setdefault(tweet.text, (bool(tweet.positive), bool(tweet.negative)))
    database_unique = len(pool)
    duplicates_removed = len(rows) - database_unique

    supplement_added = 0
    if supplementary_path and Path(supplementary_path).exists():
        for row in read_tweet_rows(supplementary_path):
            if row["text"] not in pool:
                pool[row["text"]] = (bool(row["positive"]), bool(row["negative"]))
                supplement_added += 1

    texts = list(pool.keys())
    positives = [pool[text][0] for text in texts]
    negatives = [pool[text][1] for text in texts]

    stats = {
        "database_rows": len(rows),
        "database_unique": database_unique,
        "duplicates_removed": duplicates_removed,
        "supplement_added": supplement_added,
    }
    return texts, positives, negatives, stats


def assess_dataset(
    positives: list[bool],
    negatives: list[bool],
    stats: dict[str, object],
) -> dict[str, object]:
    """Décrit la taille et l'équilibre du pool d'évaluation pour le rapport."""
    total = len(positives)
    positive = sum(positives)
    negative = sum(negatives)
    neutral = sum(
        1 for pos, neg in zip(positives, negatives, strict=True) if not pos and not neg
    )
    sufficient = total >= 100
    return {
        "samples": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "sufficient": sufficient,
        "database_rows": stats["database_rows"],
        "duplicates_removed": stats["duplicates_removed"],
        "supplement_added": stats["supplement_added"],
        "note": (
            "Le jeu de données est suffisamment volumineux pour une évaluation crédible."
            if sufficient
            else "Le jeu de données est petit : les métriques sont indicatives et devraient "
            "être consolidées avec davantage de tweets annotés."
        ),
    }


def evaluate_from_database(
    database_url: str,
    output_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    figures_dir: str | None = None,
    supplementary_path: str | None = None,
) -> dict[str, object]:
    texts, positives, negatives, stats = build_evaluation_pool(
        database_url, supplementary_path
    )
    if len(texts) < 8:
        raise ValueError("Au moins huit tweets annotés sont nécessaires pour évaluer.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figures_directory = Path(figures_dir) if figures_dir else destination.parent

    columns = {"positive": positives, "negative": negatives}
    labels: dict[str, object] = {}
    for key in ("positive", "negative"):
        metrics = evaluate_classifier(
            texts,
            columns[key],
            test_size=test_size,
            random_state=random_state,
        )
        figure_path = render_confusion_matrix_png(
            metrics["confusion_matrix"]["values"],
            f"Matrice de confusion — {CLASSIFIER_TITLES[key]}",
            figures_directory / FIGURE_FILENAMES[key],
        )
        # On stocke le nom de fichier, résolu par rapport au dossier du JSON.
        metrics["figure"] = figure_path.name
        labels[key] = metrics

    evaluation = {
        "samples": len(texts),
        "test_size": test_size,
        "random_state": random_state,
        "dataset": assess_dataset(positives, negatives, stats),
        "labels": labels,
    }

    destination.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return evaluation


def print_summary(evaluation: dict[str, object]) -> None:
    dataset = evaluation["dataset"]
    print(
        f"Évaluation sur {evaluation['samples']} tweets uniques "
        f"({dataset['positive']} positifs, {dataset['negative']} négatifs, "
        f"{dataset['neutral']} neutres)."
    )
    print(
        f"  base: {dataset['database_rows']} lignes, "
        f"{dataset['duplicates_removed']} doublons retirés, "
        f"{dataset['supplement_added']} tweets réalistes ajoutés."
    )
    for key in ("positive", "negative"):
        metrics = evaluation["labels"][key]
        print(
            f"[{CLASSIFIER_TITLES[key]}] accuracy={metrics['accuracy']} "
            f"precision={metrics['precision']} recall={metrics['recall']} "
            f"f1={metrics['f1_score']}"
        )
        for warning in metrics["warnings"]:
            print(f"  ! {warning}")


if __name__ == "__main__":
    result = evaluate_from_database(
        Config.DATABASE_URL,
        Config.EVALUATION_PATH,
        supplementary_path=Config.SUPPLEMENTARY_DATASET_PATH,
    )
    print_summary(result)
