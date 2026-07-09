import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from moodmetrics.config import Config
from moodmetrics.database import Tweet, create_session_factory
from scripts.train_model import train_classifier


CONFUSION_LABELS = [False, True]
CONFUSION_LABEL_NAMES = ["absent", "present"]


def load_training_rows(database_url: str) -> list[Tweet]:
    session_factory, _ = create_session_factory(database_url)
    with session_factory() as session:
        return session.query(Tweet).order_by(Tweet.id).all()


def stratify_if_possible(labels: list[bool]) -> list[bool] | None:
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    return labels


def evaluate_classifier(
    texts: list[str],
    labels: list[bool],
    test_size: float = 0.25,
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

    return {
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "confusion_matrix": {
            "labels": CONFUSION_LABEL_NAMES,
            "values": confusion_matrix(
                test_labels,
                predictions,
                labels=CONFUSION_LABELS,
            ).tolist(),
        },
    }


def evaluate_from_database(
    database_url: str,
    output_path: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, object]:
    tweets = load_training_rows(database_url)
    if len(tweets) < 8:
        raise ValueError("Au moins huit tweets annotés sont nécessaires pour évaluer.")

    texts = [tweet.text for tweet in tweets]
    evaluation = {
        "samples": len(tweets),
        "test_size": test_size,
        "random_state": random_state,
        "labels": {
            "positive": evaluate_classifier(
                texts,
                [tweet.positive for tweet in tweets],
                test_size=test_size,
                random_state=random_state,
            ),
            "negative": evaluate_classifier(
                texts,
                [tweet.negative for tweet in tweets],
                test_size=test_size,
                random_state=random_state,
            ),
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return evaluation


if __name__ == "__main__":
    evaluate_from_database(Config.DATABASE_URL, Config.EVALUATION_PATH)
