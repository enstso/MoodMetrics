from pathlib import Path

import joblib

from moodmetrics.config import Config
from moodmetrics.database import Tweet, create_session_factory
from moodmetrics.modeling import get_model_spec


def train_classifier(texts: list[str], labels: list[bool], model_name: str | None = None):
    if len(set(labels)) < 2:
        raise ValueError("Chaque label doit contenir au moins deux classes.")
    model = get_model_spec(model_name).build()
    return model.fit(texts, labels)


def train_from_database(database_url: str, model_path: str, model_name: str | None = None):
    session_factory, _ = create_session_factory(database_url)
    with session_factory() as session:
        tweets = session.query(Tweet).order_by(Tweet.id).all()

    if len(tweets) < 4:
        raise ValueError("Au moins quatre tweets annotés sont nécessaires.")

    texts = [tweet.text for tweet in tweets]
    spec = get_model_spec(model_name)
    models = {
        "metadata": {"model": spec.to_metadata(), "samples": len(tweets)},
        "positive": train_classifier(texts, [tweet.positive for tweet in tweets], spec.name),
        "negative": train_classifier(texts, [tweet.negative for tweet in tweets], spec.name),
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, destination)


if __name__ == "__main__":
    train_from_database(Config.DATABASE_URL, Config.MODEL_PATH)
