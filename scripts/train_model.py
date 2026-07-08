from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from moodmetrics.config import Config
from moodmetrics.database import Tweet, create_session_factory


def train_classifier(texts: list[str], labels: list[bool]):
    if len(set(labels)) < 2:
        raise ValueError("Chaque label doit contenir au moins deux classes.")
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(max_iter=1_000, class_weight="balanced"),
    )
    return model.fit(texts, labels)


def train_from_database(database_url: str, model_path: str):
    session_factory, _ = create_session_factory(database_url)
    with session_factory() as session:
        tweets = session.query(Tweet).order_by(Tweet.id).all()

    if len(tweets) < 4:
        raise ValueError("Au moins quatre tweets annotés sont nécessaires.")

    texts = [tweet.text for tweet in tweets]
    models = {
        "positive": train_classifier(texts, [tweet.positive for tweet in tweets]),
        "negative": train_classifier(texts, [tweet.negative for tweet in tweets]),
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, destination)


if __name__ == "__main__":
    train_from_database(Config.DATABASE_URL, Config.MODEL_PATH)

