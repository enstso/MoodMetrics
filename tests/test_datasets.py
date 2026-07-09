import csv

import pytest
from sqlalchemy import select

from moodmetrics.database import Tweet, create_session_factory
from moodmetrics.datasets import import_tweet_rows, read_tweet_rows


def write_dataset(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "positive", "negative"])
        writer.writerows(rows)


def test_read_tweet_rows_validates_and_casts_labels(tmp_path):
    dataset_path = tmp_path / "tweets.csv"
    write_dataset(
        dataset_path,
        [
            ["Très bon service", "1", "0"],
            ["Service lent", "false", "true"],
        ],
    )

    assert read_tweet_rows(dataset_path) == [
        {"text": "Très bon service", "positive": True, "negative": False},
        {"text": "Service lent", "positive": False, "negative": True},
    ]


def test_read_tweet_rows_rejects_invalid_labels(tmp_path):
    dataset_path = tmp_path / "tweets.csv"
    write_dataset(dataset_path, [["Tweet ambigu", "1", "1"]])

    with pytest.raises(ValueError, match="positif et négatif"):
        read_tweet_rows(dataset_path)


def test_import_tweet_rows_adds_missing_rows_without_duplicates(tmp_path):
    dataset_path = tmp_path / "tweets.csv"
    database_url = f"sqlite:///{tmp_path / 'tweets.db'}"
    write_dataset(
        dataset_path,
        [
            ["Très bon service", "1", "0"],
            ["Service lent", "0", "1"],
        ],
    )

    assert import_tweet_rows(database_url, dataset_path) == 2
    assert import_tweet_rows(database_url, dataset_path) == 0

    session_factory, _ = create_session_factory(database_url)
    with session_factory() as session:
        tweets = session.scalars(select(Tweet).order_by(Tweet.id)).all()

    assert [tweet.text for tweet in tweets] == ["Très bon service", "Service lent"]
