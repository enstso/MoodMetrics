import csv
from pathlib import Path

from sqlalchemy import select

from moodmetrics.database import Base, Tweet, create_session_factory


EXPECTED_COLUMNS = {"text", "positive", "negative"}
TRUE_VALUES = {"1", "true", "yes", "oui"}
FALSE_VALUES = {"0", "false", "no", "non"}


def parse_binary_label(value: str, line_number: int, column: str) -> bool:
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"Ligne {line_number}: la colonne {column} doit valoir 0/1 ou true/false."
    )


def read_tweet_rows(dataset_path: str | Path) -> list[dict[str, object]]:
    path = Path(dataset_path)
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if set(reader.fieldnames or []) != EXPECTED_COLUMNS:
            columns = ", ".join(sorted(EXPECTED_COLUMNS))
            raise ValueError(f"Le dataset doit contenir exactement les colonnes: {columns}.")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            text = (row["text"] or "").strip()
            if not text:
                raise ValueError(f"Ligne {line_number}: le tweet ne peut pas être vide.")

            positive = parse_binary_label(row["positive"], line_number, "positive")
            negative = parse_binary_label(row["negative"], line_number, "negative")
            if positive and negative:
                raise ValueError(
                    f"Ligne {line_number}: un tweet ne peut pas être positif et négatif."
                )

            rows.append({"text": text, "positive": positive, "negative": negative})

    return rows


def import_tweet_rows(database_url: str, dataset_path: str | Path, replace: bool = False) -> int:
    rows = read_tweet_rows(dataset_path)
    session_factory, engine = create_session_factory(database_url)
    Base.metadata.create_all(engine)

    with session_factory() as session:
        if replace:
            session.query(Tweet).delete()
            existing_texts = set()
        else:
            existing_texts = set(session.scalars(select(Tweet.text)).all())

        tweets = [
            Tweet(
                text=str(row["text"]),
                positive=bool(row["positive"]),
                negative=bool(row["negative"]),
            )
            for row in rows
            if row["text"] not in existing_texts
        ]
        session.add_all(tweets)
        session.commit()
        return len(tweets)
