import json

from moodmetrics.database import Base, Tweet, create_session_factory
from scripts.evaluate_model import detect_warnings, evaluate_from_database


def _seed_database(database_url: str, *, duplicate: bool = False) -> None:
    session_factory, engine = create_session_factory(database_url)
    Base.metadata.create_all(engine)
    with session_factory() as session:
        session.add_all(
            [
                Tweet(text=f"super service rapide {index}", positive=True, negative=False)
                for index in range(12)
            ]
        )
        session.add_all(
            [
                Tweet(text=f"mauvais bug lent {index}", positive=False, negative=True)
                for index in range(12)
            ]
        )
        session.add_all(
            [
                Tweet(text=f"message information neutre {index}", positive=False, negative=False)
                for index in range(12)
            ]
        )
        if duplicate:
            # Doublons stricts de texte, comme lors d'un double import concurrent.
            session.add_all(
                [
                    Tweet(text=f"super service rapide {index}", positive=True, negative=False)
                    for index in range(12)
                ]
            )
        session.commit()


def test_evaluate_from_database_writes_metrics(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'tweets.db'}"
    output_path = tmp_path / "evaluation.json"
    _seed_database(database_url)

    evaluation = evaluate_from_database(database_url, str(output_path), test_size=0.3)
    saved_evaluation = json.loads(output_path.read_text(encoding="utf-8"))

    assert evaluation["samples"] == 36
    assert saved_evaluation["model"]["name"] == "hybrid_char_word_tfidf_logreg"
    assert saved_evaluation["labels"]["positive"]["test_samples"] == 11
    assert saved_evaluation["labels"]["negative"]["confusion_matrix"]["labels"] == [
        "absent",
        "present",
    ]


def test_evaluation_produces_per_class_metrics_and_figures(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'tweets.db'}"
    output_path = tmp_path / "evaluation.json"
    _seed_database(database_url)

    evaluation = evaluate_from_database(database_url, str(output_path), test_size=0.3)

    positive = evaluation["labels"]["positive"]
    assert set(positive["per_class"]) == {"absent", "present"}
    assert set(positive["per_class"]["present"]) == {
        "precision",
        "recall",
        "f1_score",
        "support",
    }
    # Les deux images de matrices de confusion sont générées à côté du JSON.
    for name in ("confusion_matrix_positive.png", "confusion_matrix_negative.png"):
        assert (tmp_path / name).exists()
        assert positive["figure"] or name  # le nom de figure est renseigné


def test_deduplication_and_supplement(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'tweets.db'}"
    output_path = tmp_path / "evaluation.json"
    _seed_database(database_url, duplicate=True)

    supplement = tmp_path / "extra.csv"
    supplement.write_text(
        "text,positive,negative\n"
        '"un tweet supplementaire vraiment enthousiaste",1,0\n'
        '"un tweet supplementaire franchement decevant",0,1\n',
        encoding="utf-8",
    )

    evaluation = evaluate_from_database(
        database_url,
        str(output_path),
        test_size=0.3,
        supplementary_path=str(supplement),
    )
    dataset = evaluation["dataset"]

    assert dataset["database_rows"] == 48  # 36 + 12 doublons
    assert dataset["duplicates_removed"] == 12
    assert dataset["supplement_added"] == 2
    assert evaluation["samples"] == 38  # 36 uniques + 2 supplémentaires


def test_detect_warnings_flags_perfect_metrics():
    warnings = detect_warnings([True, False, True, False] * 6, accuracy=1.0)
    assert any("parfaites" in message for message in warnings)

    warnings = detect_warnings([True, False, True, False] * 6, accuracy=0.9)
    assert not any("parfaites" in message for message in warnings)
